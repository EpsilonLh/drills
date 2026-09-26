"""Independent no-update rollouts, then validation/report regeneration (--report-only)."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import traceback

from filelock import FileLock
import numpy as np
import torch

from common import (ROOT, HERE, GROUPS, POLICIES, EVAL_SEEDS, SNAPSHOTS, ActorCritic, Normalizer,
                    config, dump, identity, load, now, sha, state_hash, supervise)
from drills.fpga_session import FPGASession
from drills.experiment import verify_netlist


def read_json(path):
    return json.loads(Path(path).read_text())


def checkpoint_path(root, policy, name, seed):
    group = 'uniform' if policy == 'uniform' else 'trained'
    episode = int(policy.split('-')[1]) if policy.startswith('trained-') else 0
    return root / 'training' / group / name / f'seed-{seed}/snapshots/{episode}.pt'


def inference_rollout(cfg, circuit, network, evaluation_seed, folder, uniform=False):
    """Uses its own game/normalizer/generator; never receives a training agent."""
    game = FPGASession(cfg, circuit, folder)
    generator = torch.Generator(device='cpu').manual_seed(evaluation_seed)
    state = game.reset()
    normalizer = Normalizer(len(state), cfg['method']['normalization'])
    actions, rewards = [], []
    done = False
    while not done:
        state = normalizer.normalize(state)
        with torch.no_grad():
            logits, _ = network(torch.as_tensor(state, device='cpu'))
            probs = logits.softmax(-1) if not uniform else torch.full_like(logits, 1 / len(logits))
            action = torch.multinomial(probs, 1, generator=generator).item()
        state, reward, done = game.step(action)
        actions.append(action)
        rewards.append(reward)
    game.export_best()
    verify_netlist(cfg, circuit, Path(folder) / 'best-mapped.v', game.best)
    return dict(evaluation_seed=evaluation_seed, actions=actions, rewards=rewards,
                best=game.best, terminal=dict(luts=game.luts, levels=game.levels,
                                              feasible=game.levels <= circuit['max_levels']))


def training_validation(cfg):
    root = Path(cfg['runtime']['output_dir'])
    result = dict(runs=0, first_episodes_equal=True, initial_states_equal=True, frozen_unchanged=True,
                  frozen_snapshot_alias=True, trained_changed=True, checks=[], errors=[])
    for name in cfg['protocol']['circuits']:
        for seed in cfg['protocol']['seeds']:
            initial_states = []
            for group in GROUPS:
                folder = root / 'training' / group / name / f'seed-{seed}'
                if not (folder / 'result.json').exists():
                    result['errors'].append(f'Missing {group}/{name}/{seed}')
                    continue
                row = read_json(folder / 'result.json')
                if row['status'] != 'complete' or row['episodes_completed'] != cfg['protocol']['episodes']:
                    result['errors'].append(f'Incomplete {group}/{name}/{seed}')
                    continue
                result['runs'] += 1
                initial = load(folder / 'snapshots/0.pt')
                final = load(folder / 'checkpoint.pt')
                initial_states.append(tuple(state_hash(initial[k]) for k in ('network', 'optimizer', 'rng_state')))
                changed = state_hash(initial['network']) != state_hash(final['network'])
                check = dict(group=group, circuit=name, seed=seed, changed=changed,
                             rng_advanced=not torch.equal(initial['rng_state'], final['rng_state']))
                if group != 'trained':
                    check['optimizer_unchanged'] = state_hash(initial['optimizer']) == state_hash(final['optimizer'])
                    result['frozen_unchanged'] &= not changed and check['optimizer_unchanged']
                    for episode in SNAPSHOTS:
                        saved = load(folder / f'snapshots/{episode}.pt')
                        result['frozen_snapshot_alias'] &= state_hash(saved['network']) == state_hash(initial['network'])
                else:
                    result['trained_changed'] &= changed
                # All hashes exclude elapsed timings; checkpoint bytes are also pinned separately.
                check['snapshot_sha256'] = {str(e): sha(folder / f'snapshots/{e}.pt') for e in SNAPSHOTS}
                result['checks'].append(check)
            if len(initial_states) == 3:
                result['initial_states_equal'] &= len(set(initial_states)) == 1
                a = root / 'training/trained' / name / f'seed-{seed}/episodes/1/log.csv'
                b = root / 'training/frozen' / name / f'seed-{seed}/episodes/1/log.csv'
                result['first_episodes_equal'] &= a.read_bytes() == b.read_bytes()
    if result['errors'] or not all(result[k] for k in ('first_episodes_equal', 'initial_states_equal',
                                                      'frozen_unchanged', 'frozen_snapshot_alias')):
        raise ValueError(f'Unfair or incomplete experiment: {result}')
    return result


def evaluate_task(cfg, policy, name, seed, resume):
    root = Path(cfg['runtime']['output_dir'])
    folder = root / 'evaluation' / policy / name / f'seed-{seed}'
    if folder.exists() and any(folder.iterdir()) and not resume:
        raise FileExistsError(folder)
    checkpoint = checkpoint_path(root, policy, name, seed)
    digest = sha(checkpoint)
    saved = load(checkpoint)
    torch.set_num_threads(cfg['environment']['torch_threads'])
    # Constructing an inference network does not advance the caller's global RNG either.
    with torch.random.fork_rng(devices=[]):
        network = ActorCritic(len(cfg['method']['features']), len(cfg['protocol']['actions']), cfg['method']['network'])
    network.load_state_dict(saved['network'])
    network.eval()
    network.requires_grad_(False)
    initial_hash = state_hash(network.state_dict())
    rows = []
    try:
        for evaluation_seed in EVAL_SEEDS:
            destination = folder / f'rollout-{evaluation_seed}'
            record = destination / 'rollout.json'
            if resume and record.exists():
                row = read_json(record)
                if row['checkpoint_sha256'] != digest or row['status'] != 'complete':
                    raise ValueError('Cannot reuse rollout from a different checkpoint.')
            else:
                row = inference_rollout(cfg, cfg['protocol']['circuits'][name], network,
                                        evaluation_seed, destination, policy == 'uniform')
                row.update(policy=policy, circuit=name, training_seed=seed, checkpoint_sha256=digest, status='complete')
                dump(record, row)
            rows.append(row)
            dump(folder / 'status.json', dict(status='running', completed=len(rows), time=now()))
            print(f'{policy}/{name}/{seed}: rollout {len(rows)}/30 best={row["best"]["luts"]}', flush=True)
        if initial_hash != state_hash(network.state_dict()) or sha(checkpoint) != digest:
            raise RuntimeError('Evaluation changed weights or the training checkpoint.')
        dump(folder / 'result.json', dict(status='complete', policy=policy, circuit=name, training_seed=seed,
            checkpoint_sha256=digest, network_unchanged=True, checkpoint_unchanged=True, rollouts=rows))
        dump(folder / 'status.json', dict(status='complete', completed=len(rows), time=now()))
    except Exception as error:
        dump(folder / 'status.json', dict(status='failed', completed=len(rows), error=repr(error),
                                         traceback=traceback.format_exc(), time=now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--task', nargs=3)
    args = parser.parse_args()
    cfg = config()
    root = Path(cfg['runtime']['output_dir'])
    fingerprint, _ = identity(cfg)
    training_manifest = read_json(root / 'experiment.json')
    if training_manifest['fingerprint'] != fingerprint:
        raise ValueError('Training source/configuration/tool fingerprint changed.')
    if args.task:
        policy, name, seed = args.task
        if policy not in POLICIES or name not in cfg['protocol']['circuits'] or int(seed) not in cfg['protocol']['seeds']:
            raise ValueError('Evaluation outside prescribed protocol.')
        evaluate_task(cfg, policy, name, int(seed), args.resume)
        return
    if args.report_only:
        from analysis import generate
        generate(cfg)
        return
    try:
        validation = training_validation(cfg)
    except Exception:
        from analysis import generate
        generate(cfg)
        raise
    evaluator_sources = {str(p.relative_to(ROOT)): sha(p) for p in (HERE / 'evaluate.py', HERE / 'common.py')}
    target = root / 'evaluation.json'
    if target.exists() and (not args.resume or read_json(target)['sources'] != evaluator_sources):
        raise ValueError('Use --resume with the identical evaluator.')
    with FileLock(str(root / '.suite.lock'), timeout=0):
        manifest = dict(command=[sys.executable, '-B', '-u', str(HERE / 'evaluate.py'), *sys.argv[1:]],
                        started_at=now(), sources=evaluator_sources, training_fingerprint=fingerprint,
                        training_validation=validation, evaluation_seeds=EVAL_SEEDS, policies=list(POLICIES))
        if args.resume:
            manifest['previous_invocation'] = read_json(target)
        dump(target, manifest)
        tasks = []
        for policy in POLICIES:
            for name in cfg['protocol']['circuits']:
                for seed in cfg['protocol']['seeds']:
                    command = [sys.executable, '-B', '-u', str(HERE / 'evaluate.py'), '--task', policy, name, str(seed)]
                    if args.resume:
                        command.append('--resume')
                    tasks.append((f'{policy}-{name}-{seed}', command))
        manifest['processes'] = supervise(tasks, root / 'logs/evaluation', args.resume)
        manifest.update(finished_at=now(), status='complete' if all(p['exit_code'] == 0 for p in manifest['processes']) else 'incomplete')
        dump(target, manifest)
        dump(HERE / 'evaluation-provenance.json', manifest)
        from analysis import generate
        generate(cfg)
        if manifest['status'] != 'complete':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
