"""Frozen inference banks with paired sampling seeds and nested budget prefixes."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import traceback

from filelock import FileLock
import torch

from common import (ROOT, HERE, GROUPS, POLICIES, EVAL_SEEDS, EVAL_PREFIXES, SNAPSHOTS,
                    ActorCritic, Normalizer, config, dump, identity, load, now, sha,
                    state_hash, supervise)
from drills.fpga_session import FPGASession
from drills.experiment import verify_netlist


def read(path):
    return json.loads(Path(path).read_text())


def physical_seeds(cfg, policy):
    return cfg['protocol']['seeds'][:1] if policy == 'uniform' else cfg['protocol']['seeds']


def checkpoint_path(root, policy, name, seed):
    group = 'uniform' if policy == 'uniform' else 'trained'
    episode = int(policy.split('-')[1]) if policy.startswith('trained-') else 0
    return root / 'training' / group / name / f'seed-{seed}/snapshots/{episode}.pt'


def inference_rollout(cfg, circuit, network, evaluation_seed, folder, uniform=False):
    game = FPGASession(cfg, circuit, folder)
    generator = torch.Generator(device='cpu').manual_seed(evaluation_seed)
    state = game.reset()
    normalizer = Normalizer(len(state), cfg['method']['normalization'])
    actions, rewards, prefixes = [], [], []
    done = False
    while not done:
        state = normalizer.normalize(state)
        with torch.no_grad():
            logits, _ = network(torch.as_tensor(state, device='cpu'))
            probs = torch.full_like(logits, 1 / len(logits)) if uniform else logits.softmax(-1)
            action = torch.multinomial(probs, 1, generator=generator).item()
        state, reward, done = game.step(action)
        actions.append(action)
        rewards.append(reward)
        if game.iteration in EVAL_PREFIXES:
            prefixes.append(dict(steps=game.iteration, best=copy.deepcopy(game.best),
                terminal=dict(luts=game.luts, levels=game.levels,
                              feasible=game.levels <= circuit['max_levels']), reward=sum(rewards)))
    game.export_best()
    verify_netlist(cfg, circuit, Path(folder) / 'best-mapped.v', game.best)
    return dict(evaluation_seed=evaluation_seed, actions=actions, rewards=rewards,
        prefixes=prefixes, best=game.best, terminal=dict(luts=game.luts, levels=game.levels,
                                                      feasible=game.levels <= circuit['max_levels']))


def training_validation(cfg):
    root = Path(cfg['runtime']['output_dir'])
    old = ROOT / 'results/learning-effectiveness/training'
    result = dict(runs=0, complete_runs=0, first_episodes_equal=True, initial_states_equal=True,
        old_initial_states_equal=True, old_first_10_equal=True, frozen_unchanged=True,
        frozen_snapshot_alias=True, trained_changed=True, checks=[], errors=[])
    for name in cfg['protocol']['circuits']:
        for seed in cfg['protocol']['seeds']:
            initial_states = []
            for group in GROUPS:
                folder = root / 'training' / group / name / f'seed-{seed}'
                if not (folder / 'checkpoint.pt').exists():
                    result['errors'].append(f'Missing {group}/{name}/{seed}')
                    continue
                final = load(folder / 'checkpoint.pt')
                row = read(folder / 'result.json') if (folder / 'result.json').exists() else dict(
                    status='incomplete',episodes_completed=final['episodes_completed'])
                complete = row['status']=='complete' and row['episodes_completed']==cfg['protocol']['episodes']
                if not complete:
                    result['errors'].append(f'Incomplete {group}/{name}/{seed}')
                if final['episodes_completed'] < 1:
                    continue
                result['runs'] += 1
                result['complete_runs'] += int(complete)
                initial = load(folder / 'snapshots/0.pt')
                initial_states.append(tuple(state_hash(initial[k]) for k in ('network','optimizer','rng_state')))
                previous = load(old / group / name / f'seed-{seed}/snapshots/0.pt')
                result['old_initial_states_equal'] &= all(state_hash(initial[k]) == state_hash(previous[k])
                                                         for k in ('network','optimizer','rng_state'))
                with (folder / 'episodes/1/log.csv').open() as stream:
                    current_log = list(csv.DictReader(stream))[:11]
                with (old / group / name / f'seed-{seed}/episodes/1/log.csv').open() as stream:
                    previous_log = list(csv.DictReader(stream))
                result['old_first_10_equal'] &= current_log == previous_log
                changed = state_hash(initial['network']) != state_hash(final['network'])
                saved_snapshots = [e for e in SNAPSHOTS if (folder / f'snapshots/{e}.pt').exists()]
                if any(e not in saved_snapshots for e in SNAPSHOTS if e<=final['episodes_completed']):
                    raise ValueError(f'Missing committed milestone snapshot: {folder}')
                check = dict(group=group, circuit=name, seed=seed, changed=changed,
                    status='complete' if complete else 'incomplete',episodes_completed=final['episodes_completed'],
                    checkpoint_sha256=sha(folder / 'checkpoint.pt'),
                    rng_advanced=not torch.equal(initial['rng_state'],final['rng_state']),
                    snapshot_sha256={str(e):sha(folder / f'snapshots/{e}.pt') for e in saved_snapshots},
                    missing_snapshots=[e for e in SNAPSHOTS if e not in saved_snapshots])
                if group != 'trained':
                    check['optimizer_unchanged'] = state_hash(initial['optimizer']) == state_hash(final['optimizer'])
                    result['frozen_unchanged'] &= not changed and check['optimizer_unchanged']
                    result['frozen_snapshot_alias'] &= all(state_hash(load(folder / f'snapshots/{e}.pt')['network'])
                                                          == state_hash(initial['network']) for e in saved_snapshots)
                else:
                    result['trained_changed'] &= changed
                result['checks'].append(check)
            if len(initial_states) == len(GROUPS):
                result['initial_states_equal'] &= len(set(initial_states)) == 1
                a = root / 'training/trained' / name / f'seed-{seed}/episodes/1/log.csv'
                b = root / 'training/frozen' / name / f'seed-{seed}/episodes/1/log.csv'
                result['first_episodes_equal'] &= a.read_bytes() == b.read_bytes()
    invariants = ('first_episodes_equal','initial_states_equal','old_initial_states_equal',
                  'old_first_10_equal','frozen_unchanged','frozen_snapshot_alias')
    if any(not result[k] for k in invariants):
        raise ValueError(f'Training validation invariant failed: {result}')
    return result


def evaluate_task(cfg, policy, name, seed, resume, fingerprint):
    root = Path(cfg['runtime']['output_dir'])
    folder = root / 'evaluation' / policy / name / f'seed-{seed}'
    if folder.exists() and any(folder.iterdir()) and not resume:
        raise FileExistsError(folder)
    checkpoint = checkpoint_path(root,policy,name,seed)
    digest, saved = sha(checkpoint), load(checkpoint)
    if saved['experiment_fingerprint'] != fingerprint or saved['experiment_group'] != ('uniform' if policy=='uniform' else 'trained'):
        raise ValueError('Evaluation checkpoint belongs to a different experiment/group.')
    torch.set_num_threads(cfg['environment']['torch_threads'])
    with torch.random.fork_rng(devices=[]):
        network = ActorCritic(len(cfg['method']['features']),len(cfg['protocol']['actions']),cfg['method']['network'])
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
                row = read(record)
                if row['checkpoint_sha256'] != digest or row['status'] != 'complete':
                    raise ValueError('Cannot reuse a rollout from another checkpoint.')
            else:
                row = inference_rollout(cfg,cfg['protocol']['circuits'][name],network,evaluation_seed,destination,policy=='uniform')
                row.update(policy=policy,circuit=name,training_seed=seed,checkpoint_sha256=digest,status='complete')
                dump(record,row)
            rows.append(row)
            dump(folder / 'status.json',dict(status='running',completed=len(rows),time=now()))
            print(f'{policy}/{name}/{seed}: rollout {len(rows)}/{len(EVAL_SEEDS)} best={row["best"]["luts"]}',flush=True)
        if initial_hash != state_hash(network.state_dict()) or sha(checkpoint) != digest:
            raise RuntimeError('Evaluation changed weights or the training checkpoint.')
        dump(folder / 'result.json',dict(status='complete',policy=policy,circuit=name,training_seed=seed,
            checkpoint_sha256=digest,network_unchanged=True,checkpoint_unchanged=True,rollouts=rows))
        dump(folder / 'status.json',dict(status='complete',completed=len(rows),time=now()))
    except Exception as error:
        dump(folder / 'status.json',dict(status='failed',completed=len(rows),error=repr(error),
                                       traceback=traceback.format_exc(),time=now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--task',nargs=3)
    args = parser.parse_args()
    cfg = config()
    root = Path(cfg['runtime']['output_dir'])
    fingerprint, _ = identity(cfg)
    manifest = read(root / 'experiment.json')
    if manifest['fingerprint'] != fingerprint:
        raise ValueError('Training source/configuration/tool fingerprint changed.')
    if args.report_only:
        from analysis import generate
        generate(cfg)
        return
    if args.task:
        policy,name,seed = args.task
        if policy not in POLICIES or name not in cfg['protocol']['circuits'] or int(seed) not in physical_seeds(cfg,policy):
            raise ValueError('Evaluation task outside protocol.')
        evaluate_task(cfg,policy,name,int(seed),args.resume,fingerprint)
        return
    validation = training_validation(cfg)
    dump(HERE/'validation.json', validation)
    source_hashes = {str(p.relative_to(ROOT)):sha(p) for p in (HERE/'evaluate.py',HERE/'common.py')}
    target = root / 'evaluation.json'
    if args.resume and not target.exists():
        raise FileNotFoundError('--resume requires an existing evaluation manifest.')
    if target.exists() and (not args.resume or read(target)['sources'] != source_hashes):
        raise ValueError('Use --resume with the identical evaluator.')
    with FileLock(str(root / '.suite.lock'),timeout=0):
        evaluation = dict(command=[sys.executable,'-B','-u',str(HERE/'evaluate.py'),*sys.argv[1:]],
            started_at=now(),sources=source_hashes,training_fingerprint=fingerprint,
            training_validation=validation,evaluation_seeds=EVAL_SEEDS,policies=list(POLICIES),
            prefixes=list(EVAL_PREFIXES),uniform_bank_shared=True)
        if args.resume:
            evaluation['previous_invocation'] = read(target)
        dump(target,evaluation)
        tasks, skipped = [], []
        for policy in POLICIES:
            for name in cfg['protocol']['circuits']:
                for seed in physical_seeds(cfg,policy):
                    command = [sys.executable,'-B','-u',str(HERE/'evaluate.py'),'--task',policy,name,str(seed)]
                    if args.resume:
                        command.append('--resume')
                    if checkpoint_path(root,policy,name,seed).exists():
                        tasks.append((f'{policy}-{name}-{seed}',command))
                    else:
                        skipped.append(dict(policy=policy,circuit=name,seed=seed,reason='Missing checkpoint'))
        evaluation['processes'] = supervise(tasks,root/'logs/evaluation',args.resume)
        evaluation.update(finished_at=now(),skipped=skipped,status='complete' if not skipped and not validation['errors']
                          and all(p['exit_code']==0 for p in evaluation['processes']) else 'incomplete')
        dump(target,evaluation)
        dump(HERE/'evaluation-provenance.json',evaluation)
        from analysis import generate
        generate(cfg)
        if evaluation['status'] != 'complete':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
