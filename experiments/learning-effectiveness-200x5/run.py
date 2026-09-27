"""Run all prescribed 200x5 searches; --resume requires unchanged protocol/source/tools."""
import argparse
from pathlib import Path
import platform
import subprocess
import sys
import traceback

from filelock import FileLock
import torch
import yaml

from common import (ROOT, HERE, GROUPS, ORIGINAL, DiagnosticA2C, config, dump, group_config,
                    identity, load, now, sha, state_hash, supervise, EVAL_SEEDS)
from drills.experiment import run_baselines, verify_netlist


def run_task(cfg, group, name, seed, fingerprint, resume):
    cfg = group_config(cfg, group, fingerprint)
    folder = Path(cfg['runtime']['output_dir']) / name / f'seed-{seed}'
    if folder.exists() and any(folder.iterdir()) and not resume:
        raise FileExistsError(folder)
    restoring = resume and (folder / 'checkpoint.pt').exists()
    agent = DiagnosticA2C(cfg, cfg['protocol']['circuits'][name], seed, folder, restoring)
    if resume and (folder / 'result.json').exists():
        saved = __import__('json').loads((folder / 'result.json').read_text())
        if saved['status'] == 'complete':
            print(f'Already complete: {group}/{name}/{seed}', flush=True)
            return
    try:
        for _ in range(agent.episodes_completed, cfg['protocol']['episodes']):
            reward = agent.run_episode()
            dump(folder / 'status.json', dict(status='running', episodes=agent.episodes_completed, time=now()))
            print(f"{group}/{name}/seed-{seed}: episode {agent.episodes_completed}/{cfg['protocol']['episodes']} reward={reward}", flush=True)
        verify_netlist(cfg, cfg['protocol']['circuits'][name], folder / 'best-mapped.v', agent.game.best)
        initial, final = load(folder / 'snapshots/0.pt'), load(agent.checkpoint)
        dump(folder / 'parameter-validation.json', dict(initial_network=state_hash(initial['network']),
            final_network=state_hash(final['network']), initial_optimizer=state_hash(initial['optimizer']),
            final_optimizer=state_hash(final['optimizer']), rng_advanced=not torch.equal(initial['rng_state'], final['rng_state']),
            fingerprint=fingerprint, learning_enabled=cfg['method']['learning_enabled']))
        dump(folder / 'result.json', dict(group=group, circuit=name, seed=seed, status='complete',
            episodes_completed=agent.episodes_completed, best=agent.game.best,
            rewards=agent.rewards, training_seconds=agent.training_seconds, config=cfg))
        dump(folder / 'status.json', dict(status='complete', episodes=agent.episodes_completed, time=now()))
    except Exception as error:
        dump(folder / 'status.json', dict(status='failed', episodes=agent.episodes_completed,
                                         error=repr(error), traceback=traceback.format_exc(), time=now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--task', nargs=3, metavar=('GROUP', 'CIRCUIT', 'SEED'))
    args = parser.parse_args()
    cfg = config()
    fingerprint, evidence = identity(cfg)
    if args.task:
        group, name, seed = args.task
        if group not in GROUPS or name not in cfg['protocol']['circuits'] or int(seed) not in cfg['protocol']['seeds']:
            raise ValueError('Task outside prescribed protocol.')
        run_task(cfg, group, name, int(seed), fingerprint, args.resume)
        return
    root = Path(cfg['runtime']['output_dir'])
    manifest_file = root / 'experiment.json'
    if args.resume and not manifest_file.exists():
        raise FileNotFoundError('--resume requires an existing experiment manifest.')
    if root.exists():
        if not args.resume or not manifest_file.exists():
            raise FileExistsError(f'Use --resume only for identical existing evidence: {root}')
        import json
        if json.loads(manifest_file.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Resume requires identical sources, configuration, dependencies and tool/data hashes.')
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / '.suite.lock'), timeout=0):
        manifest = dict(fingerprint=fingerprint, **evidence, original_algorithm_commit=ORIGINAL,
            branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
            implementation_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            platform=platform.platform(), command=[sys.executable, '-B', '-u', str(HERE / 'run.py'), *sys.argv[1:]],
            started_at=now(), groups=list(GROUPS), evaluation_seeds=EVAL_SEEDS,
            tools_versions={k: subprocess.check_output([cfg['runtime'][k], *options], text=True).strip()
                for k, options in [('abc_binary', ['-c', 'version']), ('yosys_binary', ['-V'])]})
        if args.resume:
            manifest['previous_invocation'] = json.loads(manifest_file.read_text())
        dump(manifest_file, manifest)
        if not (root / 'baseline.json').exists():
            run_baselines(cfg)
        tasks = []
        for group in GROUPS:
            for name in cfg['protocol']['circuits']:
                for seed in cfg['protocol']['seeds']:
                    command = [sys.executable, '-B', '-u', str(HERE / 'run.py'), '--task', group, name, str(seed)]
                    if args.resume:
                        command.append('--resume')
                    tasks.append((f'{group}-{name}-{seed}', command))
        manifest['processes'] = supervise(tasks, root / 'logs/training', args.resume)
        manifest.update(finished_at=now(), status='complete' if all(p['exit_code'] == 0 for p in manifest['processes']) else 'incomplete')
        dump(manifest_file, manifest)
        dump(HERE / 'provenance.json', manifest)
        print(f'Training status: {manifest["status"]}', flush=True)
        if manifest['status'] != 'complete':
            from analysis import generate
            generate(cfg)
            raise SystemExit(1)


if __name__ == '__main__':
    main()
