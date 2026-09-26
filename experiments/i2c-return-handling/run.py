"""Fixed i2c return ablation; reference is read-only and every new group is isolated."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
from time import perf_counter

import numpy as np
import torch
import yaml
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from drills.experiment import load_config, run_baselines, verify_netlist, write_report
from drills.model import A2C

GROUPS = {'A': ('none-c1', 1.0, True), 'B': ('none-c10', 10.0, True), 'C': ('frozen', 1.0, False)}
REFERENCE_COMMIT = 'f6e829680221133bd9282831dc55e40b2f671ffa'
REFERENCE = ROOT / 'results/change-rewards'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(value):
    if isinstance(value, dict):
        return {str(k): normalized(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalized(v) for v in value]
    return value


def state_hash(value):
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_hashes():
    paths = [ROOT / 'drills.py', *sorted((ROOT / 'drills').glob('*.py')), HERE / 'run.py', HERE / 'worker.py']
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def fingerprint(config, group):
    clean = json.loads(json.dumps(config))
    clean['runtime'].pop('experiment_fingerprint', None)
    return hashlib.sha256(json.dumps(dict(group=group, config=normalized(clean), sources=source_hashes()),
                                    sort_keys=True).encode()).hexdigest()


def check_resume(root, expected, resume):
    if not root.exists():
        return
    if not resume:
        raise FileExistsError(f'Fresh experiment output already exists: {root}')
    filename = root / 'experiment.json'
    if not filename.exists() or json.loads(filename.read_text()).get('fingerprint') != expected:
        raise ValueError(f'Resume requires the same group, configuration and source fingerprint: {root}')


def verify_reference():
    committed = ['provenance.json', 'validation.json', 'step-logs.tar.gz', 'params.yml']
    records = {}
    for name in committed:
        path = ROOT / 'experiments/change-rewards' / name
        old = subprocess.check_output(['git', 'show', f'{REFERENCE_COMMIT}:{path.relative_to(ROOT)}'], cwd=ROOT)
        if hashlib.sha256(old).hexdigest() != sha256(path):
            raise ValueError(f'Reference evidence changed: {path}')
        records[str(path.relative_to(ROOT))] = sha256(path)
    provenance = json.loads((ROOT / 'experiments/change-rewards/provenance.json').read_text())
    for name, expected in provenance['source_hashes'].items():
        old = subprocess.check_output(['git', 'show', f'{REFERENCE_COMMIT}:{name}'], cwd=ROOT)
        if hashlib.sha256(old).hexdigest() != expected:
            raise ValueError(f'Reference source did not match recorded training: {name}')
    with tarfile.open(ROOT / 'experiments/change-rewards/step-logs.tar.gz', 'r:gz') as archive:
        hashes = json.load(archive.extractfile('SHA256.json'))
        for name, expected in hashes.items():
            if name.startswith('new/i2c/') or name in ['new/experiment.json', 'new/baseline.json']:
                path = REFERENCE / name.removeprefix('new/')
                if sha256(path) != expected:
                    raise ValueError(f'Reference output changed: {path}')
                records[str(path.relative_to(ROOT))] = expected
    for name, expected in provenance['benchmark_hashes'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Benchmark changed: {name}')
    return records


def group_config(group):
    label, scale, learning = GROUPS[group]
    config = load_config(HERE / f'{label}.yml')
    expected = load_config(ROOT / 'experiments/change-rewards/params.yml')
    expected['protocol']['circuits'] = {'i2c': expected['protocol']['circuits']['i2c']}
    expected['method']['normalization']['returns'] = 'none'
    expected['method']['reward']['feasible_scale'] = scale
    expected['method']['learning_enabled'] = learning
    expected['runtime']['output_dir'] = str(ROOT / 'results/i2c-return-handling' / label)
    if normalized(config) != normalized(expected):
        raise ValueError(f'Unexpected settings for fixed group {group}')
    config['runtime']['experiment_fingerprint'] = fingerprint(config, group)
    return config


def run_seed(task):
    config, group, seed, resume = task
    folder = Path(config['runtime']['output_dir']) / 'i2c' / f'seed-{seed}'
    circuit = config['protocol']['circuits']['i2c']
    restoring = resume and (folder / 'checkpoint.pt').exists()
    agent = A2C(config, circuit, seed, folder, resume=restoring)
    initial_file = folder / 'initial.pt'
    if not restoring:
        shutil.copyfile(agent.checkpoint, initial_file)
    initial = torch.load(initial_file, map_location='cpu', weights_only=True)
    initial_network, initial_optimizer = state_hash(initial['network']), state_hash(initial['optimizer'])
    for _ in range(agent.episodes_completed, config['protocol']['episodes']):
        reward = agent.run_episode()
        if not config['method']['learning_enabled']:
            if state_hash(agent.network.state_dict()) != initial_network or state_hash(agent.optimizer.state_dict()) != initial_optimizer:
                raise RuntimeError('Frozen policy or optimizer changed.')
        print(f'{group}/seed-{seed}: episode {agent.episodes_completed}/100, reward={reward}', flush=True)
    verify_netlist(config, circuit, folder / 'best-mapped.v', agent.game.best)
    final = torch.load(agent.checkpoint, map_location='cpu', weights_only=True)
    dump(folder / 'parameter-validation.json', dict(initial_network=initial_network,
        final_network=state_hash(final['network']), initial_optimizer=initial_optimizer,
        final_optimizer=state_hash(final['optimizer']), rng_advanced=not torch.equal(initial['rng_state'], final['rng_state']),
        learning_enabled=config['method']['learning_enabled'], fingerprint=config['runtime']['experiment_fingerprint']))
    dump(folder / 'result.json', dict(circuit='i2c', seed=seed, status='complete', group=group,
        training_seconds=agent.training_seconds, episodes_completed=agent.episodes_completed,
        best=agent.game.best, rewards=agent.rewards))
    return group, seed, agent.game.best['luts']


def run_group(config, group, resume):
    root = Path(config['runtime']['output_dir'])
    expected = config['runtime']['experiment_fingerprint']
    check_resume(root, expected, resume)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / '.lock'), timeout=0):
        dump(root / 'experiment.json', dict(config=config, group=group, fingerprint=expected))
        (root / 'params.yml').write_text(yaml.safe_dump(config, sort_keys=False))
        run_baselines(config)
        tasks = [(config, group, seed, resume) for seed in config['protocol']['seeds']]
        try:
            with ProcessPoolExecutor(config['runtime']['workers'], mp_context=get_context('spawn')) as pool:
                list(pool.map(run_seed, tasks))
        finally:
            write_report(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='Continue only identical recorded groups/configurations.')
    args = parser.parse_args()
    reference_hashes = verify_reference()
    configs = {group: group_config(group) for group in GROUPS}
    for group, config in configs.items():
        check_resume(Path(config['runtime']['output_dir']), config['runtime']['experiment_fingerprint'], args.resume)
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != 'change-rewards':
        raise ValueError('Run this experiment on change-rewards.')
    old_provenance = json.loads((ROOT / 'experiments/change-rewards/provenance.json').read_text())
    tools = {}
    for name, key, argv in [('abc', 'abc_binary', ['-c', 'version']), ('yosys', 'yosys_binary', ['-V'])]:
        binary = Path(configs['A']['runtime'][key])
        if sha256(binary) != old_provenance[name]['sha256']:
            raise ValueError(f'{name} differs from the reference tool binary.')
        tools[name] = dict(path=str(binary), sha256=sha256(binary),
                           version=subprocess.check_output([str(binary), *argv], text=True).strip())
    provenance_file = HERE / 'provenance.json'
    current = dict(command=[sys.executable, '-B', '-u', str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]],
        started_at=datetime.now(timezone.utc).isoformat(), branch=branch,
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_hashes=source_hashes(), config_hashes={f'{label}.yml': sha256(HERE / f'{label}.yml') for label, _, _ in GROUPS.values()},
        reference_commit=REFERENCE_COMMIT, reference_hashes=reference_hashes,
        python=sys.version, torch=torch.__version__, numpy=np.__version__, pyyaml=yaml.__version__,
        platform=platform.platform(), tools=tools, groups={})
    if args.resume and provenance_file.exists():
        previous = json.loads(provenance_file.read_text())
        if previous['source_hashes'] != current['source_hashes'] or previous['config_hashes'] != current['config_hashes']:
            raise ValueError('Cannot resume after experiment sources or configurations changed.')
        current['previous_invocation'] = previous
    dump(provenance_file, current)
    for group, config in configs.items():
        started = perf_counter()
        log = Path(config['runtime']['output_dir']).parent / f'{group}-training.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-B', '-u', str(HERE / 'worker.py'), group]
        if args.resume:
            command.append('--resume')
        print(f'Starting group {group}. Log: {log}', flush=True)
        with log.open('a' if args.resume else 'w') as stream:
            completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        current['groups'][group] = dict(exit_code=completed.returncode, elapsed_seconds=perf_counter() - started,
            fingerprint=config['runtime']['experiment_fingerprint'], log=str(log.relative_to(ROOT)), log_sha256=sha256(log))
        dump(provenance_file, current)
        if completed.returncode:
            raise RuntimeError(f'Group {group} failed; see {log}')
    if reference_hashes != verify_reference():
        raise ValueError('Reference changed during the new experiment.')
    current.update(finished_at=datetime.now(timezone.utc).isoformat(), exit_code=0)
    dump(provenance_file, current)
    print('All nine new runs complete.', flush=True)


if __name__ == '__main__':
    main()
