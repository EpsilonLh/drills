"""Run the predeclared matched-dimension experiments with logs and a hard timeout."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from drills.experiment import load_config
from drills.model import ActorCritic


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results/state-validation'
PERFORMANCE_COMMIT = 'e1a99bf58ee4cf52a951465966eac9371facb197'


def fingerprint(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().numpy().tobytes())
    return digest.hexdigest()


def verify_initialization():
    config = load_config(ROOT / 'params-new-state.yml')
    source = subprocess.check_output(['git', 'show', f'{PERFORMANCE_COMMIT}:drills/model.py'], cwd=ROOT, text=True)
    namespace = {'__name__': 'drills._previous_init', '__package__': 'drills'}
    exec(compile(source, 'previous_model.py', 'exec'), namespace)
    checks = []
    for seed in range(10):
        states = []
        for constructor in [ActorCritic, ActorCritic, namespace['ActorCritic']]:
            torch.manual_seed(seed)
            states.append(constructor(11, len(config['protocol']['actions']), config['method']['network']).state_dict())
        if any(not torch.equal(states[0][name], state[name]) for state in states[1:] for name in states[0]):
            raise RuntimeError(f'Initial weight mismatch for seed {seed}.')
        checks.append(dict(seed=seed, zero=fingerprint(states[0]), performance=fingerprint(states[1]),
                           historical_performance=fingerprint(states[2]), exact_match=True))
    (OUTPUT / 'initialization-check.json').write_text(json.dumps(dict(inputs=11, parameters=938,
        historical_commit=PERFORMANCE_COMMIT, seeds=checks), indent=2) + '\n')


def preserve_performance_runs():
    target = OUTPUT / 'performance'
    target.mkdir(exist_ok=True)
    copies = []
    for circuit in ['int2float', 'i2c', 'max']:
        for seed in [0, 1, 2]:
            source = ROOT / 'results/new-state' / circuit / f'seed-{seed}'
            destination = target / circuit / f'seed-{seed}'
            if destination.exists():
                raise FileExistsError(f'Copied performance run already exists: {destination}')
            shutil.copytree(source, destination)
            files = {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source.rglob('*') if p.is_file()}
            for relative, expected in files.items():
                if hashlib.sha256((destination / relative).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f'Copy hash mismatch: {destination / relative}')
            copies.append(dict(source=str(source), destination=str(destination), file_sha256=files))
    (OUTPUT / 'reused-performance-runs.json').write_text(json.dumps(copies, indent=2) + '\n')


def run_config(filename, label, resume=False):
    config = load_config(ROOT / filename)
    directory = Path(config['runtime']['output_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    metadata = OUTPUT / label
    if metadata.exists():
        raise FileExistsError(f'Stage metadata already exists: {metadata}')
    metadata.mkdir()
    command = ['.tools/conda-env/bin/python', 'drills.py', 'train', 'fpga', filename]
    if resume:
        command.append('--resume')
    source_paths = [ROOT / 'drills.py', *sorted((ROOT / 'drills').glob('*.py')), ROOT / filename]
    snapshot = metadata / 'source'
    hashes = {}
    for path in source_paths:
        relative = path.relative_to(ROOT)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = dict(command=command, config=config, source_sha256=hashes,
                      started_at=datetime.now(timezone.utc).isoformat(), hard_timeout_seconds=3600,
                      base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                      python=sys.version, torch=torch.__version__, numpy=np.__version__,
                      yosys=subprocess.check_output([config['runtime']['yosys_binary'], '-V'], text=True).strip(),
                      abc=subprocess.check_output([config['runtime']['abc_binary'], '-c', 'version'], text=True).strip())
    log_path = metadata / 'training.log'
    started, last = time.monotonic(), -30
    expected = [(name, seed) for name in config['protocol']['circuits'] for seed in config['protocol']['seeds']]
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        provenance['pid'] = process.pid
        (metadata / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        print(f'Started {label}: PID {process.pid}; expected runs={len(expected)}.', flush=True)
        timed_out = False
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed >= 3600:
                print(f'{label}: hard timeout reached; terminating process group.', flush=True)
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            if elapsed - last >= 30:
                lines = log_path.read_text().splitlines()
                progress = next((line for line in reversed(lines) if ': episode ' in line), 'initializing')
                complete = sum((directory / name / f'seed-{seed}' / 'result.json').exists() for name, seed in expected)
                status = dict(status='running', pid=process.pid, elapsed_seconds=elapsed,
                              complete_runs=complete, required_runs=len(expected), latest_progress=progress)
                (metadata / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
                print(f'{label}: {elapsed:.0f}s; complete={complete}/{len(expected)}; {progress}', flush=True)
                last = elapsed
            time.sleep(1)
        code = process.wait()
    status = dict(status='timeout' if timed_out else ('complete' if code == 0 else 'failed'),
                  exit_code=code, elapsed_seconds=time.monotonic() - started,
                  finished_at=datetime.now(timezone.utc).isoformat(),
                  complete_runs=sum((directory / name / f'seed-{seed}' / 'result.json').exists() for name, seed in expected),
                  required_runs=len(expected))
    (metadata / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(dict(stage=label, **status)), flush=True)
    if code != 0 or timed_out:
        print('\n'.join(log_path.read_text().splitlines()[-40:]), flush=True)
        raise RuntimeError(f'{label} did not complete. No automatic retry was attempted.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['first', 'ten'])
    args = parser.parse_args()
    if not (OUTPUT / 'preserved-history.json').exists():
        raise FileNotFoundError('Preserved historical fingerprints are required before training.')
    if args.stage == 'first':
        verify_initialization()
        run_config('params-state-zero.yml', 'stage1-zero')
    else:
        status = json.loads((OUTPUT / 'stage1-zero/status.json').read_text())
        if status['status'] != 'complete':
            raise RuntimeError('Stage one must complete before expanding the fixed seed set.')
        preserve_performance_runs()
        run_config('params-state-zero-10.yml', 'stage2-zero', resume=True)
        run_config('params-state-performance-10.yml', 'stage2-performance', resume=True)


if __name__ == '__main__':
    main()
