"""Run only the new reward search, preserving and fingerprinting the historical control."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(value):
    if isinstance(value, dict):
        return {str(key): normalized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalized(item) for item in value]
    return value


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    manifest_path = ROOT / 'results/new-state/baseline-provenance.json'
    historical = json.loads(manifest_path.read_text())
    for name, expected in historical['result_file_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Historical control changed: {name}')
    for name, expected in historical['source_hashes'].items():
        source = subprocess.check_output(['git', 'show', f'{historical["baseline_commit"]}:{name}'], cwd=ROOT)
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError(f'Historical source hash mismatch: {name}')
    for name in ['drills.py', 'drills/model.py', 'drills/features.py', 'requirements.txt']:
        if sha256(ROOT / name) != historical['source_hashes'][name]:
            raise ValueError(f'Unintended training change: {name}')
    params = yaml.safe_load((HERE / 'params.yml').read_text())
    comparable = normalized(params)
    comparable['method']['reward'].pop('feasible_mode')
    comparable['method']['reward'].pop('feasible_scale')
    for name, circuit in comparable['protocol']['circuits'].items():
        circuit['file'] = str((HERE / circuit['file']).resolve())
    for section in ['protocol', 'method', 'environment']:
        if comparable[section] != normalized(historical['config'][section]):
            raise ValueError(f'Non-reward settings differ from historical control: {section}')
    if params['method']['reward']['feasible_mode'] != 'normalized_delta' or params['method']['reward']['feasible_scale'] != 1:
        raise ValueError('This experiment fixes normalized_delta with c=1.')
    if params['runtime']['workers'] != historical['config']['runtime']['workers']:
        raise ValueError('Worker count differs from historical control.')
    output = (HERE / params['runtime']['output_dir']).resolve()
    if output.exists():
        raise FileExistsError(f'Fresh training requires a new output directory: {output}')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != 'change-rewards':
        raise ValueError(f'Expected change-rewards branch, got {branch}')
    output.mkdir(parents=True)
    command = [sys.executable, '-B', '-u', 'drills.py', 'train', 'fpga',
               str((HERE / 'params.yml').relative_to(ROOT))]
    sources = [ROOT / 'drills.py', ROOT / 'params.yml', *sorted((ROOT / 'drills').glob('*.py')),
               HERE / 'params.yml', HERE / 'run.py', ROOT / 'tests/test_rewards.py']
    provenance = dict(
        command=command, working_directory=str(ROOT), started_at=now(), branch=branch,
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_hashes={str(path.relative_to(ROOT)): sha256(path) for path in sources},
        benchmark_hashes={str(path.relative_to(ROOT)): sha256(path) for path in sorted((ROOT / 'benchmarks').glob('*.aig'))},
        python=sys.version, torch=torch.__version__, numpy=np.__version__, pyyaml=yaml.__version__,
        platform=platform.platform(), machine=platform.machine(),
        historical_control=dict(root='results', baseline_commit=historical['baseline_commit'],
            manifest=str(manifest_path.relative_to(ROOT)), manifest_sha256=sha256(manifest_path),
            checked_artifacts=len(historical['result_file_sha256']), source_hashes=historical['source_hashes'],
            historical_tool_versions='Not recorded by the historical control manifest.'))
    for key, args in [('yosys', ['-V']), ('abc', ['-c', 'version'])]:
        binary_key = 'yosys_binary' if key == 'yosys' else 'abc_binary'
        binary = (HERE / params['runtime'][binary_key]).resolve()
        provenance[key] = dict(path=str(binary), sha256=sha256(binary),
                               version=subprocess.check_output([str(binary), *args], text=True).strip())
    filename = HERE / 'provenance.json'
    filename.write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Training started. Log: {output / "training.log"}', flush=True)
    with (output / 'training.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    provenance.update(finished_at=now(), exit_code=result.returncode,
                      training_log_sha256=sha256(output / 'training.log'))
    filename.write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Training exit code: {result.returncode}', flush=True)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
