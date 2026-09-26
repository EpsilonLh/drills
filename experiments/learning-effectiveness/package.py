"""Check provenance links, historical preservation and deliverable hashes before commit."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys

from common import ROOT, HERE, config, dump, identity, now, sha


def read(path):
    return json.loads(path.read_text())


def main():
    cfg = config()
    root = Path(cfg['runtime']['output_dir'])
    training, evaluation = read(root / 'experiment.json'), read(root / 'evaluation.json')
    fingerprint, _ = identity(cfg)
    if training['status'] != 'complete' or evaluation['status'] != 'complete' or training['fingerprint'] != fingerprint:
        raise ValueError('The full experiment is incomplete or its fingerprint changed.')
    for filename, digest in evaluation['sources'].items():
        if sha(ROOT / filename) != digest:
            raise ValueError(f'Evaluator source changed: {filename}')
    for check in evaluation['training_validation']['checks']:
        folder = root / 'training' / check['group'] / check['circuit'] / f'seed-{check["seed"]}'
        for episode, digest in check['snapshot_sha256'].items():
            if sha(folder / 'snapshots' / f'{episode}.pt') != digest:
                raise ValueError(f'Evaluation changed a training snapshot: {folder}, episode {episode}')
        if sha(folder / 'checkpoint.pt') != check['snapshot_sha256']['100']:
            raise ValueError(f'Evaluation changed the final training checkpoint: {folder}')
    tests = read(HERE / 'test-results.json')
    if not tests['successful'] or tests['skipped']:
        raise ValueError('The complete validation test suite must pass without skips.')
    for filename, digest in {**tests['source_hashes'], **tests['test_hashes']}.items():
        if sha(ROOT / filename) != digest:
            raise ValueError(f'Tested source changed: {filename}')
    if sha(HERE / 'tests.log') != tests['log_sha256']:
        raise ValueError('Test output changed.')
    validation = read(HERE / 'validation.json')
    netlists = read(HERE / 'netlist-validation.json')
    if not validation['reward_and_best_replay'] or validation['evaluation_rollouts'] != 1080:
        raise ValueError('Missing log validation.')
    if netlists['status'] != 'complete' or netlists['exports_checked'] != 1110 or netlists['netlists_verified'] != 2217:
        raise ValueError('Missing exported-netlist equivalence checks.')
    analysis = read(HERE / 'analysis-provenance.json')
    for filename, digest in analysis['sources'].items():
        if sha(HERE / filename) != digest:
            raise ValueError(f'Analysis source changed: {filename}')
    if sha(HERE / 'evidence.tar.gz') != analysis['evidence_sha256']:
        raise ValueError('Compressed evidence changed.')
    for filename, digest in read(HERE / 'plot-validation.json')['hashes'].items():
        if sha(HERE / filename) != digest:
            raise ValueError(f'Plot input/output changed: {filename}')
    for filename, digest in read(HERE / 'model-manifest.json')['hashes'].items():
        if sha(root / filename) != digest:
            raise ValueError(f'Saved model/probe changed: {filename}')
    history_path = ROOT / 'results/new-state/baseline-provenance.json'
    reference = read(HERE / 'historical-reference.json')
    if sha(history_path) != reference['manifest_sha256']:
        raise ValueError('Historical manifest changed.')
    old_i2c = reference['historical_i2c']
    old_report = subprocess.check_output(['git', 'show', old_i2c['commit'] + ':' + old_i2c['report']], cwd=ROOT)
    if hashlib.sha256(old_report).hexdigest() != old_i2c['report_sha256']:
        raise ValueError('Historical i2c reference changed.')
    historical = read(history_path)['result_file_sha256']
    mismatches = [filename for filename, digest in historical.items() if sha(ROOT / filename) != digest]
    if mismatches:
        raise ValueError(f'Historical outputs changed: {mismatches}')
    reference.update(final_checked_at=now(), historical_file_count=len(historical), all_hashes_match=True)
    dump(HERE / 'historical-reference.json', reference)
    files = sorted(p for p in HERE.iterdir() if p.is_file() and p.name != 'artifact-manifest.json')
    other_sources = [ROOT / 'README.md', ROOT / 'drills/model.py', ROOT / 'tests/test_learning_effectiveness.py']
    dump(HERE / 'artifact-manifest.json', dict(command=[sys.executable, '-B', str(Path(__file__).resolve())],
        packaged_at=now(), training_fingerprint=fingerprint, training_runs=27, evaluation_tasks=36,
        evaluation_rollouts=1080, tests_passed=tests['tests_run'], netlists_verified=netlists['netlists_verified'],
        training_checkpoints_unchanged_after_evaluation=True,
        historical_files_unchanged=len(historical),
        files={str(p.relative_to(ROOT)): dict(sha256=sha(p), bytes=p.stat().st_size) for p in files + other_sources}))
    print(f'Packaged {len(files) + len(other_sources)} artifacts; {len(historical)} historical files unchanged.')


if __name__ == '__main__':
    main()
