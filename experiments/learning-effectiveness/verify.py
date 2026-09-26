"""Recheck every final exported netlist, including the unmapped best.v, with ABC CEC."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from common import HERE, GROUPS, POLICIES, EVAL_SEEDS, config, dump, now, sha


def verify(task):
    cfg, circuit, folder, expected, mapped_name, raw_name = task
    mapped = folder / mapped_name
    netlists = [mapped] + ([folder / raw_name] if raw_name else [])
    hashes = {p.name: sha(p) for p in netlists}
    command = f'read "{mapped}"; print_stats; '
    command += ' '.join(f'cec "{circuit["file"]}" "{p}";' for p in netlists)
    argv = [cfg['runtime']['abc_binary'], '-c', command]
    started = now()
    timer = time.perf_counter()
    record = dict(folder=str(folder.relative_to(Path(cfg['runtime']['output_dir']))), command=argv,
                  netlist_sha256=hashes, started_at=started, expected=expected)
    try:
        process = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        output = process.stdout
        (folder / 'full-equivalence.log').write_text(output)
        values = re.findall(r'\bnd\s*=\s*(\d+)[^\n]*?\blev\s*=\s*(\d+)', output)
        metrics = tuple(map(int, values[0])) if values else None
        matches = output.count('Networks are equivalent')
        if process.returncode != 0 or metrics != (expected['luts'], expected['levels']) or matches != len(netlists):
            raise RuntimeError(f'CEC/metrics failed: exit={process.returncode}, metrics={metrics}, equivalent={matches}')
        if hashes != {p.name: sha(p) for p in netlists}:
            raise RuntimeError('Netlists changed during validation.')
        record.update(status='complete', netlists_verified=matches,
                      log_sha256=sha(folder / 'full-equivalence.log'))
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ''
        if isinstance(output, bytes):
            output = output.decode(errors='replace')
        (folder / 'full-equivalence.log').write_text(output)
        record.update(status='failed', netlists_verified=0, error=repr(error),
                      log_sha256=sha(folder / 'full-equivalence.log'))
    except Exception as error:
        record.update(status='failed', netlists_verified=0, error=repr(error))
    record.update(finished_at=now(), elapsed_seconds=time.perf_counter() - timer)
    return record


def main():
    cfg = config()
    root = Path(cfg['runtime']['output_dir'])
    tasks = []
    def read(path):
        return json.loads(path.read_text())
    baseline = read(root / 'baseline.json')
    for name, circuit in cfg['protocol']['circuits'].items():
        tasks.append((cfg, circuit, root / 'baseline' / name, baseline[name], 'mapped.v', None))
        for group in GROUPS:
            for seed in cfg['protocol']['seeds']:
                folder = root / 'training' / group / name / f'seed-{seed}'
                result = read(folder / 'result.json')
                if result['status'] != 'complete':
                    raise ValueError('Cannot certify an incomplete training run.')
                tasks.append((cfg, circuit, folder, result['best'], 'best-mapped.v', 'best.v'))
        for policy in POLICIES:
            for seed in cfg['protocol']['seeds']:
                folder = root / 'evaluation' / policy / name / f'seed-{seed}'
                result = read(folder / 'result.json')
                if result['status'] != 'complete' or len(result['rollouts']) != len(EVAL_SEEDS):
                    raise ValueError('Cannot certify an incomplete evaluation task.')
                for row in result['rollouts']:
                    tasks.append((cfg, circuit, folder / f'rollout-{row["evaluation_seed"]}',
                                  row['best'], 'best-mapped.v', 'best.v'))
    started = now()
    records = []
    with ThreadPoolExecutor(max_workers=cfg['runtime']['workers']) as executor:
        futures = [executor.submit(verify, task) for task in tasks]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 100 == 0:
                print(f'CEC: {len(records)}/{len(tasks)} exports checked', flush=True)
    records.sort(key=lambda r: r['folder'])
    successful = all(r['status'] == 'complete' for r in records)
    dump(HERE / 'netlist-validation.json', dict(command=[sys.executable, '-B', str(Path(__file__).resolve())],
        source_sha256=sha(Path(__file__)), abc_sha256=sha(cfg['runtime']['abc_binary']), started_at=started,
        finished_at=now(), status='complete' if successful else 'failed', workers=cfg['runtime']['workers'],
        exports_checked=len(records), netlists_verified=sum(r['netlists_verified'] for r in records), records=records))
    models = sorted(root.rglob('*.pt'))
    dump(HERE / 'model-manifest.json', dict(root=str(root), models_committed=False, files=len(models),
        hashes={str(p.relative_to(root)): sha(p) for p in models},
        portability='Weights are retained locally under results; another machine must retrain or copy these files.'))
    print(f'CEC: {len(records)} exports, {sum(r["netlists_verified"] for r in records)} netlists; success={successful}', flush=True)
    if not successful:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
