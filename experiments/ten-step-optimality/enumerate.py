"""Exhaustively score every action prefix, without pruning or changing ABC settings."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def structural_hash(path):
    # ABC adds a wall-clock comment. Preserve every other byte, including node order.
    lines = Path(path).read_bytes().splitlines(keepends=True)
    if lines and lines[0].startswith(b'// Benchmark ') and b' written by ABC on ' in lines[0]:
        lines = lines[1:]
    return hashlib.sha256(b''.join(lines)).hexdigest()


def rank(row):
    return (0, row['luts'], row['levels']) if row['feasible'] else (1, row['levels'], row['luts'])


def tasks(cfg, depth):
    index = 0
    for name in cfg['protocol']['circuits']:
        for length in range(depth + 1):
            for sequence in itertools.product(cfg['protocol']['actions'], repeat=length):
                yield dict(index=index, circuit=name, depth=length, actions=list(sequence))
                index += 1


def command(cfg, row, folder):
    circuit = cfg['protocol']['circuits'][row['circuit']]
    sequence = cfg['protocol']['initial_sequence'] + row['actions']
    flow = f'read "{circuit["file"]}"; ' + '; '.join(sequence) + '; '
    # Identical mapping flow to FPGASession._run; feature extraction is unnecessary.
    flow += f'write_verilog "{folder / "current.v"}"; if -K {cfg["protocol"]["lut_inputs"]}; '
    flow += f'write_verilog "{folder / "mapped.v"}"; print_stats;'
    return [cfg['runtime']['abc_binary'], '-c', flow]


def score(cfg, row, folder):
    timer = time.perf_counter()
    for name in ('current.v', 'mapped.v'):
        (folder / name).unlink(missing_ok=True)
    argv = command(cfg, row, folder)
    process = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    output = process.stdout
    values = re.findall(r'\bnd\s*=\s*(\d+)[^\n]*?\blev\s*=\s*(\d+)', output)
    if process.returncode or not values or not all((folder / p).exists() for p in ('current.v', 'mapped.v')):
        raise RuntimeError(f'ABC failed: {output}')
    luts, levels = map(int, values[-1])
    result = dict(row)
    result.update(status='complete', luts=luts, levels=levels,
        feasible=levels <= cfg['protocol']['circuits'][row['circuit']]['max_levels'],
        structural_sha256=structural_hash(folder / 'current.v'),
        mapped_structural_sha256=structural_hash(folder / 'mapped.v'),
        elapsed_seconds=time.perf_counter() - timer)
    return result, dict(index=row['index'], command=argv, output=output)


def records(folder):
    result = {}
    for path in sorted(folder.glob('worker-*/records.jsonl')):
        for line in path.read_text().splitlines():
            if line:
                row = json.loads(line)
                if row['index'] in result:
                    raise ValueError('A candidate index was recorded twice.')
                result[row['index']] = row
    return result


def worker(cfg, depth, number, folder, resume):
    destination = folder / f'worker-{number}'
    destination.mkdir(parents=True, exist_ok=True)
    saved = {}
    record_path = destination / 'records.jsonl'
    if resume and record_path.exists():
        raw = record_path.read_text()
        if raw and not raw.endswith('\n'):
            complete, _, tail = raw.rpartition('\n')
            (destination / 'interrupted-tail.txt').write_text(tail)
            raw = complete + '\n' if complete else ''
            record_path.write_text(raw)
        saved = {r['index']: r for r in (json.loads(line) for line in raw.splitlines() if line)}
    count = 0
    with record_path.open('a') as stream, (destination / 'commands.jsonl').open('a') as logs:
        for task in tasks(cfg, depth):
            if task['index'] % 3 != number or task['index'] in saved:
                continue
            try:
                row, log = score(cfg, task, destination)
            except Exception as error:
                dump(destination / 'failure.json', dict(task=task, error=repr(error), time=now()))
                raise
            logs.write(json.dumps(log) + '\n')
            logs.flush()
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            count += 1
            dump(destination / 'status.json', dict(status='running', new_candidates=count, last_index=task['index'], time=now()))
    dump(destination / 'status.json', dict(status='complete', new_candidates=count, time=now()))


def validate_and_report(cfg, depth, folder, manifest):
    data = records(folder)
    expected = list(tasks(cfg, depth))
    if set(data) != {t['index'] for t in expected}:
        raise ValueError('Candidate coverage is incomplete.')
    for task in expected:
        if any(data[task['index']][key] != task[key] for key in ('circuit', 'depth', 'actions')):
            raise ValueError('A candidate was assigned to the wrong sequence.')
    rows = [data[index] for index in sorted(data)]
    with (HERE / 'candidates.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'actions': json.dumps(row['actions'])})
    summary = []
    for name, circuit in cfg['protocol']['circuits'].items():
        selected = [r for r in rows if r['circuit'] == name]
        previous_structures = set()
        for limit in range(depth + 1):
            eligible = [r for r in selected if r['depth'] <= limit]
            current = [r for r in selected if r['depth'] == limit]
            feasible = [r for r in eligible if r['feasible']]
            best = min(feasible, key=lambda r: (rank(r), r['index'])) if feasible else None
            structures = {r['structural_sha256'] for r in eligible}
            item = dict(circuit=name, maximum_depth=limit, candidates=len(eligible),
                feasible_candidates=len(feasible), best=best, distinct_structures=len(structures),
                new_structures=len(structures - previous_structures),
                structures_at_exact_depth=len({r['structural_sha256'] for r in current}))
            previous_structures = structures
            summary.append(item)
        best = min(selected, key=lambda r: (rank(r), r['index']))
        export = folder / 'best' / name
        export.mkdir(parents=True, exist_ok=True)
        replay, _ = score(cfg, best, export)
        for key in ('luts', 'levels', 'structural_sha256', 'mapped_structural_sha256'):
            if replay[key] != best[key]:
                raise ValueError('Best candidate replay mismatch.')
        for netlist in ('current.v', 'mapped.v'):
            argv = [cfg['runtime']['abc_binary'], '-c', f'cec "{circuit["file"]}" "{export / netlist}";']
            output = subprocess.check_output(argv, text=True)
            (export / (netlist + '.cec.log')).write_text(output)
            if 'Networks are equivalent' not in output:
                raise ValueError('Best netlist failed CEC.')
        dump(export / 'best.json', best)
    dump(HERE / 'summary.json', dict(status='complete', max_depth=depth, expected_candidates=len(expected),
        evaluated_candidates=len(rows), complete_sequence_coverage=True, pruning=False,
        structural_identity='SHA256 of ABC Verilog bytes with only the generated timestamp comment removed; no states merged',
        results=summary, experiment=manifest))
    archive = HERE / 'evidence.tar.gz'
    import tarfile
    paths = sorted(p for p in folder.rglob('*') if p.is_file() and p.name not in ('current.v', 'mapped.v'))
    paths += sorted((folder / 'best').glob('*/*.v'))
    with tarfile.open(archive, 'w:gz') as stream:
        for path in paths:
            stream.add(path, arcname=str(path.relative_to(folder)))
    hashes = {str(p.relative_to(folder)): sha(p) for p in paths}
    dump(HERE / 'evidence-hashes.json', dict(archive_sha256=sha(archive), files=hashes))
    with tarfile.open(archive, 'r:gz') as stream:
        for name, digest in hashes.items():
            if hashlib.sha256(stream.extractfile(name).read()).hexdigest() != digest:
                raise ValueError('Archive readback failed.')
    dump(HERE / 'provenance.json', manifest)
    known = read(ROOT / 'experiments/learning-effectiveness/summary.json')['training']
    lines = ['# 0–4 步动作序列精确枚举', '',
        '本实验完整遍历固定起点、原 7 个动作、LUT6 与原深度约束下的所有 0–4 步序列。'
        '只省去神经网络和 Yosys 特征提取，ABC 优化及映射命令保持原形式。没有剪枝或状态合并。', '',
        '| 电路 | ≤0步最优LUT | ≤1步 | ≤2步 | ≤3步 | ≤4步 | 已知≤10步最好值 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in cfg['protocol']['circuits']:
        scores = [str(r['best']['luts']) if r['best'] else '无可行解' for r in summary if r['circuit'] == name]
        old = min(r['best']['luts'] for r in known if r['circuit'] == name and r['best']['feasible'])
        lines.append('| ' + name + ' | ' + ' | '.join(scores) + f' | {old} |')
    lines += ['', '| 电路 | 长度 | 累计序列数 | 累计不同结构数 | 本层新结构数 |', '|---|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f'| {row["circuit"]} | {row["maximum_depth"]} | {row["candidates"]} | {row["distinct_structures"]} | {row["new_structures"]} |')
    lines += ['', '结构数只统计完全相同的序列化网表，不用 LUT/层数或功能等价代替结构相同。'
              '此处没有利用重复结构跳过任何分支；尚未证明重载状态可保留 ABC 的全部后续行为。', '',
              '每个电路检查 2,801 个候选，三个电路共 8,403 个。每个长度的覆盖均核对；'
              '最佳解重新执行并核对结构与映射结果，映射和未映射网表均通过 CEC。', '',
              '这是本工具流程在不超过 4 步时的精确最优值。10 步最优和电路全局最优尚未得到证明。'
              '已有 10 步最好值只是未知最优 LUT 的上界。计入 0–10 步，朴素树有 329,554,457 个节点/电路，'
              '下一层增长和安全状态合并仍需验证，不能将浅层去重率直接外推到 10 步。', '',
              '```bash', '.tools/conda-env/bin/python -B -u experiments/ten-step-optimality/enumerate.py --depth 4',
              '# 中断后仅相同代码、工具、输入和设置允许续跑',
              '.tools/conda-env/bin/python -B -u experiments/ten-step-optimality/enumerate.py --depth 4 --resume', '```', '']
    (HERE / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depth', type=int, default=4, choices=range(5))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', type=int, choices=range(3))
    args = parser.parse_args()
    cfg = read(ROOT / 'experiments/learning-effectiveness/provenance.json')['config']
    folder = ROOT / 'results/ten-step-optimality' / f'depth-{args.depth}'
    if args.worker is not None:
        worker(cfg, args.depth, args.worker, folder, args.resume)
        return
    payload = dict(source_sha256=sha(__file__), protocol=cfg['protocol'],
        abc_sha256=sha(cfg['runtime']['abc_binary']),
        circuits_sha256={n: sha(c['file']) for n, c in cfg['protocol']['circuits'].items()}, depth=args.depth)
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if folder.exists():
        if not args.resume or read(folder / 'experiment.json')['fingerprint'] != fingerprint:
            raise ValueError('Resume requires the identical enumeration source, tools, data and protocol.')
    folder.mkdir(parents=True, exist_ok=True)
    manifest = dict(fingerprint=fingerprint, **payload, started_at=now(),
        command=[sys.executable, '-B', '-u', str(Path(__file__).resolve()), *sys.argv[1:]], workers=3,
        implementation_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    if args.resume:
        manifest['previous_invocation'] = read(folder / 'experiment.json')
    dump(folder / 'experiment.json', manifest)
    processes = []
    for number in range(3):
        argv = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--depth', str(args.depth), '--worker', str(number)]
        if args.resume:
            argv.append('--resume')
        log = (folder / f'worker-{number}.log').open('a')
        process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((process, log, argv))
    timer, last = time.perf_counter(), -60
    while any(p.poll() is None for p, _, _ in processes):
        elapsed = time.perf_counter() - timer
        if elapsed > 1800:
            for process, _, _ in processes:
                if process.poll() is None:
                    import os
                    os.killpg(process.pid, signal.SIGTERM)
            raise TimeoutError('Enumeration hard timeout after 30 minutes; evidence retained.')
        if elapsed - last >= 60:
            completed = sum(sum(1 for _ in p.open()) for p in folder.glob('worker-*/records.jsonl'))
            item = dict(time=now(), elapsed_seconds=elapsed, evaluated=completed)
            with (folder / 'monitor.jsonl').open('a') as stream:
                stream.write(json.dumps(item) + '\n')
            print(f'Enumeration: {completed} / {3 * sum(7 ** i for i in range(args.depth + 1))}', flush=True)
            last = elapsed
        time.sleep(0.2)
    exits = []
    for process, log, argv in processes:
        log.close()
        exits.append(dict(command=argv, exit_code=process.returncode))
    manifest.update(finished_at=now(), wall_seconds=time.perf_counter() - timer, processes=exits,
                    status='complete' if all(r['exit_code'] == 0 for r in exits) else 'failed')
    dump(folder / 'experiment.json', manifest)
    if manifest['status'] != 'complete':
        raise RuntimeError('Enumeration failed; partial records and failure evidence retained.')
    validate_and_report(cfg, args.depth, folder, manifest)
    print(f'Exact enumeration complete: {HERE / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
