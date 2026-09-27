"""Compare exact-enumeration records with the unchanged real FPGASession tool flow."""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from drills.fpga_session import FPGASession
from drills.experiment import load_config
from enumerate import structural_hash, command


def main():
    # YAML preserves integer reward-table keys; JSON provenance stringifies them.
    cfg = load_config(ROOT / 'experiments/learning-effectiveness/protocol.yml')
    original = json.loads((ROOT / 'experiments/learning-effectiveness/provenance.json').read_text())['config']
    if cfg['protocol'] != original['protocol']:
        raise ValueError('Reference protocol differs from the enumerated protocol.')
    bank = {(r['circuit'], tuple(json.loads(r['actions']))): r for r in csv.DictReader((HERE / 'candidates.csv').open())}
    by_index = {int(r['index']): r for r in bank.values()}
    summary = json.loads((HERE / 'summary.json').read_text())
    folder = ROOT / 'results/ten-step-optimality' / f'depth-{summary["max_depth"]}'
    verified = set()
    for path in folder.glob('worker-*/commands.jsonl'):
        for line in path.read_text().splitlines():
            log = json.loads(line)
            index = log['index']
            if index in verified:
                raise AssertionError('Duplicate recorded tool command.')
            row = by_index[index]
            expected = command(cfg, {**row, 'actions': json.loads(row['actions'])}, path.parent)
            values = re.findall(r'\bnd\s*=\s*(\d+)[^\n]*?\blev\s*=\s*(\d+)', log['output'])
            if expected != log['command'] or tuple(map(int, values[-1])) != (int(row['luts']), int(row['levels'])):
                raise AssertionError('A CSV result does not match its recorded command and tool output.')
            verified.add(index)
    if verified != set(by_index):
        raise AssertionError('Recorded command coverage is incomplete.')
    comparisons = []
    sequence = ['rewrite', 'refactor -z', 'resub', 'balance']
    with tempfile.TemporaryDirectory() as directory:
        for name, circuit in cfg['protocol']['circuits'].items():
            session = FPGASession(cfg, circuit, Path(directory) / name)
            session.reset()
            for depth in range(5):
                if depth:
                    session.step(cfg['protocol']['actions'].index(sequence[depth - 1]))
                row = bank[(name, tuple(sequence[:depth]))]
                actual = dict(luts=session.luts, levels=session.levels,
                    structural_sha256=structural_hash(session.episode_dir / 'current.v'),
                    mapped_structural_sha256=structural_hash(session.episode_dir / 'mapped.v'))
                matched = all(str(value) == row[key] for key, value in actual.items())
                comparisons.append(dict(circuit=name, depth=depth, actions=sequence[:depth], matched=matched, **actual))
                if not matched:
                    raise AssertionError(f'Enumeration changed the tool flow: {name}, depth {depth}')
    result = dict(command=[sys.executable, '-B', str(Path(__file__).resolve())],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_source_sha256=hashlib.sha256((ROOT / 'drills/fpga_session.py').read_bytes()).hexdigest(),
        candidates_sha256=hashlib.sha256((HERE / 'candidates.csv').read_bytes()).hexdigest(),
        comparisons=len(comparisons), recorded_tool_commands_verified=len(verified),
        all_matched=True, records=comparisons)
    (HERE / 'reference-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'{len(comparisons)} real-tool reference comparisons and {len(verified)} recorded commands passed.')


if __name__ == '__main__':
    main()
