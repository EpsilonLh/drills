"""The complete assessment: configuration, prescribed runs and all-seed reporting."""
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import re
import shutil
from subprocess import check_output

from filelock import FileLock
import numpy as np
import torch
import yaml

from .model import A2C
from .fpga_session import performance_feature_names, performance_feature_mask


def load_config(filename):
    filename = Path(filename).resolve()
    config = yaml.safe_load(filename.read_text(encoding='utf-8'))
    protocol, method, environment, runtime = (config[k] for k in ['protocol', 'method', 'environment', 'runtime'])
    performance_features = performance_feature_names(method)
    performance_feature_mask(method)
    for value in [protocol['episodes'], protocol['iterations'], runtime['workers'], environment['torch_threads']]:
        if type(value) is not int or value <= 0:
            raise ValueError('Episodes, iterations, workers and threads must be positive integers.')
    seeds = protocol['seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds):
        raise ValueError('Provide a nonempty list of distinct nonnegative integer seeds.')
    if not protocol['circuits'] or not protocol['actions'] or not protocol['initial_sequence']:
        raise ValueError('Circuits, actions and initial sequence must not be empty.')
    if not 0 <= protocol['evaluation']['std_ddof'] < len(seeds):
        raise ValueError('Standard deviation ddof must be smaller than the number of seeds.')
    for key in ['abc_binary', 'yosys_binary']:
        binary = runtime[key]
        if '/' in binary or '\\' in binary:
            binary = str((filename.parent / Path(binary).expanduser()).resolve())
        runtime[key] = shutil.which(binary)
        if runtime[key] is None:
            raise FileNotFoundError(binary)
    for circuit in protocol['circuits'].values():
        if performance_features and (type(circuit['max_levels']) is not int or circuit['max_levels'] <= 0):
            raise ValueError('Performance features require a positive integer max_levels.')
        circuit['file'] = str((filename.parent / Path(circuit['file']).expanduser()).resolve())
    runtime['output_dir'] = str((filename.parent / Path(runtime['output_dir']).expanduser()).resolve())
    return config


def run_baselines(config):
    results = {}
    protocol = config['protocol']
    for name, circuit in protocol['circuits'].items():
        directory = Path(config['runtime']['output_dir']) / 'baseline' / name
        directory.mkdir(parents=True, exist_ok=True)
        mapped = directory / 'mapped.v'
        mapped.unlink(missing_ok=True)
        flow = protocol['initial_sequence'] + protocol['resyn2']
        command = f'read "{circuit["file"]}"; ' + '; '.join(flow) + '; '
        command += f'if -K {protocol["lut_inputs"]}; print_stats; write_verilog "{mapped}";'
        output = check_output([config['runtime']['abc_binary'], '-c', command], text=True)
        (directory / 'abc.log').write_text(output)
        luts, levels = map(int, re.findall(r'\bnd\s*=\s*(\d+)[^\n]*?\blev\s*=\s*(\d+)', output)[-1])
        results[name] = dict(luts=luts, levels=levels)
        verify_netlist(config, circuit, mapped, results[name])
    (Path(config['runtime']['output_dir']) / 'baseline.json').write_text(json.dumps(results, indent=2) + '\n')


def verify_netlist(config, circuit, mapped, expected):
    command = f'read "{mapped}"; print_stats; cec "{circuit["file"]}" "{mapped}";'
    output = check_output([config['runtime']['abc_binary'], '-c', command], text=True)
    (mapped.parent / 'equivalence.log').write_text(output)
    values = tuple(map(int, re.findall(r'\bnd\s*=\s*(\d+)[^\n]*?\blev\s*=\s*(\d+)', output)[-1]))
    if values != (expected['luts'], expected['levels']) or 'Networks are equivalent' not in output:
        raise RuntimeError('Exported netlist metrics or combinational equivalence check failed.')


def _run_one(task):
    config, name, seed, resume = task
    directory = Path(config['runtime']['output_dir']) / name / f'seed-{seed}'
    circuit = config['protocol']['circuits'][name]
    agent = A2C(config, circuit, seed, directory,
                resume=resume and (directory / 'checkpoint.pt').exists())
    budget = config['protocol']['episodes']
    for _ in range(agent.episodes_completed, budget):
        reward = agent.run_episode()
        print(f'{name}/seed-{seed}: episode {agent.episodes_completed}/{budget}, reward={reward}', flush=True)
    verify_netlist(config, circuit, directory / 'best-mapped.v', agent.game.best)
    result = dict(circuit=name, seed=seed, status='complete',
                  training_seconds=agent.training_seconds,
                  episodes_completed=agent.episodes_completed, best=agent.game.best, rewards=agent.rewards)
    temporary = directory / 'result.tmp'
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(directory / 'result.json')


def write_report(config):
    root, protocol = Path(config['runtime']['output_dir']), config['protocol']
    rows, summary = [], {}
    lines = ['# Assessment results', '', 'Every prescribed seed is shown. Levels are mapped LUT depth.', '',
             '| Circuit | Seed | Status | Episodes | LUTs | Levels | Feasible | Training seconds |',
             '|---|---:|---|---:|---:|---:|---|---:|']
    for name in protocol['circuits']:
        group = []
        for seed in protocol['seeds']:
            directory = root / name / f'seed-{seed}'
            if (directory / 'result.json').exists():
                row = json.loads((directory / 'result.json').read_text())
            else:
                row = dict(circuit=name, seed=seed, status='missing', episodes_completed=0, best=None)
                if (directory / 'checkpoint.pt').exists():
                    saved = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
                    row.update(status='incomplete',
                               episodes_completed=saved['episodes_completed'], best=saved['best'],
                               training_seconds=saved.get('training_seconds'))
            best = row['best'] or dict(luts=None, levels=None, feasible=False)
            rows.append(row)
            group.append(row)
            lines.append(f'| {name} | {seed} | {row["status"]} | {row["episodes_completed"]} | '
                         f'{best["luts"]} | {best["levels"]} | {best["feasible"]} | {row.get("training_seconds")} |')
        valid = all(r['status'] == 'complete' and r['episodes_completed'] == protocol['episodes']
                    and r['best']['feasible'] for r in group)
        summary[name] = dict(seeds=len(group), complete=sum(r['status'] == 'complete' for r in group),
                             feasible=sum(bool(r['best'] and r['best']['feasible']) for r in group),
                             luts_mean=None, luts_std=None, levels_mean=None, levels_std=None)
        times = [r.get('training_seconds') for r in group]
        summary[name]['training_seconds_total'] = sum(times) if all(t is not None for t in times) else None
        summary[name]['training_seconds_mean'] = float(np.mean(times)) if all(t is not None for t in times) else None
        if valid:
            for metric in ['luts', 'levels']:
                values = [r['best'][metric] for r in group]
                summary[name][metric + '_mean'] = float(np.mean(values))
                summary[name][metric + '_std'] = float(np.std(values, ddof=protocol['evaluation']['std_ddof']))
    lines += ['', '| Circuit | Complete / required | Feasible / required | LUTs mean ± std | Levels mean ± std |',
              '|---|---|---|---|---|']
    for name, value in summary.items():
        averages = ['not aggregated' if value[m + '_mean'] is None else
                    f'{value[m + "_mean"]:.3f} ± {value[m + "_std"]:.3f}' for m in ['luts', 'levels']]
        lines.append(f'| {name} | {value["complete"]}/{value["seeds"]} | {value["feasible"]}/{value["seeds"]} | '
                     + ' | '.join(averages) + ' |')
    lines += ['', 'No aggregate score is computed if any prescribed seed is incomplete or infeasible.',
              'Each completed seed reports its best feasible training-time mapping, not an independent test-set score.',
              'Training time is elapsed wall time for episode initialization, ABC/Yosys search and network updates; excludes checkpoint export, model initialization, resyn2 and final CEC. Concurrent runs share machine resources.',
              '', '| Circuit | Training seconds total (sum of seeds) | Training seconds mean |', '|---|---:|---:|']
    for name, value in summary.items():
        lines.append(f'| {name} | {value["training_seconds_total"]} | {value["training_seconds_mean"]} |')
    lines.append('')
    (root / 'results.json').write_text(json.dumps(dict(runs=rows, summary=summary), indent=2) + '\n')
    (root / 'results.md').write_text('\n'.join(lines), encoding='utf-8')


def run_experiment(filename, mode, resume=False):
    config = load_config(filename)
    root = Path(config['runtime']['output_dir'])
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / '.lock'), timeout=0):
        manifest = root / 'experiment.json'
        manifest.write_text(json.dumps(dict(config=config), indent=2) + '\n')
        (root / 'params.yml').write_text(yaml.safe_dump(config, sort_keys=False))
        if mode == 'report':
            write_report(config)
            return
        if mode in ['train', 'baseline']:
            run_baselines(config)
        if mode == 'baseline':
            return
        tasks = [(config, name, seed, resume)
                 for name in config['protocol']['circuits'] for seed in config['protocol']['seeds']]
        # Preflight every run before starting any worker; never overwrite completed/partial training.
        for _, name, seed, restoring in tasks:
            directory = root / name / f'seed-{seed}'
            if directory.exists() and any(directory.iterdir()) and not restoring:
                raise FileExistsError(f'{directory} already exists; use --resume.')
        try:
            if config['runtime']['workers'] == 1:
                for task in tasks:
                    _run_one(task)
            else:
                with ProcessPoolExecutor(config['runtime']['workers'], mp_context=get_context('spawn')) as pool:
                    list(pool.map(_run_one, tasks))
        finally:
            write_report(config)
    print(f'Results: {root / "results.md"}', flush=True)
