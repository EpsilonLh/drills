"""Validate every search transition and export a self-contained historical comparison."""
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import tarfile

import numpy as np
import torch

from run import HERE, ROOT, normalized, sha256


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_reward(config, previous, current, initial, strategy):
    old_luts, old_levels = previous
    luts, levels = current
    reward = config['method']['reward']
    area = str(int(luts < old_luts) - int(luts > old_luts))
    if levels <= config['max_levels']:
        if strategy == 'new' and old_levels <= config['max_levels']:
            return (old_luts - luts) / initial, True
        return reward['feasible'][area], False
    depth = str(int(levels < old_levels) - int(levels > old_levels))
    return reward['infeasible'][depth][area], False


def validate_run(root, config, strategy, name, seed):
    protocol = config['protocol']
    folder = root / name / f'seed-{seed}'
    result = json.loads((folder / 'result.json').read_text())
    require(result['status'] == 'complete' and result['episodes_completed'] == protocol['episodes'],
            f'Incomplete run: {folder}')
    require(result['circuit'] == name and result['seed'] == seed, f'Run identity mismatch: {folder}')
    require(len(result['rewards']) == protocol['episodes'], f'Wrong reward count: {folder}')
    params = dict(method=config['method'], max_levels=protocol['circuits'][name]['max_levels'])
    best, best_rank, feasible_best = None, None, None
    trajectory, archive_files = [], []
    affected, actions, max_error = 0, 0, 0.0
    for episode in range(1, protocol['episodes'] + 1):
        filename = folder / 'episodes' / str(episode) / 'log.csv'
        archive_files.append((filename, f'{strategy}/{name}/seed-{seed}/episodes/{episode}/log.csv'))
        with filename.open(newline='') as source:
            rows = list(csv.DictReader(source))
        require(len(rows) == protocol['iterations'] + 1, f'Wrong step count: {filename}')
        require([int(row['iteration']) for row in rows] == list(range(len(rows))), f'Invalid steps: {filename}')
        initial = int(rows[0]['luts'])
        require(strategy == 'old' or initial > 0, f'Zero initial LUTs: {filename}')
        sequence = list(protocol['initial_sequence'])
        previous, total = None, 0.0
        for iteration, row in enumerate(rows):
            luts, levels, reward = int(row['luts']), int(row['levels']), float(row['reward'])
            require(math.isfinite(reward), f'Nonfinite reward: {filename}:{iteration}')
            if iteration:
                require(row['optimization'] in protocol['actions'], f'Unknown action: {filename}')
                sequence.append(row['optimization'])
                expected, consecutive_feasible = expected_reward(params, previous, (luts, levels), initial, strategy)
                affected += consecutive_feasible if strategy == 'new' else (
                    previous[1] <= params['max_levels'] and levels <= params['max_levels'])
                actions += 1
                max_error = max(max_error, abs(expected - reward))
                require(math.isclose(expected, reward, rel_tol=1e-12, abs_tol=1e-12),
                        f'Wrong reward: {filename}:{iteration}, {reward} != {expected}')
                total += reward
            else:
                require(reward == 0 and row['optimization'] == sequence[-1], f'Invalid initialization: {filename}')
            feasible = levels <= params['max_levels']
            rank = (0, luts, levels) if feasible else (1, levels, luts)
            if (iteration or protocol['evaluation']['include_initial']) and (best_rank is None or rank < best_rank):
                best_rank = rank
                best = dict(luts=luts, levels=levels, feasible=feasible, sequence=list(sequence),
                            episode=episode, iteration=iteration)
            if feasible and (iteration or protocol['evaluation']['include_initial']):
                if feasible_best is None or (luts, levels) < feasible_best:
                    feasible_best = (luts, levels)
            previous = (luts, levels)
        require(math.isclose(total, result['rewards'][episode - 1], rel_tol=1e-12, abs_tol=1e-12),
                f'Episode reward mismatch: {filename}')
        trajectory.append(dict(strategy=strategy, circuit=name, seed=seed, episode=episode,
            actions_completed=episode * protocol['iterations'],
            best_feasible_luts=feasible_best[0] if feasible_best else None,
            best_feasible_levels=feasible_best[1] if feasible_best else None))
    require(best == result['best'] == json.loads((folder / 'best.json').read_text()), f'Best mismatch: {folder}')
    require(best['feasible'], f'No feasible final result: {folder}')
    saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
    require(saved['episodes_completed'] == protocol['episodes'] and saved['best'] == best
            and saved['rewards'] == result['rewards'], f'Checkpoint mismatch: {folder}')
    original_cec = (folder / 'equivalence.log').read_text()
    require('Networks are equivalent' in original_cec, f'Missing original CEC: {folder}')
    mapped = folder / 'best-mapped.v'
    circuit = protocol['circuits'][name]['file']
    abc = json.loads((HERE / 'provenance.json').read_text())['abc']['path']
    command = f'read "{mapped}"; print_stats; cec "{circuit}" "{mapped}";'
    output = subprocess.check_output([abc, '-c', command], text=True)
    cec_file = HERE / 'cec' / f'{strategy}-{name}-seed-{seed}.log'
    cec_file.parent.mkdir(exist_ok=True)
    cec_file.write_text(output)
    metrics = tuple(map(int, re.findall(r'\bnd\s*=\s*(\d+)[^\n]*?\blev\s*=\s*(\d+)', output)[-1]))
    require(metrics == (best['luts'], best['levels']) and 'Networks are equivalent' in output,
            f'Fresh CEC or metric check failed: {folder}')
    for filename in ['best.json', 'result.json', 'equivalence.log', 'best-mapped.v']:
        archive_files.append((folder / filename, f'{strategy}/{name}/seed-{seed}/{filename}'))
    validation = dict(strategy=strategy, circuit=name, seed=seed, episodes=protocol['episodes'], actions=actions,
        consecutive_feasible_actions=affected, reward_max_absolute_error=max_error,
        best_matches_logs=True, checkpoint_matches=True, original_cec_passed=True,
        fresh_cec_passed=True, fresh_cec_log=str(cec_file.relative_to(ROOT)), fresh_cec_sha256=sha256(cec_file))
    return result, trajectory, archive_files, validation


def write_archive(files):
    destination = HERE / 'step-logs.tar.gz'
    manifest = {name: sha256(path) for path, name in files}
    # Fixed archive metadata makes repeated evidence generation reproducible.
    with destination.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for path, name in sorted(files, key=lambda item: item[1]):
                data = path.read_bytes()
                entry = tarfile.TarInfo(name)
                entry.size, entry.mode = len(data), 0o644
                archive.addfile(entry, io.BytesIO(data))
            data = (json.dumps(manifest, indent=2) + '\n').encode()
            entry = tarfile.TarInfo('SHA256.json')
            entry.size, entry.mode = len(data), 0o644
            archive.addfile(entry, io.BytesIO(data))
    # Read back the delivered archive and verify every member, not only its sources.
    with tarfile.open(destination, 'r:gz') as archive:
        for name, expected in manifest.items():
            require(hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected,
                    f'Archive integrity failed: {name}')
    return dict(path=destination.name, sha256=sha256(destination), files=len(files), bytes=destination.stat().st_size)


def main():
    provenance = json.loads((HERE / 'provenance.json').read_text())
    require(provenance.get('exit_code') == 0, 'New training did not finish successfully.')
    for name, expected in provenance['source_hashes'].items():
        require(sha256(ROOT / name) == expected, f'Training source changed: {name}')
    require(sha256(ROOT / 'results/change-rewards/training.log') == provenance['training_log_sha256'],
            'Training log changed.')
    historical_path = ROOT / provenance['historical_control']['manifest']
    require(sha256(historical_path) == provenance['historical_control']['manifest_sha256'], 'Historical manifest changed.')
    historical = json.loads(historical_path.read_text())
    for name, expected in historical['result_file_sha256'].items():
        require(sha256(ROOT / name) == expected, f'Historical result changed: {name}')
    roots = dict(old=ROOT / 'results', new=ROOT / 'results/change-rewards')
    configs = {strategy: normalized(json.loads((root / 'experiment.json').read_text())['config'])
               for strategy, root in roots.items()}
    old, new = configs['old'], configs['new']
    require(old == normalized(historical['config']), 'Historical config no longer matches its manifest.')
    comparable = json.loads(json.dumps(new))
    require(comparable['method']['reward'].pop('feasible_mode') == 'normalized_delta', 'Wrong reward mode.')
    require(comparable['method']['reward'].pop('feasible_scale') == 1, 'Wrong reward scale.')
    for section in ['protocol', 'method', 'environment']:
        require(comparable[section] == old[section], f'Unexpected setting change: {section}')
    require(new['runtime']['workers'] == old['runtime']['workers'], 'Worker count changed.')
    require(json.loads((roots['old'] / 'baseline.json').read_text()) ==
            json.loads((roots['new'] / 'baseline.json').read_text()), 'resyn2 mapping changed.')
    for name, circuit in new['protocol']['circuits'].items():
        source = subprocess.check_output(['git', 'show', f'{historical["baseline_commit"]}:benchmarks/{Path(circuit["file"]).name}'], cwd=ROOT)
        require(hashlib.sha256(source).hexdigest() == sha256(Path(circuit['file'])), f'Benchmark changed: {name}')
    runs, validations, trajectories, files = {}, [], [], []
    for strategy, root in roots.items():
        for name in new['protocol']['circuits']:
            for seed in new['protocol']['seeds']:
                result, trajectory, archive_files, validation = validate_run(root, configs[strategy], strategy, name, seed)
                runs[strategy, name, seed] = result
                validations.append(validation)
                trajectories.extend(trajectory)
                files.extend(archive_files)
        for filename in ['experiment.json', 'baseline.json', 'results.json', 'results.md']:
            files.append((root / filename, f'{strategy}/{filename}'))
    pairs, summaries = [], {}
    for name in new['protocol']['circuits']:
        old_values, new_values, deltas = [], [], []
        for seed in new['protocol']['seeds']:
            previous, current = runs['old', name, seed], runs['new', name, seed]
            delta = previous['best']['luts'] - current['best']['luts']
            pairs.append(dict(circuit=name, seed=seed, old_luts=previous['best']['luts'], new_luts=current['best']['luts'],
                old_levels=previous['best']['levels'], new_levels=current['best']['levels'],
                lut_reduction=delta, outcome='win' if delta > 0 else 'loss' if delta < 0 else 'tie',
                old_training_seconds=previous['training_seconds'], new_training_seconds=current['training_seconds'],
                old_best=previous['best'], new_best=current['best']))
            old_values.append(previous['best']['luts'])
            new_values.append(current['best']['luts'])
            deltas.append(delta)
        summaries[name] = dict(old_mean=float(np.mean(old_values)), old_std=float(np.std(old_values)),
            new_mean=float(np.mean(new_values)), new_std=float(np.std(new_values)),
            mean_lut_reduction=float(np.mean(deltas)), percent_reduction=100 * float(np.mean(deltas)) / float(np.mean(old_values)),
            wins=sum(delta > 0 for delta in deltas), ties=deltas.count(0), losses=sum(delta < 0 for delta in deltas))
    comparison = dict(metric='Best feasible training-time LUTs; reduction = old - new (positive is better)',
        control='Historical completed results; no old reward training was rerun',
        protocol=new['protocol'], reward=new['method']['reward'], summary=summaries, pairs=pairs)
    dump(HERE / 'comparison.json', comparison)
    fields = ['circuit', 'seed', 'old_luts', 'new_luts', 'old_levels', 'new_levels', 'lut_reduction', 'outcome',
              'old_training_seconds', 'new_training_seconds']
    with (HERE / 'comparison.csv').open('w', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(pairs)
    with (HERE / 'trajectories.csv').open('w', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=list(trajectories[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(trajectories)
    archive = write_archive(files)
    tests = json.loads((HERE / 'test-results.json').read_text())
    require(tests['exit_code'] == 0 and tests['tests_passed'] == 12 and tests['tests_skipped'] == 0,
            'Required tests not complete.')
    require(tests['test_source_sha256'] == sha256(ROOT / 'tests/test_rewards.py'), 'Test source changed.')
    validation = dict(recorded_at=datetime.now(timezone.utc).isoformat(), tests=tests,
        evaluator_sha256=sha256(Path(__file__).resolve()),
        historical_artifacts_checked=len(historical['result_file_sha256']), historical_artifacts_unchanged=True,
        non_reward_settings_equal=True, benchmarks_match_control_commit=True, resyn2_equal=True,
        runs=validations, total_actions_checked=sum(run['actions'] for run in validations), archive=archive)
    dump(HERE / 'validation.json', validation)
    lines = ['# LUT 幅度奖励实验', '',
        '固定 c=1，仅前后均满足层数约束时采用 `(previous LUTs - current LUTs) / initial LUTs`。',
        '三个电路、种子 0/1/2、每种子 100 轮 × 10 步，新策略从头训练；旧奖励直接复用已有完成结果。', '',
        '## 最好可行 LUT', '',
        '| 电路 | 旧均值 ± 总体标准差 | 新均值 ± 总体标准差 | 平均减少 LUT | 减少比例 | 胜/平/负 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, summary in summaries.items():
        lines.append(f'| {name} | {summary["old_mean"]:.3f} ± {summary["old_std"]:.3f} | '
            f'{summary["new_mean"]:.3f} ± {summary["new_std"]:.3f} | {summary["mean_lut_reduction"]:+.3f} | '
            f'{summary["percent_reduction"]:+.3f}% | {summary["wins"]}/{summary["ties"]}/{summary["losses"]} |')
    lines += ['', '减少量为旧 LUT 减新 LUT，正值表示改善。所有新旧最终最优解均满足层数约束。', '',
        '| 电路 | 种子 | 旧 LUT | 新 LUT | 旧层数 | 新层数 | 减少 LUT |', '|---|---:|---:|---:|---:|---:|---:|']
    for pair in pairs:
        lines.append(f'| {pair["circuit"]} | {pair["seed"]} | {pair["old_luts"]} | {pair["new_luts"]} | '
                     f'{pair["old_levels"]} | {pair["new_levels"]} | {pair["lut_reduction"]:+d} |')
    lines += ['', '## 验证与解释', '',
        '- 12 项测试通过，包括奖励比例、对称性、49→50→49 原始奖励归零、约束边界、保留分支、旧配置、每轮初始化、零分母拒绝及真实工具续训一致性。',
        '- 对全部 18 次新旧运行的 18,000 个动作重新计算奖励，并逐轮核对奖励和、日志最优解与检查点；所有检查通过。',
        '- 18 个最终最优网表均使用当前 ABC 重新确认 LUT、层数与原电路组合逻辑等价性；新实验 resyn2 指标与旧数据一致。',
        '- 历史清单中的 2,768 个文件哈希全部一致；历史源码对应原提交，非奖励训练参数与新实验一致，输入电路与原提交一致。', '',
        '| 电路 | 旧日志前后均可行动作比例 | 新日志前后均可行动作比例 |', '|---|---:|---:|']
    for name in summaries:
        fractions = []
        for strategy in ['old', 'new']:
            group = [item for item in validations if item['strategy'] == strategy and item['circuit'] == name]
            fractions.append(sum(item['consecutive_feasible_actions'] for item in group) / sum(item['actions'] for item in group))
        lines.append(f'| {name} | {fractions[0]:.2%} | {fractions[1]:.2%} |')
    lines += ['',
        '三种子结果是固定预算下的描述性历史对照，不能据此声称统计显著或推广到其他电路。评估的是训练搜索中遇到的最好可行解，并非独立测试集性能。',
        '未在看到结果后调整 c、种子或预算。原始奖励尺度已变化，奖励总和不用于跨策略比较；gamma=0.99 的折扣回报也不保证往返相消。',
        '训练保留逐轮回报标准化：全程使用新公式时，正数 c 通常被标准化抵消；混合原有分支时相对尺度仍有影响。',
        '历史对照未独立记录其当时的工具版本和完整运行起止时间；provenance.json 的当前版本只描述新实验和此次重新核验。耗时仅作参考。', '',
        '## 复现与证据', '',
        '在仓库根目录运行：', '', '```bash',
        '.tools/conda-env/bin/python -B -m unittest discover -s tests -v',
        '.tools/conda-env/bin/python -B -u experiments/change-rewards/run.py',
        '.tools/conda-env/bin/python -B experiments/change-rewards/evaluate.py', '```', '',
        'run.py 要求输出目录尚不存在，防止覆盖或续训已完成数据。另一次实验需先将配置 output_dir 指向新的目录；evaluate.py 用于核验本次固定目录。',
        '工具路径写在 params.yml 中，迁移机器时可改为已安装的 ABC/Yosys 路径。历史对照位于 results/，不应修改。', '',
        '- comparison.json / comparison.csv：逐种子最优结果、动作序列、均值、标准差和配对减少量。',
        '- trajectories.csv：每轮结束时累计最好可行 LUT；尚无可行解时留空。',
        '- step-logs.tar.gz：两组全部逐步日志、最终结果、映射网表、原等价性日志和汇总，附 SHA256.json，已读回验证。',
        '- validation.json、test-results.json、cec/：测试、逐动作核验和重新执行的等价性检查。',
        '- provenance.json、params.yml：实际命令、时间、源码及电路哈希、环境、历史来源和退出状态。',
        '- 完整新模型、网表、原始日志及 training.log 保留在 results/change-rewards/，不加入 Git。', '']
    (HERE / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(summaries, indent=2))
    print(f'Validated {len(validations)} runs, {validation["total_actions_checked"]} actions. Report: {HERE / "report.md"}')


if __name__ == '__main__':
    main()
