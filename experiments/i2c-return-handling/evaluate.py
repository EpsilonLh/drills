"""Audit all twelve i2c runs and export the prescribed return-handling comparison."""
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

from run import (GROUPS, HERE, REFERENCE, ROOT, dump, group_config, normalized, sha256, source_hashes,
                 state_hash, verify_reference)
from drills.model import A2C, ActorCritic


def require(ok, message):
    if not ok:
        raise ValueError(message)


def expected_reward(config, previous, current, initial):
    old_luts, old_levels = previous
    luts, levels = current
    limit = config['protocol']['circuits']['i2c']['max_levels']
    reward = config['method']['reward']
    area = str(int(luts < old_luts) - int(luts > old_luts))
    if levels <= limit:
        if old_levels <= limit:
            return reward['feasible_scale'] * (old_luts - luts) / initial
        return reward['feasible'][area]
    depth = str(int(levels < old_levels) - int(levels > old_levels))
    return reward['infeasible'][depth][area]


def returns(rewards, config):
    raw = np.empty(len(rewards), dtype=np.float32)
    total = 0.0
    for i in reversed(range(len(rewards))):
        total = rewards[i] + config['method']['gamma'] * total
        raw[i] = total
    processed = raw.copy()
    norm = config['method']['normalization']
    if norm['returns'] == 'standardize':
        processed = (processed - processed.mean()) / max(float(processed.std()), norm['returns_epsilon'])
    agent = A2C.__new__(A2C)
    agent.method = config['method']
    require(np.array_equal(agent._get_returns(rewards), processed), 'Actual return targets differ from independently computed targets.')
    return raw, processed


def validate_run(group, root, config, seed, abc):
    folder = root / 'i2c' / f'seed-{seed}'
    result = json.loads((folder / 'result.json').read_text())
    protocol = config['protocol']
    require(result['status'] == 'complete' and result['episodes_completed'] == 100 and len(result['rewards']) == 100,
            f'Incomplete run: {folder}')
    require(result['circuit'] == 'i2c' and result['seed'] == seed, f'Wrong run identity: {folder}')
    require(group == 'reference' or result['group'] == group, f'Group identity mismatch: {folder}')
    limit = protocol['circuits']['i2c']['max_levels']
    best, best_rank, running_best = None, None, None
    actions, episodes, diagnostics, files = [], [], [], []
    infeasible, eligible, error = 0, 0, 0.0
    for episode in range(1, 101):
        path = folder / 'episodes' / str(episode) / 'log.csv'
        with path.open(newline='') as source:
            rows = list(csv.DictReader(source))
        files.append((path, f'{group}/seed-{seed}/episodes/{episode}/log.csv'))
        require(len(rows) == 11 and [int(row['iteration']) for row in rows] == list(range(11)), f'Invalid steps: {path}')
        initial, previous, rewards = int(rows[0]['luts']), None, []
        require(initial > 0, f'Zero initial LUTs: {path}')
        sequence, ep_actions, ep_best = list(protocol['initial_sequence']), [], None
        for step, row in enumerate(rows):
            luts, levels, reward = int(row['luts']), int(row['levels']), float(row['reward'])
            require(math.isfinite(reward), f'Nonfinite reward: {path}:{step}')
            if step:
                expected = expected_reward(config, previous, (luts, levels), initial)
                error = max(error, abs(reward - expected))
                require(math.isclose(reward, expected, rel_tol=1e-12, abs_tol=1e-12), f'Incorrect reward: {path}:{step}')
                require(row['optimization'] in protocol['actions'], f'Invalid action: {path}:{step}')
                sequence.append(row['optimization'])
                ep_actions.append(row['optimization'])
                rewards.append(reward)
                eligible += previous[1] <= limit and levels <= limit
                infeasible += levels > limit
            else:
                require(reward == 0 and row['optimization'] == sequence[-1], f'Invalid initialization: {path}')
            feasible = levels <= limit
            rank = (0, luts, levels) if feasible else (1, levels, luts)
            if (step or protocol['evaluation']['include_initial']) and (best_rank is None or rank < best_rank):
                best_rank = rank
                best = dict(luts=luts, levels=levels, feasible=feasible, sequence=list(sequence), episode=episode, iteration=step)
            if feasible and (step or protocol['evaluation']['include_initial']):
                ep_best = min(ep_best, luts) if ep_best is not None else luts
                running_best = min(running_best, luts) if running_best is not None else luts
            previous = (luts, levels)
        require(math.isclose(sum(rewards), result['rewards'][episode - 1], rel_tol=1e-12, abs_tol=1e-12), f'Reward sum mismatch: {path}')
        raw, processed = returns(rewards, config)
        diagnostics.append(dict(group=group, seed=seed, episode=episode, actions_completed=episode * 10,
            episode_best_feasible_luts=ep_best, running_best_feasible_luts=running_best,
            endpoint_luts=int(rows[-1]['luts']), endpoint_levels=int(rows[-1]['levels']),
            endpoint_feasible=int(rows[-1]['levels']) <= limit, reward_sum=sum(rewards),
            raw_return_mean=float(raw.mean()), raw_return_std=float(raw.std()),
            processed_return_mean=float(processed.mean()), processed_return_std=float(processed.std())))
        episodes.append(rows)
        actions.extend(ep_actions)
    require(best == result['best'] == json.loads((folder / 'best.json').read_text()) and best['feasible'], f'Best result mismatch: {folder}')
    saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
    require(saved['episodes_completed'] == 100 and saved['rewards'] == result['rewards'] and saved['best'] == best, f'Checkpoint mismatch: {folder}')
    learning = config['method'].get('learning_enabled', True)
    require(saved.get('learning_enabled', True) == learning, f'Learning mode mismatch: {folder}')
    torch.manual_seed(seed)
    initial_network = ActorCritic(len(config['method']['features']), len(protocol['actions']), config['method']['network']).state_dict()
    if group != 'reference':
        initial = torch.load(folder / 'initial.pt', map_location='cpu', weights_only=True)
        require(initial['episodes_completed'] == 0 and initial['rewards'] == [], f'Initial snapshot is not initial: {folder}')
        require(state_hash(initial['network']) == state_hash(initial_network), f'Network initialization changed: {folder}')
        require(saved['experiment_fingerprint'] == config['runtime']['experiment_fingerprint'] == initial['experiment_fingerprint'],
                f'Checkpoint fingerprint mismatch: {folder}')
        record = json.loads((folder / 'parameter-validation.json').read_text())
        require(record['initial_network'] == state_hash(initial['network']) and record['final_network'] == state_hash(saved['network'])
                and record['initial_optimizer'] == state_hash(initial['optimizer']) and record['final_optimizer'] == state_hash(saved['optimizer']),
                f'Parameter record mismatch: {folder}')
        require(record['rng_advanced'] and not torch.equal(initial['rng_state'], saved['rng_state']), f'RNG did not advance: {folder}')
        if not learning:
            require(state_hash(initial['network']) == state_hash(saved['network']) and
                    state_hash(initial['optimizer']) == state_hash(saved['optimizer']), f'Frozen policy changed: {folder}')
        files.append((folder / 'parameter-validation.json', f'{group}/seed-{seed}/parameter-validation.json'))
    weight_changes = {}
    for component in ['actor', 'critic']:
        delta = torch.cat([(saved['network'][key] - value).flatten() for key, value in initial_network.items() if key.startswith(component + '.')])
        weight_changes[component + '_rms_change'] = float(delta.square().mean().sqrt())
    require('Networks are equivalent' in (folder / 'equivalence.log').read_text(), f'Original CEC failed: {folder}')
    mapped = folder / 'best-mapped.v'
    command = f'read "{mapped}"; print_stats; cec "{protocol["circuits"]["i2c"]["file"]}" "{mapped}";'
    output = subprocess.check_output([abc, '-c', command], text=True)
    metric = tuple(map(int, re.findall(r'\bnd\s*=\s*(\d+)[^\n]*?\blev\s*=\s*(\d+)', output)[-1]))
    require(metric == (best['luts'], best['levels']) and 'Networks are equivalent' in output, f'Fresh metrics/CEC failed: {folder}')
    cec = HERE / 'cec' / f'{group}-seed-{seed}.log'
    cec.parent.mkdir(exist_ok=True)
    cec.write_text(output)
    for name in ['best.json', 'result.json', 'best-mapped.v', 'equivalence.log']:
        files.append((folder / name, f'{group}/seed-{seed}/{name}'))
    validation = dict(group=group, seed=seed, actions_checked=len(actions), infeasible_actions=infeasible,
        consecutive_feasible_actions=eligible, reward_max_absolute_error=error, best_matches_logs=True,
        checkpoint_matches=True, cec_passed=True, cec_sha256=sha256(cec), learning_enabled=learning,
        final_network_hash=state_hash(saved['network']), initial_network_hash=state_hash(initial_network), **weight_changes)
    return result, diagnostics, actions, episodes, files, validation


def archive_files(files):
    path = HERE / 'step-logs.tar.gz'
    hashes = {name: sha256(source) for source, name in files}
    with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as archive:
            for source, name in sorted(files, key=lambda item: item[1]):
                data = source.read_bytes()
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(data), 0o644
                archive.addfile(info, io.BytesIO(data))
            data = (json.dumps(hashes, indent=2) + '\n').encode()
            info = tarfile.TarInfo('SHA256.json')
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
    with tarfile.open(path, 'r:gz') as archive:
        for name, expected in hashes.items():
            require(hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected, f'Archive mismatch: {name}')
    return dict(sha256=sha256(path), files=len(files), bytes=path.stat().st_size)


def main():
    provenance = json.loads((HERE / 'provenance.json').read_text())
    require(provenance.get('exit_code') == 0 and all(item['exit_code'] == 0 for item in provenance['groups'].values()), 'Experiment incomplete.')
    require(source_hashes() == provenance['source_hashes'], 'Training sources changed.')
    for name, expected in provenance['config_hashes'].items():
        require(sha256(HERE / name) == expected, f'Experiment YAML changed: {name}')
    require(verify_reference() == provenance['reference_hashes'], 'Reference changed.')
    torch.set_num_threads(1)
    configs = {'reference': json.loads((REFERENCE / 'experiment.json').read_text())['config']}
    roots = {'reference': REFERENCE}
    for group, (label, _, _) in GROUPS.items():
        configs[group] = json.loads((ROOT / 'results/i2c-return-handling' / label / 'experiment.json').read_text())['config']
        roots[group] = ROOT / 'results/i2c-return-handling' / label
        require(normalized(configs[group]) == normalized(group_config(group)), f'Group config changed: {group}')
        require(sha256(ROOT / provenance['groups'][group]['log']) == provenance['groups'][group]['log_sha256'], f'Training log changed: {group}')
    configs = {group: json.loads(json.dumps(config)) for group, config in configs.items()}
    results, diagnostics, actions, episodes, files, validations = {}, [], {}, {}, [], []
    for group, root in roots.items():
        for seed in [0, 1, 2]:
            result, rows, sequence, logs, archive, validation = validate_run(group, root, configs[group], seed, provenance['tools']['abc']['path'])
            results[group, seed] = result
            diagnostics.extend(rows)
            actions[group, seed], episodes[group, seed] = sequence, logs
            files.extend(archive)
            validations.append(validation)
        for name in ['experiment.json', 'baseline.json', 'results.json', 'results.md']:
            files.append((root / name, f'{group}/{name}'))
        require(json.loads((root / 'baseline.json').read_text())['i2c'] ==
                json.loads((REFERENCE / 'baseline.json').read_text())['i2c'], f'resyn2 changed: {group}')
    for group in GROUPS:
        for seed in [0, 1, 2]:
            ref, new = episodes['reference', seed][0], episodes[group, seed][0]
            require([(r['optimization'], r['luts'], r['levels']) for r in ref] ==
                    [(r['optimization'], r['luts'], r['levels']) for r in new], f'First episode differs: {group}/{seed}')
    for row in diagnostics:
        seed, ep, group = row['seed'], row['episode'], row['group']
        start = (ep - 1) * 10
        row['actions_different_from_reference'] = sum(a != b for a, b in zip(actions[group, seed][start:start + 10], actions['reference', seed][start:start + 10]))
    with (HERE / 'trajectories.csv').open('w', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=list(diagnostics[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(diagnostics)
    summary, pairs = {}, []
    for group in roots:
        values = [results[group, seed]['best']['luts'] for seed in [0, 1, 2]]
        group_rows = [row for row in diagnostics if row['group'] == group]
        summary[group] = dict(luts=values, mean=float(np.mean(values)), std=float(np.std(values)),
            late_endpoint_mean=float(np.mean([row['endpoint_luts'] for row in group_rows if row['episode'] > 80])),
            endpoint_feasible_fraction=float(np.mean([row['endpoint_feasible'] for row in group_rows])),
            actions_different_from_reference=sum(row['actions_different_from_reference'] for row in group_rows),
            training_seconds=sum(results[group, seed]['training_seconds'] for seed in [0, 1, 2]))
        for seed in [0, 1, 2]:
            result = results[group, seed]
            pairs.append(dict(group=group, seed=seed, luts=result['best']['luts'], levels=result['best']['levels'],
                reduction_vs_reference=results['reference', seed]['best']['luts'] - result['best']['luts'],
                best=result['best'], training_seconds=result['training_seconds']))
    contrasts = {}
    for name, baseline, treatment in [('A_vs_reference', 'reference', 'A'), ('B_vs_A', 'A', 'B'),
                                       ('reference_vs_C', 'C', 'reference'), ('A_vs_C', 'C', 'A'), ('B_vs_C', 'C', 'B')]:
        delta = [results[baseline, seed]['best']['luts'] - results[treatment, seed]['best']['luts'] for seed in [0, 1, 2]]
        contrasts[name] = dict(baseline=baseline, treatment=treatment, reductions=delta,
            mean_reduction=float(np.mean(delta)), wins=sum(d > 0 for d in delta), ties=delta.count(0), losses=sum(d < 0 for d in delta))
    dump(HERE / 'comparison.json', dict(summary=summary, contrasts=contrasts, pairs=pairs,
        metric='Best feasible training-time LUTs; positive reduction favors treatment', fixed_budget='100 episodes x 10 actions, seeds 0/1/2'))
    with (HERE / 'comparison.csv').open('w', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=['group', 'seed', 'luts', 'levels', 'reduction_vs_reference', 'training_seconds'],
                                extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(pairs)
    archive = archive_files(files)
    tests = json.loads((HERE / 'test-results.json').read_text())
    require(tests['exit_code'] == 0 and tests['tests_skipped'] == 0, 'Tests incomplete.')
    for name, expected in tests['source_hashes'].items():
        require(sha256(ROOT / name) == expected, f'Test source changed: {name}')
    require(sha256(HERE / 'tests.log') == tests['log_sha256'], 'Test log changed.')
    dump(HERE / 'validation.json', dict(recorded_at=datetime.now(timezone.utc).isoformat(), tests=tests,
        evaluator_sha256=sha256(Path(__file__)), runs=validations, total_actions_checked=sum(v['actions_checked'] for v in validations),
        first_episode_matches_reference=True, reference_unchanged=True, archive=archive))
    lines = ['# i2c 回报处理对照实验', '',
        '固定 i2c、深度上限 4、种子 0/1/2、100 轮 × 10 步。参考组复用已完成结果，其余三组从头运行。', '',
        '| 组别 | returns | c | 更新网络 | seed 0 | seed 1 | seed 2 | LUT 均值 ± 总体标准差 |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    settings = {'reference': ('standardize', 1, '是'), 'A': ('none', 1, '是'), 'B': ('none', 10, '是'), 'C': ('none', 1, '否')}
    for group, item in summary.items():
        mode, scale, learn = settings[group]
        lines.append(f'| {group} | {mode} | {scale} | {learn} | ' + ' | '.join(map(str, item['luts'])) +
                     f' | {item["mean"]:.3f} ± {item["std"]:.3f} |')
    lines += ['', '全部最终最优结果可行并通过组合逻辑等价性检查。主指标为固定预算训练搜索中遇到的最好可行 LUT。', '',
        '| 对照 | 每种子减少量 | 平均减少 LUT | 胜/平/负 |', '|---|---|---:|---:|']
    for name, contrast in contrasts.items():
        lines.append(f'| {name} | {contrast["reductions"]} | {contrast["mean_reduction"]:+.3f} | '
                     f'{contrast["wins"]}/{contrast["ties"]}/{contrast["losses"]} |')
    lines += ['', '减少量定义为 baseline LUT 减 treatment LUT，正值代表 treatment 改善。A 对参考检查回报标准化影响，B 对 A 检查尺度影响，训练组对 C 检查学习收益。', '',
        '## 训练行为与回报', '', '| 组别 | 最后 20 轮终点 LUT 均值 | 全部终点可行率 | 与参考不同的动作 / 3000 | actor 参数变化 RMS 均值 |',
        '|---|---:|---:|---:|---:|']
    for group, item in summary.items():
        changes = [v['actor_rms_change'] for v in validations if v['group'] == group]
        lines.append(f'| {group} | {item["late_endpoint_mean"]:.3f} | {item["endpoint_feasible_fraction"]:.2%} | '
                     f'{item["actions_different_from_reference"]} | {np.mean(changes):.6f} |')
    lines += ['', '最后 20 轮终点结果是训练过程诊断，未使用独立测试集，也未加入主指标的额外搜索预算。',
        'trajectories.csv 保存逐轮最好值、终点 LUT/层数/可行性、原始与处理后折扣回报均值及标准差、动作差异。', '',
        '| 组别 | 不可行动作 / 3000 | 前后均可行动作比例 |', '|---|---:|---:|']
    for group in roots:
        rows = [v for v in validations if v['group'] == group]
        lines.append(f'| {group} | {sum(v["infeasible_actions"] for v in rows)} | {sum(v["consecutive_feasible_actions"] for v in rows) / 3000:.2%} |')
    lines += ['', 'C 是冻结初始、依赖状态的随机策略，不是均匀随机搜索。actor、critic、Adam 状态每轮均验证不变；随机采样继续推进。',
        '三组同种子第一轮动作和映射指标与参考完全一致，确保初始策略、状态处理和采样保持一致。',
        '全部回报由日志重新计算，并与实际训练目标函数逐项核对；none 保留折扣回报，standardize 保留原有逐轮处理。', '',
        '## 验证与证据', '',
        f'- {tests["tests_passed"]} 项测试通过，包含原有奖励测试、新回报目标与配置校验、冻结状态、跨组恢复拒绝、三组真实工具续训一致性。',
        '- 12 次新旧运行的 12,000 个动作奖励、每轮奖励和、日志最优解和检查点均核验通过；12 个最优网表重新通过当前 ABC 的指标及 CEC 检查。',
        '- 参考结果、原提交源码与压缩包来源哈希核对通过，所有组使用与参考相同的工具二进制。',
        '- comparison.json / comparison.csv：逐种子结果、最优动作序列、均值/标准差及预定对照。',
        '- step-logs.tar.gz：四组全部逐步日志、最优映射网表、结果和原 CEC 记录，含逐文件 SHA256.json，已读回验证。',
        '- validation.json、tests.log、test-results.json、cec/：完整验证结果。',
        '- provenance.json、三个 YAML：命令、时间、配置指纹、源码及参考来源、环境和退出状态。',
        '- 完整检查点、初始网络快照和原始输出保存在 results/i2c-return-handling/。', '',
        '## 复现', '', '```bash', '.tools/conda-env/bin/python -B -m unittest discover -s tests -v',
        '.tools/conda-env/bin/python -B -u experiments/i2c-return-handling/run.py',
        '# 中断后，只能使用完全相同的组配置、源码和输出目录续跑',
        '.tools/conda-env/bin/python -B -u experiments/i2c-return-handling/run.py --resume',
        '.tools/conda-env/bin/python -B experiments/i2c-return-handling/evaluate.py', '```', '',
        '首次运行要求三个输出目录尚不存在；复核本次已完成数据仅运行 evaluate.py。参考数据仅复用，不重新训练。',
        '仅三个种子，结论是固定电路、预算、尺度下的描述性对照，不声称统计显著。没有在看到结果后改变预算、种子、尺度或搜索长度。',
        '', '## 结果解读', '',
        f'A 相对参考平均减少 {contrasts["A_vs_reference"]["mean_reduction"]:+.3f} LUT；取消逐轮回报标准化没有改善最好值，种子 1 反而多了 2 个 LUT。',
        f'B 相对 A 平均减少 {contrasts["B_vs_A"]["mean_reduction"]:+.3f} LUT；扩大尺度改变了采样动作和终点指标，但三个种子的最好值均持平。',
        f'参考组相对冻结组平均减少 {contrasts["reference_vs_C"]["mean_reduction"]:.3f} LUT，A、B 各减少 {contrasts["A_vs_C"]["mean_reduction"]:.3f} LUT。当前学习有小幅收益，主要搜索结果仍可由冻结初始策略达到。',
        '这组结果没有支持通过取消标准化或将 c 增至 10 改善 i2c 的最好可行 LUT，现有默认回报标准化保持不变。结果也不能单独确定学习收益较小的原因。', '']
    (HERE / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(summary=summary, contrasts=contrasts), indent=2))
    print('All 12 runs and 12,000 actions validated.')


if __name__ == '__main__':
    main()
