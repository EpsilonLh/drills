"""Audit original-reward learning at two horizons; export all prescribed comparisons."""
import csv
import copy
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
                 state_hash, verify_reference, PREVIOUS_FROZEN, ORIGINAL_COMMIT)
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
    require(group == 'trained-10' or result['group'] == group, f'Group identity mismatch: {folder}')
    limit = protocol['circuits']['i2c']['max_levels']
    steps = protocol['iterations']
    best, best_rank, running_best = None, None, None
    actions, episodes, diagnostics, files, trajectory = [], [], [], [], []
    infeasible, eligible, error = 0, 0, 0.0
    for episode in range(1, 101):
        path = folder / 'episodes' / str(episode) / 'log.csv'
        with path.open(newline='') as source:
            rows = list(csv.DictReader(source))
        files.append((path, f'{group}/seed-{seed}/episodes/{episode}/log.csv'))
        require(len(rows) == steps + 1 and [int(row['iteration']) for row in rows] == list(range(steps + 1)), f'Invalid steps: {path}')
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
            if step:
                trajectory.append(dict(group=group, seed=seed, episode=episode, iteration=step,
                    actions_completed=(episode - 1) * steps + step, optimization=row['optimization'],
                    luts=luts, levels=levels, feasible=feasible, reward=reward,
                    running_best_feasible_luts=running_best))
            previous = (luts, levels)
        require(math.isclose(sum(rewards), result['rewards'][episode - 1], rel_tol=1e-12, abs_tol=1e-12), f'Reward sum mismatch: {path}')
        raw, processed = returns(rewards, config)
        diagnostics.append(dict(group=group, seed=seed, episode=episode, actions_completed=episode * steps,
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
    if group != 'trained-10':
        initial = torch.load(folder / 'initial.pt', map_location='cpu', weights_only=True)
        require(initial['optimizer']['state'] == {}, f'Initial Adam is not empty: {folder}')
        require(torch.equal(initial['rng_state'], torch.Generator().manual_seed(seed).get_state()), f'Initial RNG changed: {folder}')
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
    for name in ['best.v', 'best-mapped.v']:
        require(saved['best_netlists'][name] == (folder / name).read_text(), f'Export differs from checkpoint: {folder}/{name}')
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
        final_network_hash=state_hash(saved['network']), initial_network_hash=state_hash(initial_network),
        checkpoint_sha256=sha256(folder / 'checkpoint.pt'), **weight_changes)
    return result, diagnostics, actions, episodes, files, validation, trajectory


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


def write_csv(path, rows, compressed=False):
    if compressed:
        with path.open('wb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as destination:
                writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
    else:
        with path.open('w', newline='') as destination:
            writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


def main():
    provenance = json.loads((HERE / 'provenance.json').read_text())
    require(provenance.get('exit_code') == 0 and all(g['exit_code'] == 0 for g in provenance['groups'].values()), 'Experiment incomplete.')
    require(source_hashes() == provenance['source_hashes'], 'Training sources changed.')
    for name, expected in provenance['config_hashes'].items():
        require(sha256(HERE / name) == expected, f'Experiment YAML changed: {name}')
    require(verify_reference() == provenance['reference_hashes'], 'Historical evidence changed.')
    torch.set_num_threads(1)
    configs = {'trained-10': json.loads((REFERENCE / 'experiment.json').read_text())['config']}
    roots = {'trained-10': REFERENCE}
    for group, (label, _, _) in GROUPS.items():
        root = ROOT / 'results/i2c-learning-horizon' / label
        configs[group] = json.loads((root / 'experiment.json').read_text())['config']
        roots[group] = root
        require(normalized(configs[group]) == normalized(group_config(group)), f'Group configuration changed: {group}')
        require(sha256(ROOT / provenance['groups'][group]['log']) == provenance['groups'][group]['log_sha256'], f'Training log changed: {group}')
    configs = {g: normalized(c) for g, c in configs.items()}
    for group, (label, steps, learning) in GROUPS.items():
        expected = copy.deepcopy(configs['trained-10'])
        expected['protocol']['circuits'] = {'i2c': expected['protocol']['circuits']['i2c']}
        expected['protocol']['iterations'] = steps
        expected['method']['reward']['feasible_mode'] = 'table'
        expected['method']['learning_enabled'] = learning
        expected['runtime']['output_dir'] = str(roots[group])
        actual = copy.deepcopy(configs[group])
        actual['runtime'].pop('experiment_fingerprint')
        require(actual == expected, f'Non-prescribed changes relative to original training: {group}')
    historical = json.loads((ROOT / 'results/new-state/baseline-provenance.json').read_text())
    require(sha256(ROOT / 'drills/features.py') == historical['source_hashes']['drills/features.py'], 'Original feature extraction changed.')
    results, actions, episodes, diagnostics, trajectories, validations, files = {}, {}, {}, [], [], [], []
    for group, root in roots.items():
        for seed in [0, 1, 2]:
            result, rows, sequence, logs, archived, validation, steps = validate_run(group, root, configs[group], seed, provenance['tools']['abc']['path'])
            results[group, seed] = result
            actions[group, seed], episodes[group, seed] = sequence, logs
            diagnostics.extend(rows)
            trajectories.extend(steps)
            validations.append(validation)
            files.extend(archived)
        for name in ['experiment.json', 'baseline.json', 'results.json', 'results.md']:
            files.append((root / name, f'{group}/{name}'))
        require(json.loads((root / 'baseline.json').read_text())['i2c'] ==
                json.loads((REFERENCE / 'baseline.json').read_text())['i2c'], f'resyn2 changed: {group}')

    metrics = lambda rows: [(r['optimization'], r['luts'], r['levels']) for r in rows]
    for seed in [0, 1, 2]:
        ref = episodes['trained-10', seed][0]
        for group in GROUPS:
            require(metrics(episodes[group, seed][0][:11]) == metrics(ref), f'First 10 steps differ: {group}/{seed}')
        require(episodes['frozen-10', seed][0] == ref, f'First 10-step episode differs: {seed}')
        require(episodes['trained-30', seed][0] == episodes['frozen-30', seed][0], f'First 30-step episode differs: {seed}')
        snapshots = [torch.load(roots[g] / 'i2c' / f'seed-{seed}/initial.pt', map_location='cpu', weights_only=True) for g in GROUPS]
        for key in ['network', 'optimizer', 'rng_state']:
            require(len({state_hash(s[key]) for s in snapshots}) == 1, f'Initial states differ: {key}/{seed}')
        for episode in range(1, 101):
            old_file = PREVIOUS_FROZEN / 'i2c' / f'seed-{seed}/episodes/{episode}/log.csv'
            with old_file.open(newline='') as source:
                old_rows = list(csv.DictReader(source))
            require(metrics(old_rows) == metrics(episodes['frozen-10', seed][episode - 1]), f'Frozen reward invariance failed: {seed}/{episode}')
        old_saved = torch.load(PREVIOUS_FROZEN / 'i2c' / f'seed-{seed}/checkpoint.pt', map_location='cpu', weights_only=True)
        new_saved = torch.load(roots['frozen-10'] / 'i2c' / f'seed-{seed}/checkpoint.pt', map_location='cpu', weights_only=True)
        for key in ['network', 'optimizer', 'rng_state', 'best']:
            require(state_hash(old_saved[key]) == state_hash(new_saved[key]), f'Frozen final states differ: {key}/{seed}')

    for row in diagnostics:
        group, seed, ep = row['group'], row['seed'], row['episode']
        steps = configs[group]['protocol']['iterations']
        frozen = f'frozen-{steps}'
        start = (ep - 1) * steps
        row['actions_different_from_same_horizon_frozen'] = sum(a != b for a, b in zip(
            actions[group, seed][start:start + steps], actions[frozen, seed][start:start + steps]))
    write_csv(HERE / 'episodes.csv', diagnostics)
    write_csv(HERE / 'trajectories.csv.gz', trajectories, compressed=True)
    summary, pairs = {}, []
    for group in roots:
        values = [results[group, seed]['best']['luts'] for seed in [0, 1, 2]]
        rows = [r for r in diagnostics if r['group'] == group]
        checks = [v for v in validations if v['group'] == group]
        summary[group] = dict(luts=values, mean=float(np.mean(values)), std=float(np.std(values)),
            steps=configs[group]['protocol']['iterations'], learning_enabled=configs[group]['method'].get('learning_enabled', True),
            late_endpoint_mean=float(np.mean([r['endpoint_luts'] for r in rows if r['episode'] > 80])),
            endpoint_feasible_fraction=float(np.mean([r['endpoint_feasible'] for r in rows])),
            infeasible_actions=sum(v['infeasible_actions'] for v in checks), actions_checked=sum(v['actions_checked'] for v in checks),
            different_actions=sum(r['actions_different_from_same_horizon_frozen'] for r in rows),
            actor_rms_change=float(np.mean([v['actor_rms_change'] for v in checks])),
            critic_rms_change=float(np.mean([v['critic_rms_change'] for v in checks])))
        for seed in [0, 1, 2]:
            result = results[group, seed]
            frozen = f'frozen-{configs[group]["protocol"]["iterations"]}'
            pairs.append(dict(group=group, seed=seed, luts=result['best']['luts'], levels=result['best']['levels'],
                learning_gain=results[frozen, seed]['best']['luts'] - result['best']['luts'], best=result['best']))
    contrasts = {}
    for name, baseline, treatment in [('learning_10', 'frozen-10', 'trained-10'), ('learning_30', 'frozen-30', 'trained-30'),
                                     ('longer_trained', 'trained-10', 'trained-30'), ('longer_frozen', 'frozen-10', 'frozen-30')]:
        delta = [results[baseline, seed]['best']['luts'] - results[treatment, seed]['best']['luts'] for seed in [0, 1, 2]]
        contrasts[name] = dict(baseline=baseline, treatment=treatment, reductions=delta,
            mean_reduction=float(np.mean(delta)),
            percent_reduction=100 * float(np.mean(delta)) / summary[baseline]['mean'],
            wins=sum(d > 0 for d in delta), ties=delta.count(0), losses=sum(d < 0 for d in delta))
    changes = [b - a for a, b in zip(contrasts['learning_10']['reductions'], contrasts['learning_30']['reductions'])]
    comparison = dict(summary=summary, contrasts=contrasts, pairs=pairs,
        learning_gain_change=dict(per_seed=changes, mean=float(np.mean(changes))),
        metric='Best feasible training-time LUTs; learning gain = frozen - trained',
        budget='100 episodes, seeds 0/1/2; 1000 actions/seed at 10 steps and 3000 at 30 steps')
    dump(HERE / 'comparison.json', comparison)
    write_csv(HERE / 'comparison.csv', [{k: v for k, v in r.items() if k != 'best'} for r in pairs])
    archive = archive_files(files)
    tests = json.loads((HERE / 'test-results.json').read_text())
    require(tests['exit_code'] == 0 and tests['tests_skipped'] == 0, 'Tests incomplete.')
    for name, expected in tests['source_hashes'].items():
        require(sha256(ROOT / name) == expected, f'Test source changed: {name}')
    require(sha256(HERE / 'tests.log') == tests['log_sha256'], 'Test log changed.')
    total = sum(v['actions_checked'] for v in validations)
    require(total == 24000, 'Wrong prescribed total budget.')
    dump(HERE / 'validation.json', dict(recorded_at=datetime.now(timezone.utc).isoformat(), tests=tests,
        evaluator_sha256=sha256(Path(__file__)), runs=validations, total_actions_checked=total,
        original_algorithm_commit=ORIGINAL_COMMIT, first_episode_pairs_match=True, first_ten_steps_match_history=True,
        initial_states_match=True, frozen_reward_invariance=True, historical_outputs_unchanged=True,
        archive=archive, trajectories_sha256=sha256(HERE / 'trajectories.csv.gz')))
    write_report(summary, contrasts, changes, tests)
    print(json.dumps(dict(summary=summary, contrasts=contrasts, learning_gain_change=changes), indent=2))
    print('All 12 runs and 24,000 actions validated.')


def write_report(summary, contrasts, changes, tests):
    lines = ['# 原始奖励下的学习收益：10 步与 30 步', '',
        '固定 i2c、层数上限 4、种子 0/1/2、100 轮。所有组使用原始离散奖励表与 standardize 回报处理。',
        '10 步训练复用修改奖励前的完整结果，其余九次运行从头初始化；每种子的搜索预算由 1,000 增至 3,000 个动作。', '',
        '| 组别 | seed 0 | seed 1 | seed 2 | 最好可行 LUT 均值 ± 总体标准差 |', '|---|---:|---:|---:|---:|']
    for group, item in summary.items():
        lines.append(f'| {group} | ' + ' | '.join(map(str, item['luts'])) + f' | {item["mean"]:.3f} ± {item["std"]:.3f} |')
    lines += ['', '| 对照 | 逐种子 LUT 减少量 | 平均减少 LUT | 胜/平/负 |', '|---|---|---:|---:|']
    for name, item in contrasts.items():
        lines.append(f'| {name} | {item["reductions"]} | {item["mean_reduction"]:+.3f} | {item["wins"]}/{item["ties"]}/{item["losses"]} |')
    g10, g30 = contrasts['learning_10']['mean_reduction'], contrasts['learning_30']['mean_reduction']
    if g30 < 0:
        verdict = f'30 步训练组反而比同预算冻结组平均多 {abs(g30):.3f} LUT，延长序列没有体现出额外学习收益。'
    elif g30 == 0:
        verdict = '30 步训练组与同预算冻结组的最好值均值持平，延长序列没有体现出平均学习收益。'
    else:
        verdict = f'30 步训练组比同预算冻结组平均少 {g30:.3f} LUT；收益是否增大应结合下面的配对差值判断。'
    lines += ['', '减少量为 baseline LUT 减 treatment LUT。learning 对照的 baseline 是同长度冻结组，正值表示训练更好；longer 对照的 baseline 是 10 步组。', '',
        '## 结果解读', '',
        f'原始奖励在 10 步下，学习平均减少 {g10:+.3f} LUT（{contrasts["learning_10"]["percent_reduction"]:+.3f}%）；30 步下平均减少 {g30:+.3f} LUT（{contrasts["learning_30"]["percent_reduction"]:+.3f}%）。',
        verdict,
        f'延长后的学习收益变化为 {changes}，均值 {np.mean(changes):+.3f} LUT。三种子结果只能作为本电路与固定预算下的描述性证据。',
        '10 步冻结组的全部动作、LUT 和层数与上一轮冻结组一致，确认奖励和回报处理不会影响不更新网络的策略采样。',
        '跨长度改善同时包含更长序列、三倍动作预算，以及更长回报序列和损失求和的影响。不能将跨长度改善直接归因于强化学习。',
        '冻结策略为按状态输出动作概率的初始随机网络。没有载入已训练权重，也不是均匀随机搜索。', '',
        '## 训练行为', '', '| 组别 | 最后 20 轮终点 LUT 均值 | 终点可行率 | 不可行动作 / 总动作 | 相对同长度冻结组不同动作 | actor / critic 参数变化 RMS |',
        '|---|---:|---:|---:|---:|---:|']
    for group, item in summary.items():
        lines.append(f'| {group} | {item["late_endpoint_mean"]:.3f} | {item["endpoint_feasible_fraction"]:.2%} | '
                     f'{item["infeasible_actions"]}/{item["actions_checked"]} | {item["different_actions"]} | '
                     f'{item["actor_rms_change"]:.6f} / {item["critic_rms_change"]:.6f} |')
    lines += ['', '逐轮终点指标为训练诊断；主指标仍为固定预算内遇到的最好可行 LUT。',
        'trajectories.csv.gz 保存全部 24,000 个动作后的最好值与指标；episodes.csv 保存每轮终点、原始／处理后回报尺度与动作差异。', '',
        '冻结组的处理后回报按配置离线重算用于核验，不用于网络更新。', '',
        '![各个种子的最好可行 LUT 轨迹](best-trajectories.svg)', '',
        '## 验证与来源', '',
        f'- {tests["tests_passed"]} 项测试通过，无跳过；包含与 adf7bef 原奖励及标准化回报／网络更新的精确一致性检查，以及三组真实工具续跑一致性。',
        '- 12 次新旧运行的 24,000 个动作奖励、每轮奖励和、检查点、日志最优解及导出内容核验通过；12 个最优网表重新通过 LUT、层数和 ABC CEC 检查。',
        '- 同种子所有新组初始网络、Adam、RNG 相同；训练／冻结第一轮一致，30 步第一轮前 10 步与历史一致。',
        '- 冻结组逐轮检查 actor、critic 和 Adam 不变，采样 RNG 正常推进；10 步冻结组最终状态亦与上一轮冻结组一致。',
        '- 原始训练源码和结果由历史清单及已提交证据哈希验证；旧结果未改写。当前工具二进制与最近两轮相同，原始历史训练清单未记录工具版本。',
        '- provenance.json、三个 YAML：命令、时间、配置指纹、来源哈希和环境；validation.json、tests.log、test-results.json、cec/：验证证据。',
        '- comparison.json / comparison.csv：逐种子结果与预定对照；step-logs.tar.gz：全部逐步日志、结果、最优映射网表及原 CEC，含 SHA256.json 并已读回核验。',
        '- best-trajectories.svg / .png：用 ReportLab 绘制的逐种子轨迹；plot-validation.json 记录绘图命令、环境与数据／输出哈希。',
        '- 完整模型、初始权重快照和原始输出保存在 results/i2c-learning-horizon/。', '',
        '## 复现', '', '```bash', '.tools/conda-env/bin/python -B -m unittest discover -s tests -v',
        '.tools/conda-env/bin/python -B -u experiments/i2c-learning-horizon/run.py',
        '# 中断后使用相同组、配置、源码续跑', '.tools/conda-env/bin/python -B -u experiments/i2c-learning-horizon/run.py --resume',
        '.tools/conda-env/bin/python -B experiments/i2c-learning-horizon/evaluate.py', '```', '',
        '首次运行要求新输出目录不存在；已有本次完整结果时仅运行 evaluate.py 复核。未根据结果改变种子、步数或轮数。',
        '绘图另运行 plot.py，使用 Codex 自带的 ReportLab 环境；实际 Python 路径与完整命令记录于 plot-validation.json。', '']
    (HERE / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
