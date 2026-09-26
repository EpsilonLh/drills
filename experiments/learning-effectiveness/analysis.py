"""Rebuild all tables from committed-episode logs, preserving missing/infeasible runs."""
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from common import HERE, ROOT, GROUPS, POLICIES, EVAL_SEEDS, dump, load, now, returns_for, sha
from drills.fpga_session import FPGASession


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def hierarchical_bootstrap(differences, repetitions=10000, seed=20260927):
    """Paired differences indexed [training seed, evaluation seed]."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Bootstrap needs a complete finite paired matrix.')
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for i in range(repetitions):
        training = rng.integers(0, values.shape[0], size=values.shape[0])
        # Common evaluation seeds are shared across all models (uniform rows are identical).
        # Resample those columns jointly, preserving their dependence across training rows.
        evaluation = rng.integers(0, values.shape[1], size=(1, values.shape[1]))
        estimates[i] = values[training[:, None], evaluation].mean()
    return [float(v) for v in np.quantile(estimates, [0.025, 0.975])]


def rank(row):
    return FPGASession.rank(row)


def validate_log(path, cfg, circuit):
    rows = list(csv.DictReader(path.open()))
    if len(rows) != cfg['protocol']['iterations'] + 1:
        raise ValueError(f'Unexpected step count: {path}')
    reward_game = FPGASession.__new__(FPGASession)
    reward_game.method, reward_game.circuit = cfg['method'], circuit
    previous, processed, sequence = None, [], list(cfg['protocol']['initial_sequence'])
    for i, row in enumerate(rows):
        if int(row['iteration']) != i:
            raise ValueError(f'Noncontiguous log: {path}')
        luts, levels, reward = int(row['luts']), int(row['levels']), float(row['reward'])
        if previous:
            if row['optimization'] not in cfg['protocol']['actions']:
                raise ValueError('Action outside protocol.')
            reward_game.luts, reward_game.levels = previous['luts'], previous['levels']
            expected = reward_game._get_reward(luts, levels)
            sequence.append(row['optimization'])
        else:
            expected = 0
        if reward != expected:
            raise ValueError(f'Reward mismatch: {path}, step {i}')
        item = dict(iteration=i, optimization=row['optimization'], luts=luts, levels=levels,
                    reward=reward, feasible=levels <= circuit['max_levels'], sequence=list(sequence))
        processed.append(item)
        previous = item
    return processed


def generate(cfg):
    root = Path(cfg['runtime']['output_dir'])
    runs, curves, episodes, diagnostics, steps, validation, behavior = [], [], [], [], [], [], []
    for group in GROUPS:
        for name, circuit in cfg['protocol']['circuits'].items():
            for seed in cfg['protocol']['seeds']:
                folder = root / 'training' / group / name / f'seed-{seed}'
                result = read(folder / 'result.json') if (folder / 'result.json').exists() else dict(status='missing', episodes_completed=0, best=None)
                if result['status'] != 'complete' and (folder / 'checkpoint.pt').exists():
                    checkpoint = load(folder / 'checkpoint.pt')
                    result.update(status='incomplete', episodes_completed=checkpoint['episodes_completed'], best=checkpoint['best'])
                runs.append(dict(group=group, circuit=name, seed=seed, **{k: result.get(k) for k in
                    ('status', 'episodes_completed', 'best', 'training_seconds')}))
                best, candidate = None, 0
                diags = [json.loads(line) for line in (folder / 'diagnostics.jsonl').read_text().splitlines()] if (folder / 'diagnostics.jsonl').exists() else []
                diags = [d for d in diags if d['episode'] <= result['episodes_completed']]
                if len(diags) != result['episodes_completed']:
                    raise ValueError(f'Diagnostic coverage mismatch: {folder}')
                all_steps = [json.loads(line) for line in (folder / 'steps.jsonl').read_text().splitlines()] if (folder / 'steps.jsonl').exists() else []
                all_steps = [s for s in all_steps if s['episode'] <= result['episodes_completed']]
                if len(all_steps) != result['episodes_completed'] * cfg['protocol']['iterations']:
                    raise ValueError(f'Step evidence coverage mismatch: {folder}')
                by_episode = {e: [s for s in all_steps if s['episode'] == e] for e in range(1, result['episodes_completed'] + 1)}
                roundtrips, positive_roundtrips = 0, 0
                raw_inputs, normalized_inputs = [], []
                for ep in range(1, result['episodes_completed'] + 1):
                    logged = validate_log(folder / f'episodes/{ep}/log.csv', cfg, circuit)
                    rewards = [r['reward'] for r in logged[1:]]
                    raw_returns, normalized_returns = returns_for(rewards, cfg['method'])
                    diag = diags[ep - 1]
                    if diag['episode'] != ep or diag['reward_sum'] != sum(rewards):
                        raise ValueError('Episode diagnostics mismatch.')
                    for key, value in [('raw_returns_std', raw_returns.std()), ('processed_returns_std', normalized_returns.std())]:
                        if not np.isclose(diag[key], value, atol=1e-7):
                            raise ValueError('Return scale mismatch.')
                    if result.get('rewards') and result['rewards'][ep - 1] != sum(rewards):
                        raise ValueError('Checkpoint rewards mismatch.')
                    for i, logged_row in enumerate(logged):
                        if i == 0 and not cfg['protocol']['evaluation']['include_initial']:
                            continue
                        ranked = dict(**logged_row, episode=ep)
                        if best is None or rank(ranked) < rank(best):
                            best = ranked
                        if i:
                            evidence = by_episode[ep][i - 1]
                            if any(evidence[k] != logged_row[k] for k in ('luts', 'levels', 'reward', 'optimization')):
                                raise ValueError('Instrumented steps differ from environment logs.')
                            if cfg['protocol']['actions'][evidence['action']] != logged_row['optimization']:
                                raise ValueError('Action index differs from command.')
                            candidate += 1
                            curves.append(dict(group=group, circuit=name, seed=seed, candidate=candidate,
                                               luts=best['luts'], levels=best['levels'], feasible=best['feasible']))
                    for i in range(2, len(logged)):
                        a, b, c = logged[i - 2:i + 1]
                        if a['luts'] == c['luts'] and a['luts'] != b['luts'] and a['feasible'] and b['feasible'] and c['feasible']:
                            roundtrips += 1
                            positive_roundtrips += int(b['reward'] + c['reward'] > 0)
                    raw_inputs.extend(s['raw_state'] for s in by_episode[ep])
                    normalized_inputs.extend(s['normalized_state'] for s in by_episode[ep])
                    if not np.equal(by_episode[ep][0]['normalized_state'], 0).all():
                        raise ValueError('First normalized state is not zero.')
                    episode_best = min(logged if cfg['protocol']['evaluation']['include_initial'] else logged[1:], key=rank)
                    episodes.append(dict(group=group, circuit=name, seed=seed, episode=ep,
                        best_luts=episode_best['luts'], best_feasible=episode_best['feasible'],
                        terminal_luts=logged[-1]['luts'], terminal_levels=logged[-1]['levels'],
                        terminal_feasible=logged[-1]['feasible'], reward=sum(rewards)))
                    diagnostics.append(dict(group=group, circuit=name, seed=seed, **diag))
                if best is not None:
                    if any(best[k] != result['best'][k] for k in ('luts', 'levels', 'feasible', 'sequence', 'episode', 'iteration')):
                        raise ValueError(f'Reported best does not replay from logs: {folder}')
                for item in all_steps:
                    steps.append(dict(group=group, circuit=name, seed=seed, **item))
                if raw_inputs:
                    behavior.append(dict(group=group, circuit=name, seed=seed, positive_lut_roundtrips=positive_roundtrips,
                        lut_roundtrips=roundtrips, zero_reward_fraction=float(np.mean([s['reward'] == 0 for s in all_steps])),
                        raw_constant_features=[f for f, v in zip(cfg['method']['features'], np.ptp(raw_inputs, axis=0)) if v == 0],
                        normalized_zero_features=[f for f, v in zip(cfg['method']['features'], np.max(np.abs(normalized_inputs), axis=0)) if v == 0]))
                validation.append(dict(group=group, circuit=name, seed=seed, completed_episodes=result['episodes_completed'],
                                       replayed_steps=candidate, status=result['status'], best_replayed=best is not None))
    evaluation_rows, evaluation_cases = [], []
    for policy in POLICIES:
        for name, circuit in cfg['protocol']['circuits'].items():
            for seed in cfg['protocol']['seeds']:
                folder = root / 'evaluation' / policy / name / f'seed-{seed}'
                result = read(folder / 'result.json') if (folder / 'result.json').exists() else dict(status='missing', rollouts=[])
                records = result['rollouts']
                if records and [r['evaluation_seed'] for r in records] != EVAL_SEEDS:
                    raise ValueError('Evaluation seed coverage mismatch.')
                for row in records:
                    rollout = folder / f'rollout-{row["evaluation_seed"]}'
                    logged = validate_log(rollout / 'episodes/1/log.csv', cfg, circuit)
                    best = min(logged if cfg['protocol']['evaluation']['include_initial'] else logged[1:], key=rank)
                    if any(best[k] != row['best'][k] for k in ('luts', 'levels', 'feasible', 'sequence', 'iteration')):
                        raise ValueError('Evaluation best mismatch.')
                    if [cfg['protocol']['actions'][a] for a in row['actions']] != [r['optimization'] for r in logged[1:]]:
                        raise ValueError('Evaluation actions mismatch.')
                    if row['rewards'] != [r['reward'] for r in logged[1:]] or any(row['terminal'][k] != logged[-1][k] for k in ('luts', 'levels', 'feasible')):
                        raise ValueError('Evaluation terminal/reward mismatch.')
                    if 'Networks are equivalent' not in (rollout / 'equivalence.log').read_text():
                        raise ValueError('Missing evaluation CEC.')
                    evaluation_rows.append(dict(policy=policy, circuit=name, training_seed=seed,
                        evaluation_seed=row['evaluation_seed'], best_luts=row['best']['luts'], best_levels=row['best']['levels'],
                        best_feasible=row['best']['feasible'], terminal_luts=row['terminal']['luts'],
                        terminal_levels=row['terminal']['levels'], terminal_feasible=row['terminal']['feasible'],
                        reward=sum(row['rewards']), status=row['status']))
                all_feasible = len(records) == len(EVAL_SEEDS) and all(r['best']['feasible'] for r in records)
                terminal_feasible = len(records) == len(EVAL_SEEDS) and all(r['terminal']['feasible'] for r in records)
                evaluation_cases.append(dict(policy=policy, circuit=name, training_seed=seed, status=result['status'],
                    completed=len(records), feasible=sum(r['best']['feasible'] for r in records),
                    best_luts_mean=float(np.mean([r['best']['luts'] for r in records])) if all_feasible else None,
                    terminal_luts_mean=float(np.mean([r['terminal']['luts'] for r in records])) if terminal_feasible else None,
                    terminal_feasible=sum(r['terminal']['feasible'] for r in records)))
    comparisons = []
    for name in cfg['protocol']['circuits']:
        for baseline in ('initial', 'uniform'):
            matrix, wins, ties, losses, per_seed = [], 0, 0, 0, []
            for seed in cfg['protocol']['seeds']:
                left = [r for r in evaluation_rows if r['policy'] == baseline and r['circuit'] == name and r['training_seed'] == seed]
                right = [r for r in evaluation_rows if r['policy'] == 'trained-100' and r['circuit'] == name and r['training_seed'] == seed]
                valid = len(left) == len(right) == len(EVAL_SEEDS) and all(r['best_feasible'] for r in left + right)
                if valid:
                    difference = np.array([a['best_luts'] - b['best_luts'] for a, b in zip(left, right)])
                    matrix.append(difference)
                    wins += int((difference > 0).sum()); ties += int((difference == 0).sum()); losses += int((difference < 0).sum())
                    per_seed.append(dict(seed=seed, luts_reduction=float(difference.mean()),
                                         percent=100 * float(difference.mean()) / float(np.mean([r['best_luts'] for r in left]))))
            complete = len(matrix) == len(cfg['protocol']['seeds'])
            comparisons.append(dict(circuit=name, baseline=baseline, treatment='trained-100',
                mean_luts_reduction=float(np.mean(matrix)) if complete else None,
                ci95=hierarchical_bootstrap(matrix) if complete else None, per_seed=per_seed,
                wins=wins, ties=ties, losses=losses, complete_feasible=complete))
    summary = dict(training=runs, evaluation=evaluation_cases, comparisons=comparisons, behavior=behavior,
                   diagnostic_summary=[{k: r[k] for k in ('group', 'circuit', 'seed', 'episode', 'policy_kl',
                        'actor_drift_rms', 'critic_drift_rms', 'entropy')} for r in diagnostics if r['episode'] == 100],
                   training_complete=sum(r['status'] == 'complete' for r in runs),
                   evaluation_complete=sum(r['status'] == 'complete' for r in evaluation_cases))
    dump(HERE / 'summary.json', summary)
    dump(HERE / 'validation.json', dict(time=now(), training=validation, evaluation_rollouts=len(evaluation_rows),
                                      evaluation_cec=len(evaluation_rows), reward_and_best_replay=True))
    write_csv(HERE / 'training.csv', [dict(group=r['group'], circuit=r['circuit'], seed=r['seed'], status=r['status'],
        episodes=r['episodes_completed'], luts=(r['best'] or {}).get('luts'), levels=(r['best'] or {}).get('levels'),
        feasible=(r['best'] or {}).get('feasible'), training_seconds=r['training_seconds']) for r in runs])
    for filename, rows in [('trajectories.csv', curves), ('episodes.csv', episodes), ('diagnostics.csv', diagnostics),
                           ('steps.csv', steps), ('evaluation.csv', evaluation_rows), ('comparisons.csv', comparisons)]:
        write_csv(HERE / filename, rows)
    write_report(summary)
    archive_evidence(root)
    print(f'Report: {HERE / "report.md"}; training {summary["training_complete"]}/27; evaluation {summary["evaluation_complete"]}/36', flush=True)


def format_value(value):
    return '未汇总' if value is None else f'{value:.3f}'


def write_report(summary):
    lines = ['# DRiLLS 学习有效性实验报告', '', '## Material Passport', '',
        '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: run + validate',
        '- Origin Date: 2026-09-27',
        '- Verification Status: ' + ('VERIFIED（短程原实现一致性及断点恢复；完整实验日志复核）' if summary['training_complete'] == 27 and summary['evaluation_complete'] == 36 else 'INCOMPLETE'),
        '- Version Label: learning_effectiveness_v1', '', '## 主要发现', '',
        f'训练搜索完成 {summary["training_complete"]}/27 次，独立策略评估完成 {summary["evaluation_complete"]}/36 组。'
        '本报告分别检验权重更新、策略改变和搜索收益，不把参数变化或奖励上升单独当作学习有效的证据。', '',
        '## 条件与比较口径', '',
        '使用原始 adf7bef 算法：int2float / i2c / max，深度限制 3 / 4 / 41，训练种子 0 / 1 / 2，'
        '100 轮 × 10 步，LUT6；原奖励、episode_welford 状态归一化、标准化回报、网络与 Adam 设置不变。'
        '初始映射可参与最优值。三组为训练、初始化权重冻结、均匀随机，后三者各运行相同搜索预算。', '',
        '训练/冻结初始网络、Adam 和采样 RNG 相同，第一轮轨迹相同。冻结网络按状态输出概率，'
        '并不等于均匀随机。第 0 / 50 / 100 轮模型分别用新种子 10000–10029 进行 30 条无更新的 10 步评估；'
        '冻结三个快照一致，故共享初始化策略评估。评估只涉及训练使用的电路，不是跨电路泛化测试。', '',
        '主指标为可行率和相同候选预算下的最优可行 LUT；完整序列终点为辅助指标。'
        '任何组有缺失或不可行时，不用筛掉这些运行后的均值判断优劣。', '',
        '## 训练搜索结果', '', '| 电路 | 组别 | seed 0 LUT | seed 1 LUT | seed 2 LUT | 均值 ± 总体标准差 |',
        '|---|---|---:|---:|---:|---:|']
    for name in ('int2float', 'i2c', 'max'):
        for group in GROUPS:
            rows = [r for r in summary['training'] if r['circuit'] == name and r['group'] == group]
            values = [(r['best'] or {}).get('luts') for r in rows]
            valid = len(rows) == 3 and all(r['status'] == 'complete' and r['best']['feasible'] for r in rows)
            score = f'{np.mean(values):.3f} ± {np.std(values):.3f}' if valid else '未汇总'
            lines.append(f'| {name} | {group} | ' + ' | '.join(str(v) for v in values) + f' | {score} |')
    lines += ['', '这些分数是 1,000 个动作候选中找到的最佳解，不能单独证明策略质量改善。', '',
              '![按候选数量比较搜索轨迹](search-curves.svg)', '', '## 独立策略评估', '',
              '| 电路 | 训练种子 | 初始/冻结均值 | 50 轮均值 | 100 轮均值 | 均匀随机均值 |',
              '|---|---:|---:|---:|---:|---:|']
    for name in ('int2float', 'i2c', 'max'):
        for seed in (0, 1, 2):
            values = [next(r['best_luts_mean'] for r in summary['evaluation'] if r['circuit'] == name and
                           r['training_seed'] == seed and r['policy'] == policy) for policy in POLICIES]
            lines.append(f'| {name} | {seed} | ' + ' | '.join(format_value(v) for v in values) + ' |')
    lines += ['', '每个均值来自固定模型的 30 条评估序列，各序列取包括初始映射在内的最佳可行解。'
              '底层逐条可行率、终点、奖励均在 evaluation.csv；未完成/不可行不被删除。', '',
              '| 电路 | 对照策略 | 100 轮模型平均减少 LUT | 探索性 95% 区间 | 评估胜/平/负 | 逐训练种子减少 LUT / 百分比 |',
              '|---|---|---:|---|---|---|']
    for row in summary['comparisons']:
        interval = '未汇总' if row['ci95'] is None else f'[{row["ci95"][0]:.3f}, {row["ci95"][1]:.3f}]'
        effects = '; '.join(f'{p["seed"]}: {p["luts_reduction"]:+.3f} / {p["percent"]:+.3f}%' for p in row['per_seed'])
        lines.append(f'| {row["circuit"]} | {row["baseline"]} | {format_value(row["mean_luts_reduction"])} | '
                     f'{interval} | {row["wins"]}/{row["ties"]}/{row["losses"]} | {effects} |')
    lines += ['', '正值表示训练更好。配对分层 bootstrap 先重采样 3 个训练种子，再重采样配对评估序列，'
              '所有训练种子共同重采样同一组评估种子列，以保留共同随机数及均匀随机基线重复使用的相关性。'
              '重复 10,000 次，分析种子 20260927。只有 3 个独立训练种子，区间为探索性估计；'
              '90 条评估序列不能当作 90 次独立训练。未预设实用收益门槛，不作统计等效判断。'
              '评估原则参照 [Agarwal 等](https://arxiv.org/abs/2108.13264)和 '
              '[Henderson 等](https://arxiv.org/abs/1709.06560)。', '',
              '## 学习链路与机制诊断', '',
              '| 电路 | seed | Actor 参数变化 RMS | Critic 参数变化 RMS | 固定探针 KL | 策略熵 |',
              '|---|---:|---:|---:|---:|---:|']
    for row in summary['diagnostic_summary']:
        if row['group'] == 'trained':
            lines.append(f'| {row["circuit"]} | {row["seed"]} | {row["actor_drift_rms"]:.6f} | '
                         f'{row["critic_drift_rms"]:.6f} | {row["policy_kl"]:.6f} | {row["entropy"]:.6f} |')
    lines += ['', '诊断使用第一轮保存的归一化输入作为固定探针，逐轮记录损失、梯度、更新量、回报尺度和动作概率，'
              '不消耗采样随机数。固定探针只能覆盖这些输入附近的策略变化，不能证明全部状态空间都学得更好。', '',
              '| 电路 | 组别 | 零奖励比例 | LUT 往返次数 | 其中两步正奖励次数 |',
              '|---|---|---:|---:|---:|']
    for name in ('int2float', 'i2c', 'max'):
        for group in GROUPS:
            rows = [r for r in summary['behavior'] if r['circuit'] == name and r['group'] == group]
            if rows:
                lines.append(f'| {name} | {group} | {np.mean([r["zero_reward_fraction"] for r in rows]):.2%} | '
                             f'{sum(r["lut_roundtrips"] for r in rows)} | {sum(r["positive_lut_roundtrips"] for r in rows)} |')
    lines += ['', '每轮首个状态归一化为全零；固定输入/输出数等特征可能持续为零。'
              'behavior 字段保存逐种子的常量特征名单。可行区域减少 LUT 奖励 +3，增加奖励 −1，'
              '因此 LUT 数降低再恢复的两步过程获得正奖励。这里检测的是 LUT 数往返，不能据此断言网表完全回到原状态。'
              '奖励与面积目标错位、状态信息损失、搜索最佳值掩盖策略变化是机制解释候选；本实验未修改这些设置来验证因果。', '',
              '## 证据、限制与复现', '',
              '测试覆盖原实现与诊断路径的精确权重/Adam/RNG/动作一致性、冻结、第一轮配对、三组真实工具断点恢复、'
              '非法续跑拒绝和受控正优势更新。完整实验按逐步日志重新计算奖励、回报尺度、最优值和终点；'
              '训练及评估导出的最佳映射通过 ABC CEC。测试和验证记录见 tests.log、test-results.json、validation.json。', '',
              '完整实验每个训练种子只运行一次，未另做全部 27 次同种子重复；可复现性证据来自短程精确回归与断点恢复，'
              '不能把“完整日志复核”表述为“全部实验独立重复成功”。耗时包含额外诊断开销且并发共享资源，'
              '本报告依据候选预算比较搜索质量，不根据耗时给算法效率排名。', '',
              '核查了 11 类统计解释风险：辛普森悖论（逐电路报告）、生态谬误（训练种子是独立单位）、'
              '选择偏差与碰撞变量（不按结果筛样本）、基率忽略（完整可行率）、回归均值（固定种子与对照）、'
              '幸存者偏差（保留失败）、多重搜索与分析分叉（预定六个探索性对照，不据此宣称显著）、'
              '相关当因果与反向因果（仅对受控冻结干预和已测范围解释）。覆盖 11/11，少训练种子限制仍然存在。', '',
              '源码/配置/工具/电路指纹和完整命令见 provenance.json、evaluation-provenance.json；'
              '压缩证据 evidence.tar.gz 内有 SHA256.json。完整权重和原始输出保存在 results/learning-effectiveness/。'
              '历史 results/ 其他实验未改写。AI 用于实现实验、核对日志、统计与报告撰写。', '',
              '```bash', '.tools/conda-env/bin/python -B -m unittest discover -s tests -v',
              '.tools/conda-env/bin/python -B -u experiments/learning-effectiveness/run.py',
              '# 中断后，仅相同源码/配置/工具允许续跑',
              '.tools/conda-env/bin/python -B -u experiments/learning-effectiveness/run.py --resume',
              '.tools/conda-env/bin/python -B -u experiments/learning-effectiveness/evaluate.py',
              '# 已有完整评估时，重读证据生成报告',
              '.tools/conda-env/bin/python -B experiments/learning-effectiveness/evaluate.py --report-only', '```', '']
    (HERE / 'report.md').write_text('\n'.join(lines))


def archive_evidence(root):
    names = {'log.csv', 'result.json', 'rollout.json', 'best.v', 'best-mapped.v', 'best.json', 'equivalence.log',
             'parameter-validation.json', 'steps.jsonl', 'diagnostics.jsonl', 'experiment.json', 'evaluation.json', 'baseline.json'}
    paths = sorted(p for p in root.rglob('*') if p.is_file() and (p.name in names or 'logs' in p.relative_to(root).parts
                                                              or 'baseline' in p.relative_to(root).parts))
    hashes = {}
    with tarfile.open(HERE / 'evidence.tar.gz', 'w:gz') as archive:
        for path in paths:
            label = str(path.relative_to(root))
            hashes[label] = sha(path)
            archive.add(path, arcname=label)
        payload = json.dumps(hashes, indent=2).encode()
        metadata = tarfile.TarInfo('SHA256.json')
        metadata.size = len(payload)
        archive.addfile(metadata, io.BytesIO(payload))
    with tarfile.open(HERE / 'evidence.tar.gz', 'r:gz') as archive:
        for label, digest in hashes.items():
            if hashlib.sha256(archive.extractfile(label).read()).hexdigest() != digest:
                raise ValueError('Evidence archive readback failed.')
    dump(HERE / 'archive-validation.json', dict(files=len(hashes), sha256=sha(HERE / 'evidence.tar.gz'), readback_verified=True))
