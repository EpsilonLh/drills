"""Rebuild all tables from committed-episode logs, preserving missing/infeasible runs."""
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import tarfile
import sys

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
    if np.ptp(values) == 0:
        return [float(values.flat[0]), float(values.flat[0])]
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


def policy_comparison(rows, name, baseline, seeds):
    lut_matrix, feasible_matrix, scores, per_seed = [], [], [], []
    for seed in seeds:
        left = [r for r in rows if r['policy'] == baseline and r['circuit'] == name and r['training_seed'] == seed]
        right = [r for r in rows if r['policy'] == 'trained-100' and r['circuit'] == name and r['training_seed'] == seed]
        complete = len(left) == len(right) == len(EVAL_SEEDS) and all(r['task_status'] == 'complete' for r in left + right)
        if not complete:
            continue
        if [r['evaluation_seed'] for r in left] != [r['evaluation_seed'] for r in right]:
            raise ValueError('Paired evaluation seeds are misaligned.')
        feasible = np.array([100 * (int(b['best_feasible']) - int(a['best_feasible'])) for a, b in zip(left, right)])
        feasible_matrix.append(feasible)
        differences = None
        if all(r['best_feasible'] for r in left + right):
            differences = np.array([a['best_luts'] - b['best_luts'] for a, b in zip(left, right)])
            lut_matrix.append(differences)
        for a, b in zip(left, right):
            baseline_rank = rank(dict(feasible=a['best_feasible'], luts=a['best_luts'], levels=a['best_levels']))
            trained_rank = rank(dict(feasible=b['best_feasible'], luts=b['best_luts'], levels=b['best_levels']))
            scores.append(int(trained_rank < baseline_rank) - int(trained_rank > baseline_rank))
        per_seed.append(dict(seed=seed, luts_reduction=float(differences.mean()) if differences is not None else None,
            percent=100 * float(differences.mean()) / float(np.mean([r['best_luts'] for r in left])) if differences is not None else None,
            baseline_feasible=sum(r['best_feasible'] for r in left), trained_feasible=sum(r['best_feasible'] for r in right),
            feasibility_gain_pp=float(feasible.mean())))
    complete = len(feasible_matrix) == len(seeds)
    all_feasible = len(lut_matrix) == len(seeds)
    return dict(circuit=name, baseline=baseline, treatment='trained-100',
        mean_luts_reduction=float(np.mean(lut_matrix)) if all_feasible else None,
        ci95=hierarchical_bootstrap(lut_matrix) if all_feasible else None,
        feasibility_gain_pp=float(np.mean(feasible_matrix)) if complete else None,
        feasibility_ci95=hierarchical_bootstrap(feasible_matrix) if complete else None,
        baseline_feasible=sum(p['baseline_feasible'] for p in per_seed),
        trained_feasible=sum(p['trained_feasible'] for p in per_seed), per_seed=per_seed,
        wins=sum(v > 0 for v in scores) if complete else None, ties=sum(v == 0 for v in scores) if complete else None,
        losses=sum(v < 0 for v in scores) if complete else None, complete=complete, complete_feasible=all_feasible)


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
    training_cec = 0
    for group in GROUPS:
        for name, circuit in cfg['protocol']['circuits'].items():
            for seed in cfg['protocol']['seeds']:
                folder = root / 'training' / group / name / f'seed-{seed}'
                status_record = read(folder / 'status.json') if (folder / 'status.json').exists() else {}
                result = read(folder / 'result.json') if (folder / 'result.json').exists() else dict(
                    status='failed' if status_record.get('status') == 'failed' else 'missing',
                    error=status_record.get('error'), episodes_completed=0, best=None)
                if result['status'] != 'complete' and (folder / 'checkpoint.pt').exists():
                    checkpoint = load(folder / 'checkpoint.pt')
                    result.update(status='failed' if result['status'] == 'failed' else 'incomplete',
                                  episodes_completed=checkpoint['episodes_completed'], best=checkpoint['best'])
                runs.append(dict(group=group, circuit=name, seed=seed, **{k: result.get(k) for k in
                    ('status', 'episodes_completed', 'best', 'training_seconds', 'error')}))
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
                if result['status'] == 'complete':
                    if 'Networks are equivalent' not in (folder / 'equivalence.log').read_text():
                        raise ValueError(f'Missing training CEC: {folder}')
                    training_cec += 1
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
                if (folder / 'result.json').exists():
                    result = read(folder / 'result.json')
                else:
                    status_record = read(folder / 'status.json') if (folder / 'status.json').exists() else {}
                    partial = [read(p) for p in sorted(folder.glob('rollout-*/rollout.json'))]
                    result = dict(status='failed' if status_record.get('status') == 'failed' else
                                  ('incomplete' if partial else 'missing'), rollouts=partial, error=status_record.get('error'))
                records = result['rollouts']
                actual_seeds = [r['evaluation_seed'] for r in records]
                if len(set(actual_seeds)) != len(actual_seeds) or any(s not in EVAL_SEEDS for s in actual_seeds):
                    raise ValueError('Invalid or duplicate evaluation seed.')
                if result['status'] == 'complete' and actual_seeds != EVAL_SEEDS:
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
                        reward=sum(row['rewards']), status=row['status'], task_status=result['status']))
                all_feasible = result['status'] == 'complete' and len(records) == len(EVAL_SEEDS) and all(r['best']['feasible'] for r in records)
                terminal_feasible = result['status'] == 'complete' and len(records) == len(EVAL_SEEDS) and all(r['terminal']['feasible'] for r in records)
                evaluation_cases.append(dict(policy=policy, circuit=name, training_seed=seed, status=result['status'],
                    completed=len(records), feasible=sum(r['best']['feasible'] for r in records),
                    best_luts_mean=float(np.mean([r['best']['luts'] for r in records])) if all_feasible else None,
                    terminal_luts_mean=float(np.mean([r['terminal']['luts'] for r in records])) if terminal_feasible else None,
                    terminal_feasible=sum(r['terminal']['feasible'] for r in records), error=result.get('error')))
    comparisons = [policy_comparison(evaluation_rows, name, baseline, cfg['protocol']['seeds'])
                   for name in cfg['protocol']['circuits'] for baseline in ('initial', 'uniform')]
    search_comparisons = []
    for name in cfg['protocol']['circuits']:
        for baseline in ('frozen', 'uniform'):
            reference = [r for r in runs if r['circuit'] == name and r['group'] == baseline]
            treatment = [r for r in runs if r['circuit'] == name and r['group'] == 'trained']
            valid = all(r['status'] == 'complete' and r['best'] and r['best']['feasible'] for r in reference + treatment)
            difference = [a['best']['luts'] - b['best']['luts'] for a, b in zip(reference, treatment)] if valid else []
            search_comparisons.append(dict(circuit=name, baseline=baseline, differences=difference,
                mean_luts_reduction=float(np.mean(difference)) if valid else None,
                percent=100 * float(np.mean(difference)) / float(np.mean([r['best']['luts'] for r in reference])) if valid else None,
                wins=sum(d > 0 for d in difference), ties=sum(d == 0 for d in difference), losses=sum(d < 0 for d in difference)))
    summary = dict(training=runs, evaluation=evaluation_cases, comparisons=comparisons, behavior=behavior,
                   search_comparisons=search_comparisons, baseline=read(root / 'baseline.json') if (root / 'baseline.json').exists() else {},
                   diagnostic_summary=[{k: r[k] for k in ('group', 'circuit', 'seed', 'episode', 'policy_kl',
                        'actor_drift_rms', 'critic_drift_rms', 'entropy')} for r in diagnostics if r['episode'] == 100],
                   training_complete=sum(r['status'] == 'complete' for r in runs),
                   evaluation_complete=sum(r['status'] == 'complete' for r in evaluation_cases))
    terminal_summary = []
    for name in cfg['protocol']['circuits']:
        for policy in POLICIES:
            records = [r for r in evaluation_rows if r['circuit'] == name and r['policy'] == policy]
            complete = len(records) == 90 and all(r['task_status'] == 'complete' for r in records)
            all_feasible = complete and all(r['terminal_feasible'] for r in records)
            terminal_summary.append(dict(circuit=name, policy=policy, completed=len(records),
                feasible=sum(r['terminal_feasible'] for r in records),
                luts_mean=float(np.mean([r['terminal_luts'] for r in records])) if all_feasible else None,
                levels_mean=float(np.mean([r['terminal_levels'] for r in records])) if complete else None,
                reward_mean=float(np.mean([r['reward'] for r in records])) if complete else None))
    summary['terminal_evaluation'] = terminal_summary
    summary['historical_reference'] = read(HERE / 'historical-reference.json') if (HERE / 'historical-reference.json').exists() else {}
    summary['execution'] = {}
    for phase, filename in [('training', 'experiment.json'), ('evaluation', 'evaluation.json')]:
        if (root / filename).exists():
            manifest = read(root / filename)
            item = {k: manifest.get(k) for k in ('started_at', 'finished_at', 'status')}
            item['minutes'] = ((datetime.fromisoformat(item['finished_at']) - datetime.fromisoformat(item['started_at'])).total_seconds() / 60
                               if item['started_at'] and item['finished_at'] else None)
            summary['execution'][phase] = item
    summary['netlist_validation'] = ({k: v for k, v in read(HERE / 'netlist-validation.json').items()
                                      if k != 'records'} if (HERE / 'netlist-validation.json').exists() else None)
    dump(HERE / 'summary.json', summary)
    dump(HERE / 'validation.json', dict(time=now(), training=validation, evaluation_rollouts=len(evaluation_rows),
                                      training_cec=training_cec, evaluation_cec=len(evaluation_rows),
                                      netlist_validation=summary['netlist_validation'], reward_and_best_replay=True))
    write_csv(HERE / 'training.csv', [dict(group=r['group'], circuit=r['circuit'], seed=r['seed'], status=r['status'],
        episodes=r['episodes_completed'], luts=(r['best'] or {}).get('luts'), levels=(r['best'] or {}).get('levels'),
        feasible=(r['best'] or {}).get('feasible'), training_seconds=r['training_seconds']) for r in runs])
    for filename, rows in [('trajectories.csv', curves), ('episodes.csv', episodes), ('diagnostics.csv', diagnostics),
                           ('steps.csv', steps), ('evaluation.csv', evaluation_rows), ('comparisons.csv', comparisons)]:
        write_csv(HERE / filename, rows)
    write_report(summary)
    archive_evidence(root)
    dump(HERE / 'analysis-provenance.json', dict(command=[sys.executable, *sys.argv], finished_at=now(),
        sources={p.name: sha(p) for p in (HERE / 'analysis.py', HERE / 'evaluate.py') if p.exists()},
        protocol_sha256=sha(HERE / 'protocol.yml') if (HERE / 'protocol.yml').exists() else None,
        evidence_sha256=sha(HERE / 'evidence.tar.gz'), bootstrap_repetitions=10000, bootstrap_seed=20260927,
        pairing='training rows resampled first; common evaluation columns resampled jointly',
        independent_training_seeds=cfg['protocol']['seeds'], evaluation_seeds=EVAL_SEEDS))
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
        *conclusion_lines(summary), '',
        '## 条件与比较口径', '',
        '使用原始 adf7bef 算法：int2float / i2c / max，深度限制 3 / 4 / 41，训练种子 0 / 1 / 2，'
        '100 轮 × 10 步，LUT6；原奖励、episode_welford 状态归一化、标准化回报、网络与 Adam 设置不变。'
        '初始映射可参与最优值。三组为训练、初始化权重冻结、均匀随机，各运行相同搜索预算。', '',
        '训练/冻结初始网络、Adam 和采样 RNG 相同，第一轮轨迹相同。冻结网络按状态输出概率，'
        '并不等于均匀随机。第 0 / 50 / 100 轮模型分别用新种子 10000–10029 进行 30 条无更新的 10 步评估；'
        '冻结三个快照一致，故共享初始化策略评估。评估只涉及训练使用的电路，不是跨电路泛化测试。', '',
        '各模型使用同一组评估随机数以便配对；均匀随机策略在三个训练种子行重复相同的 30 条序列，'
        '这些重复行不提供额外独立样本，统计重采样保留其相关性。', '',
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
    lines += ['', '| 电路 | 对照 | 逐种子 LUT 减少量（对照减训练） | 平均减少 LUT | 改善百分比 | 胜/平/负 |',
              '|---|---|---|---:|---:|---|']
    for row in summary['search_comparisons']:
        lines.append(f'| {row["circuit"]} | {row["baseline"]} | {row["differences"]} | '
                     f'{format_value(row["mean_luts_reduction"])} | {format_value(row["percent"])}% | '
                     f'{row["wins"]}/{row["ties"]}/{row["losses"]} |')
    lines += ['', 'resyn2 基线：' + '；'.join(f'{name}: {r["luts"]} LUT / {r["levels"]} 层' for name, r in summary['baseline'].items()) + '。', '',
              '完整搜索的可行结果为 ' + str(sum(r['status'] == 'complete' and bool((r['best'] or {}).get('feasible'))
                                             for r in summary['training'])) + '/27。训练表的胜/平/负按最佳可行 LUT 比较；'
              '每轮的可行结果和终点见 episodes.csv。', '',
              '这些分数是 1,000 个动作候选中找到的最佳解，不能单独证明策略质量改善。', '',
              '![按候选数量比较搜索轨迹](search-curves.svg)', '', '## 独立策略评估', '',
              '| 电路 | 训练种子 | 初始/冻结均值 | 50 轮均值 | 100 轮均值 | 均匀随机均值 |',
              '|---|---:|---:|---:|---:|---:|']
    for name in ('int2float', 'i2c', 'max'):
        for seed in (0, 1, 2):
            values = [next(r for r in summary['evaluation'] if r['circuit'] == name and
                           r['training_seed'] == seed and r['policy'] == policy) for policy in POLICIES]
            lines.append(f'| {name} | {seed} | ' + ' | '.join(f'{format_value(v["best_luts_mean"])} ({v["feasible"]}/30)' for v in values) + ' |')
    lines += ['', '每个均值来自固定模型的 30 条评估序列，各序列取包括初始映射在内的最佳可行解。'
              '括号为可行条数/30。存在不可行序列时，LUT 均值不汇总；优先比较可行率。'
              '底层逐条可行率、终点、奖励均在 evaluation.csv；未完成/不可行不被删除。', '',
              '![固定模型在新采样种子上的搜索质量](policy-evaluation.svg)', '',
              '| 电路 | 对照策略 | 100 轮模型平均减少 LUT | LUT 探索性 95% 区间 | 可行率变化（百分点）及区间 | 可行优先胜/平/负 | 逐种子 LUT / 百分比 |',
              '|---|---|---:|---|---|---|---|']
    for row in summary['comparisons']:
        interval = '未汇总' if row['ci95'] is None else f'[{row["ci95"][0]:.3f}, {row["ci95"][1]:.3f}]'
        effects = '; '.join(f'{p["seed"]}: {format_value(p["luts_reduction"])} / {format_value(p["percent"])}%' for p in row['per_seed'])
        feasible_interval = '未汇总' if row['feasibility_ci95'] is None else f'[{row["feasibility_ci95"][0]:.3f}, {row["feasibility_ci95"][1]:.3f}]'
        lines.append(f'| {row["circuit"]} | {row["baseline"]} | {format_value(row["mean_luts_reduction"])} | '
                     f'{interval} | {format_value(row["feasibility_gain_pp"])} / {feasible_interval} | '
                     f'{row["wins"]}/{row["ties"]}/{row["losses"]} | {effects} |')
    lines += ['', '正值表示训练更好。胜负沿用环境排序：可行解优先；可行解先比较 LUT，再比较层数；'
              '两条都不可行时先比较层数，再比较 LUT。它保留所有评估，不赋予不可行解任意 LUT 罚分。'
              '配对分层 bootstrap 先重采样 3 个训练种子，再重采样配对评估序列，'
              '所有训练种子共同重采样同一组评估种子列，以保留共同随机数及均匀随机基线重复使用的相关性。'
              '重复 10,000 次，分析种子 20260927。只有 3 个独立训练种子，区间为探索性估计；'
              '90 条评估序列不能当作 90 次独立训练。未预设实用收益门槛，不作统计等效判断。'
              '评估原则参照 [Agarwal 等](https://arxiv.org/abs/2108.13264)和 '
              '[Henderson 等](https://arxiv.org/abs/1709.06560)。', '',
              '| 电路 | 策略 | 终点可行 / 90 | 终点 LUT 均值 | 终点层数均值 | 原始奖励和均值 |',
              '|---|---|---:|---:|---:|---:|']
    for row in summary['terminal_evaluation']:
        lines.append(f'| {row["circuit"]} | {row["policy"]} | {row["feasible"]}/90 | '
                     f'{format_value(row["luts_mean"])} | {format_value(row["levels_mean"])} | {format_value(row["reward_mean"])} |')
    lines += ['', '终点层数和奖励在完整评估中不筛样本；终点 LUT 均值只在全部终点可行时汇总。', '',
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
              '单个电路每轮起点本来固定，首态归零本身不能证明故障。'
              '奖励与面积目标错位、状态信息损失、搜索最佳值掩盖策略变化是机制解释候选；本实验未修改这些设置来验证因果。', '',
              '## 证据、限制与复现', '',
              '测试覆盖原实现与诊断路径的精确权重/Adam/RNG/动作一致性、冻结、第一轮配对、三组真实工具断点恢复、'
              '非法续跑拒绝和受控正优势更新。完整实验按逐步日志重新计算奖励、回报尺度、最优值和终点；'
              '训练及评估导出的最佳映射通过 ABC CEC。测试和验证记录见 tests.log、test-results.json、validation.json。', '',
              ('最终逐一复核 ' + str(summary['netlist_validation']['exports_checked']) + ' 份导出结果，合计 ' +
               str(summary['netlist_validation']['netlists_verified']) + ' 个映射及未映射网表通过 CEC；完整命令、'
               '网表哈希与验证输出哈希见 netlist-validation.json。' if summary['netlist_validation'] and
               summary['netlist_validation']['status'] == 'complete' else '映射与未映射网表的最终复核记录尚未生成。'), '',
              '完整实验每个训练种子只运行一次，未另做全部 27 次同种子重复；可复现性证据来自短程精确回归与断点恢复，'
              '不能把“完整日志复核”表述为“全部实验独立重复成功”。耗时包含额外诊断开销且并发共享资源，'
              '本报告依据候选预算比较搜索质量，不根据耗时给算法效率排名。', '',
              '本次正式执行耗时：' + '；'.join(f'{phase} {format_value(record["minutes"])} 分钟'
                  for phase, record in summary['execution'].items()) + '。使用 3 个 workers、每进程 1 个 Torch 线程，'
              '每分钟记录进度，单任务硬超时 30 分钟；起止时间、逐任务命令与退出状态见 provenance.json 和 evaluation-provenance.json。', '',
              '核查了 11 类统计解释风险：辛普森悖论（逐电路报告）、生态谬误（训练种子是独立单位）、'
              '选择偏差与碰撞变量（不按结果筛样本）、基率忽略（完整可行率）、回归均值（固定种子与对照）、'
              '幸存者偏差（保留失败）、多重搜索与分析分叉（预定六个探索性对照，不据此宣称显著）、'
              '相关当因果与反向因果（仅对受控冻结干预和已测范围解释）。覆盖 11/11，少训练种子限制仍然存在。', '',
              '## 如何判断学习有效及下一步', '',
              '第一层检查 Actor/Critic 的梯度、参数和 Adam 状态，配合严格不变的冻结对照，确认更新发生。'
              '第二层在固定归一化输入上比较动作概率，确认策略改变。第三层使用未参与更新的新采样种子，'
              '检验固定模型在相同约束和候选预算下是否优于初始化网络与均匀随机，并呈现全部训练种子及不确定性。'
              '前两层通过、第三层没有一致收益时，应表述为“发生了学习更新，但当前设置下未验证出稳定搜索收益”。', '',
              '若继续确认收益，应增加独立训练种子并使用新的预先固定评估种子。若尝试改进，分别对奖励与面积目标的对齐、'
              '状态归一化做单因素对照，保持搜索预算；不要同时改多项再将提升归因于某一机制。'
              '第 50 轮仅为预定诊断点，不根据结果事后将更好的中间模型替换第 100 轮主对照。', '',
              '源码/配置/工具/电路指纹和完整命令见 provenance.json、evaluation-provenance.json；'
              '统计生成代码和配对方法见 analysis-provenance.json；'
              '压缩证据 evidence.tar.gz 内有 SHA256.json。完整权重和原始输出保存在 results/learning-effectiveness/。'
              '模型与探针未纳入 Git；model-manifest.json 记录本机权重哈希。其他机器需要重新训练或复制这些权重，'
              '仅凭压缩日志不能重放固定模型评估。historical-reference.json 记录历史文件哈希复核，'
              '历史 results/ 其他实验未改写。AI 用于实现实验、核对日志、统计与报告撰写。', '',
              '```bash', '.tools/conda-env/bin/python -B -m unittest discover -s tests -v',
              '# 保存完整测试输出及命令/源码哈希',
              '.tools/conda-env/bin/python -B experiments/learning-effectiveness/check.py',
              '.tools/conda-env/bin/python -B -u experiments/learning-effectiveness/run.py',
              '# 中断后，仅相同源码/配置/工具允许续跑',
              '.tools/conda-env/bin/python -B -u experiments/learning-effectiveness/run.py --resume',
              '.tools/conda-env/bin/python -B -u experiments/learning-effectiveness/evaluate.py',
              '.tools/conda-env/bin/python -B experiments/learning-effectiveness/verify.py',
              '# 已有完整评估时，重读证据生成报告',
              '.tools/conda-env/bin/python -B experiments/learning-effectiveness/evaluate.py --report-only', '```', '']
    lines += ['在仓库根目录执行。首次完整重跑需使用新的干净结果目录；已有相同指纹结果使用 --resume，'
              '已有完整输出重新分析使用 --report-only。原始输出路径由 protocol.yml 指定，默认命令不会覆盖历史实验。', '']
    lines += ['图表由 plot.py 使用带 ReportLab 与 PyMuPDF 的 Python 运行环境生成；plot-validation.json '
              '保存本次实际命令和版本。随后运行 package.py 核对交付哈希和历史文件保留情况，'
              'artifact-manifest.json 汇总所有交付物的哈希。', '']
    (HERE / 'report.md').write_text('\n'.join(lines))


def conclusion_lines(summary):
    trained = [r for r in summary['diagnostic_summary'] if r['group'] == 'trained']
    fixed = [r for r in summary['diagnostic_summary'] if r['group'] != 'trained']
    changed = sum(r['actor_drift_rms'] > 0 and r['critic_drift_rms'] > 0 for r in trained)
    policies_changed = sum(r['policy_kl'] > 0 for r in trained)
    unchanged = sum(r['actor_drift_rms'] == 0 and r['critic_drift_rms'] == 0 for r in fixed)
    lines = [f'更新检查：训练组 {changed}/9 次 Actor 和 Critic 均改变，{policies_changed}/9 次固定探针动作分布改变；'
             f'冻结和均匀随机组 {unchanged}/18 次参数保持不变。只有完整运行后的诊断才计入此处。']
    for row in summary['search_comparisons']:
        if row['baseline'] == 'frozen' and row['mean_luts_reduction'] is not None:
            lines.append(f'{row["circuit"]} 的 1,000 候选搜索中，训练相对冻结的平均 LUT 减少量为 '
                         f'{row["mean_luts_reduction"]:.3f}（{row["percent"]:.3f}%），逐种子差值为 {row["differences"]}。')
    historical = summary['historical_reference'].get('historical_i2c')
    if historical:
        matched = all([r['best']['luts'] for r in summary['training'] if r['circuit'] == 'i2c' and
                       r['group'] == group and r['status'] == 'complete'] == historical[key]
                      for group, key in [('trained', 'trained_10'), ('frozen', 'frozen_10')])
        if matched:
            lines.append('i2c 的训练和冻结逐种子最佳值均与历史 10 步实验完全相同，复现了“冻结与训练结果差异很小”的观察。'
                         '历史 30 步结果只作背景，本次没有重跑，也不纳入新评估的统计。')
    if summary['evaluation_complete'] != 36:
        return lines + ['独立评估尚未全部完成，暂不对学习收益作最终判断。']
    for row in summary['comparisons']:
        if row['baseline'] == 'initial':
            difference = row['mean_luts_reduction']
            if difference is None:
                lines.append(f'{row["circuit"]}：100 轮策略可行 {row["trained_feasible"]}/90，初始化/冻结 '
                             f'{row["baseline_feasible"]}/90；可行率变化 {format_value(row["feasibility_gain_pp"])} 个百分点。'
                             '存在不可行序列，未汇总 LUT 收益。')
            elif difference > 0:
                lines.append(f'{row["circuit"]}：100 轮策略在新种子评估中平均比初始化/冻结策略少 {difference:.3f} LUT。')
            elif difference < 0:
                lines.append(f'{row["circuit"]}：100 轮策略在新种子评估中平均比初始化/冻结策略多 {-difference:.3f} LUT。')
            else:
                lines.append(f'{row["circuit"]}：100 轮策略与初始化/冻结策略的新种子评估均值相同。')
    consistent = all(r['complete'] and (r['feasibility_gain_pp'] > 0 or
                     (r['feasibility_gain_pp'] == 0 and r['mean_luts_reduction'] is not None and r['mean_luts_reduction'] > 0))
                     for r in summary['comparisons'])
    lines.append('本次对所有电路和两种基线观察到方向一致的平均改善，但只有 3 个训练种子，尚不能推广为普遍有效。'
                 if consistent else '本次没有观察到训练策略对三个电路、两种基线都一致的搜索收益。'
                 '这与“参数更新正常”并不矛盾，也不证明强化学习在其他预算或设置下无效。')
    return lines


def archive_evidence(root):
    names = {'log.csv', 'result.json', 'rollout.json', 'best.v', 'best-mapped.v', 'best.json', 'equivalence.log', 'full-equivalence.log',
             'parameter-validation.json', 'status.json', 'steps.jsonl', 'diagnostics.jsonl', 'experiment.json', 'evaluation.json', 'baseline.json'}
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
