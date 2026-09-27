"""Replay budgets/rewards/states, verify netlists, report and package evidence."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import numpy as np
import torch

from study import (HERE, ROOT, GROUPS, NEW_GROUPS, SEEDS, EVAL_SEEDS, Normalizer, FPGASession,
                   sha, dump, load, state_hash, now, guard, group_config, read_rows, write_csv,
                   assert_history_unchanged)
from study import history

LABELS = {'old-trained':'旧训练', 'old-frozen':'旧冻结', 'fixed-trained':'新训练',
          'fixed-frozen':'新冻结', 'uniform':'均匀随机'}


def log_replay(cfg, folder, episodes, steps=None):
    best, curve, rewards_by_episode = None, [], []
    protocol = cfg['protocol']
    for episode in range(1, episodes + 1):
        rows = list(csv.DictReader((folder / f'episodes/{episode}/log.csv').open()))
        if len(rows) != protocol['iterations'] + 1 or [int(r['iteration']) for r in rows] != list(range(protocol['iterations'] + 1)):
            raise ValueError('Missing, duplicated or reordered candidate log: ' + str(folder))
        sequence, rewards = list(protocol['initial_sequence']), []
        previous = None
        for row in rows:
            iteration = int(row['iteration'])
            luts, levels, reward = int(row['luts']), int(row['levels']), int(row['reward'])
            if iteration:
                sequence.append(row['optimization'])
                area = int(luts < previous[0]) - int(luts > previous[0])
                depth = int(levels < previous[1]) - int(levels > previous[1])
                expected = (cfg['method']['reward']['feasible'][area] if levels <= 4 else
                            cfg['method']['reward']['infeasible'][depth][area])
                if expected != reward:
                    raise ValueError('Reward replay mismatch.')
                rewards.append(reward)
                if steps is not None:
                    step = steps[(episode - 1) * protocol['iterations'] + iteration - 1]
                    if (step['episode'], step['iteration'], step['luts'], step['levels'], step['reward']) != (episode, iteration, luts, levels, reward):
                        raise ValueError('Step diagnostics disagree with candidate log.')
                    if protocol['actions'][step['action']] != row['optimization']:
                        raise ValueError('Action identity mismatch.')
            elif reward != 0 or row['optimization'] != protocol['initial_sequence'][-1]:
                raise ValueError('Initial log is invalid.')
            previous = (luts, levels)
            candidate = dict(luts=luts, levels=levels, feasible=levels <= 4, sequence=sequence.copy(),
                             episode=episode, iteration=iteration)
            if (iteration or protocol['evaluation']['include_initial']) and (best is None or FPGASession.rank(candidate) < FPGASession.rank(best)):
                best = candidate
            if iteration or episode == 1:
                curve.append(dict(candidate=(episode - 1) * protocol['iterations'] + iteration,
                                  luts=best['luts'], levels=best['levels'], feasible=best['feasible']))
        rewards_by_episode.append(sum(rewards))
    return best, curve, rewards_by_episode


def verify_export(task):
    cfg, folder, expected, raw = task
    files = [folder / 'best-mapped.v', folder / 'best.v'] if raw else [folder / 'mapped.v']
    hashes = {p.name:sha(p) for p in files}
    command = f'read "{files[0]}"; print_stats; ' + ' '.join(
        f'cec "{cfg["protocol"]["circuits"]["i2c"]["file"]}" "{p}";' for p in files)
    argv = [cfg['runtime']['abc_binary'], '-c', command]
    record = dict(folder=str(folder.relative_to(Path(cfg['runtime']['output_dir']))),
                  command=argv, expected=expected, netlist_sha256=hashes, started_at=now())
    try:
        process = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        output = process.stdout
        (folder / 'full-equivalence.log').write_text(output)
        found = re.findall(r'\bnd\s*=\s*(\d+)[^\n]*?\blev\s*=\s*(\d+)', output)
        metrics = tuple(map(int, found[0])) if found else None
        equivalents = output.count('Networks are equivalent')
        if process.returncode != 0 or metrics != (expected['luts'], expected['levels']) or equivalents != len(files):
            raise RuntimeError('Exported metrics or combinational equivalence failed.')
        if hashes != {p.name:sha(p) for p in files}:
            raise RuntimeError('Netlist changed during verification.')
        record.update(status='complete', netlists_verified=equivalents,
                      log_sha256=sha(folder / 'full-equivalence.log'))
    except Exception as error:
        record.update(status='failed', netlists_verified=0, error=repr(error))
    record['finished_at'] = now()
    return record


def verify(cfg):
    root, manifest = guard(cfg)
    checks, exports, initial_hashes = [], [], {}
    fixed_references = set()
    for group in GROUPS:
        local = group_config(cfg, group)
        for seed in SEEDS:
            folder = root / 'training' / group / 'i2c' / f'seed-{seed}'
            result = json.loads((folder / 'result.json').read_text())
            if result['status'] != 'complete' or result['episodes_completed'] != 100:
                raise ValueError('Cannot verify incomplete search.')
            steps, diagnostics = read_rows(folder / 'steps.jsonl'), read_rows(folder / 'diagnostics.jsonl')
            if len(steps) != 1000 or len(diagnostics) != 100:
                raise ValueError('Candidate or update budget mismatch.')
            best, _, rewards = log_replay(local, folder, 100, steps)
            initial, final = load(folder / 'snapshots/0.pt'), load(folder / 'checkpoint.pt')
            if best != result['best'] or best != final['best'] or rewards != result['rewards'] or rewards != final['rewards']:
                raise ValueError('Best mapping or episode rewards mismatch.')
            initial_hashes[(group, seed)] = state_hash(initial['network'])
            actor_changed = any(not torch.equal(v, final['network'][k]) for k,v in initial['network'].items() if k.startswith('actor.'))
            critic_changed = any(not torch.equal(v, final['network'][k]) for k,v in initial['network'].items() if k.startswith('critic.'))
            learning = group.endswith('-trained')
            if learning and not (actor_changed and critic_changed):
                raise ValueError('Actor or Critic did not change.')
            if not learning and (actor_changed or critic_changed or state_hash(initial['optimizer']) != state_hash(final['optimizer'])):
                raise ValueError('Frozen network or Adam changed.')
            for episode in range(100):
                segment = steps[episode * 10:(episode + 1) * 10]
                reference = np.asarray(segment[0]['raw_state'], dtype=np.float32)
                normalizer = Normalizer(9, local['method']['normalization'], local['method']['features'], reference)
                if group.startswith('fixed-'):
                    fixed_references.add(tuple(reference.tolist()))
                for step in segment:
                    raw = np.asarray(step['raw_state'], dtype=np.float32)
                    np.testing.assert_array_equal(normalizer.normalize(raw), np.asarray(step['normalized_state'], dtype=np.float32))
                    if group.startswith('fixed-') and step['reference_state'] != reference.tolist():
                        raise ValueError('Fixed reference drifted within an episode.')
                _, processed = history.returns_for([s['reward'] for s in segment], local['method'])
                if not np.isclose(float(processed.std()), diagnostics[episode]['processed_returns_std']):
                    raise ValueError('Return normalization changed.')
            checks.append(dict(group=group, seed=seed, candidates=len(steps), actor_changed=actor_changed,
                               critic_changed=critic_changed, frozen_unchanged=not learning,
                               initial_sha256=sha(folder / 'snapshots/0.pt'), final_sha256=sha(folder / 'checkpoint.pt')))
            exports.append((cfg, folder, best, True))
    if len(fixed_references) != 1:
        raise ValueError('Initial strash reference differs across fixed-mode searches.')
    for seed in SEEDS:
        if len({initial_hashes[(g,seed)] for g in GROUPS}) != 1:
            raise ValueError('Initial weights are not paired.')
        for prefix in ('old', 'fixed'):
            a = root / f'training/{prefix}-trained/i2c/seed-{seed}/episodes/1/log.csv'
            b = root / f'training/{prefix}-frozen/i2c/seed-{seed}/episodes/1/log.csv'
            if a.read_bytes() != b.read_bytes():
                raise ValueError('Training/frozen first episodes diverged.')
    evaluation_checks = []
    for group in GROUPS:
        local = group_config(cfg, group)
        for seed in ((0,) if group == 'uniform' else SEEDS):
            folder = root / 'evaluation' / group / f'seed-{seed}'
            result = json.loads((folder / 'result.json').read_text())
            if result['status'] != 'complete' or not result['network_unchanged'] or not result['checkpoint_unchanged']:
                raise ValueError('Incomplete or mutating evaluation.')
            checkpoint = root / f'training/{group}/i2c/seed-{seed}/checkpoint.pt'
            if sha(checkpoint) != result['checkpoint_sha256'] or [r['evaluation_seed'] for r in result['rollouts']] != list(EVAL_SEEDS):
                raise ValueError('Evaluation model or seed set mismatch.')
            for row in result['rollouts']:
                dest = folder / f'rollout-{row["evaluation_seed"]}'
                best, _, rewards = log_replay(local, dest, 1)
                if best != row['best'] or rewards != [sum(row['rewards'])] or len(row['actions']) != 10:
                    raise ValueError('Evaluation score or budget mismatch.')
                csv_rows = list(csv.DictReader((dest / 'episodes/1/log.csv').open()))
                if row['rewards'] != [int(r['reward']) for r in csv_rows[1:]] or [local['protocol']['actions'][a] for a in row['actions']] != [r['optimization'] for r in csv_rows[1:]]:
                    raise ValueError('Evaluation action/reward mismatch.')
                reference = np.asarray(row['reference_state'], dtype=np.float32)
                normalizer = Normalizer(9, local['method']['normalization'], local['method']['features'], reference)
                if group.startswith('fixed-') and tuple(reference.tolist()) not in fixed_references:
                    raise ValueError('Inference uses a different initial reference.')
                for step in row['steps']:
                    np.testing.assert_array_equal(normalizer.normalize(np.asarray(step['raw_state'], dtype=np.float32)),
                                                  np.asarray(step['normalized_state'], dtype=np.float32))
                terminal = csv_rows[-1]
                if (row['terminal']['luts'], row['terminal']['levels']) != (int(terminal['luts']), int(terminal['levels'])):
                    raise ValueError('Terminal metrics mismatch.')
                exports.append((cfg, dest, best, True))
            evaluation_checks.append(dict(group=group, seed=seed, rollouts=30, checkpoint_sha256=sha(checkpoint)))
    baseline = json.loads((root / 'baseline.json').read_text())['i2c']
    exports.append((cfg, root / 'baseline/i2c', baseline, False))
    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for record in pool.map(verify_export, exports):
            records.append(record)
            if len(records) % 100 == 0:
                print(f'CEC: {len(records)}/{len(exports)} exports', flush=True)
    successful = all(r['status'] == 'complete' for r in records)
    dump(HERE / 'validation.json', dict(status='complete' if successful else 'failed',
        training_runs=15, training_candidates=15000, evaluation_rollouts=390, evaluation_candidates=3900,
        paired_initial_weights=True, paired_first_episodes=True, state_and_reward_replay=True,
        fixed_reference=list(next(iter(fixed_references))), checks=checks, evaluation_checks=evaluation_checks))
    dump(HERE / 'netlist-validation.json', dict(status='complete' if successful else 'failed',
        exports_checked=len(records), netlists_verified=sum(r['netlists_verified'] for r in records), records=records))
    print(f'Validated {len(records)} exports / {sum(r["netlists_verified"] for r in records)} netlists.', flush=True)
    if not successful:
        raise RuntimeError('CEC failed; all failures retained.')


def reductions(matrix):
    engineering = matrix['old-trained'] - matrix['fixed-trained']
    old_learning = matrix['old-frozen'] - matrix['old-trained']
    fixed_learning = matrix['fixed-frozen'] - matrix['fixed-trained']
    result = {}
    for name, values in [('engineering',engineering), ('old_learning',old_learning),
                         ('fixed_learning',fixed_learning), ('learning_change',fixed_learning-old_learning)]:
        result[name] = dict(by_seed=values.tolist(), mean=float(values.mean()), std=float(values.std(ddof=0)),
                            wins=int((values>0).sum()), ties=int((values==0).sum()), losses=int((values<0).sum()))
    return result


def analyze(cfg):
    root, manifest = guard(cfg)
    training, trajectories, evaluation, eval_means = [], [], [], []
    matrices = {}
    for group in GROUPS:
        local = group_config(cfg, group)
        values = []
        for seed in SEEDS:
            folder = root / f'training/{group}/i2c/seed-{seed}'
            record = folder / 'result.json'
            result = json.loads(record.read_text()) if record.exists() else {'status':'failed' if (folder/'failure.json').exists() else 'missing', 'best':None}
            best = result.get('best') or {}
            row = dict(group=group, seed=seed, status=result['status'], reused=group not in NEW_GROUPS,
                       episodes=result.get('episodes_completed',0), luts=best.get('luts'), levels=best.get('levels'),
                       feasible=best.get('feasible',False), training_seconds=result.get('training_seconds'))
            training.append(row)
            if result['status'] == 'complete':
                _, curve, _ = log_replay(local, folder, 100)
                trajectories.extend(dict(group=group,seed=seed,**r) for r in curve)
                values.append(best['luts'] if best['feasible'] else np.nan)
            else:
                values.append(np.nan)
        matrices[group] = np.asarray(values,dtype=float)
        for seed in ((0,) if group == 'uniform' else SEEDS):
            result_path = root / f'evaluation/{group}/seed-{seed}/result.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else {'status':'failed' if (result_path.parent/'failure.json').exists() else 'missing','rollouts':[]}
            rows = result['rollouts']
            for row in rows:
                evaluation.append(dict(group=group,training_seed=seed,evaluation_seed=row['evaluation_seed'],
                    status=row['status'],best_luts=row['best']['luts'],best_levels=row['best']['levels'],
                    best_feasible=row['best']['feasible'],terminal_luts=row['terminal']['luts'],
                    terminal_levels=row['terminal']['levels'],terminal_feasible=row['terminal']['feasible'],
                    reward_sum=sum(row['rewards'])))
            all_feasible = len(rows)==30 and all(r['best']['feasible'] for r in rows)
            terminal_feasible = len(rows)==30 and all(r['terminal']['feasible'] for r in rows)
            eval_means.append(dict(group=group,seed=seed,status=result['status'],rollouts=len(rows),
                feasible=sum(r['best']['feasible'] for r in rows),
                mean_luts=float(np.mean([r['best']['luts'] for r in rows])) if all_feasible else None,
                terminal_feasible=sum(r['terminal']['feasible'] for r in rows),
                terminal_mean_luts=float(np.mean([r['terminal']['luts'] for r in rows])) if terminal_feasible else None,
                terminal_mean_levels=float(np.mean([r['terminal']['levels'] for r in rows])) if len(rows)==30 else None))
    complete = all(r['status']=='complete' and r['feasible'] for r in training) and len(evaluation)==390 and all(r['status']=='complete' and r['best_feasible'] for r in evaluation)
    search = reductions(matrices) if complete else None
    ematrix = {g:np.asarray([r['mean_luts'] for r in eval_means if r['group']==g],dtype=float) for g in GROUPS if g!='uniform'}
    assessment = reductions(ematrix) if complete else None
    threshold_pass = bool(complete and search['engineering']['mean']>=1 and search['engineering']['wins']>=2)
    summary = dict(status='complete' if complete else 'incomplete', training=training, evaluation_means=eval_means,
        search_comparisons=search, evaluation_comparisons=assessment, screening_passed=threshold_pass,
        training_candidates_new=6000, training_candidates_reused=9000, evaluation_candidates=3900,
        independent_training_seeds=3, uniform_evaluation_unique_rollouts=30)
    if complete:
        summary['frozen_representation_gain'] = dict(
            search_mean=float((matrices['old-frozen']-matrices['fixed-frozen']).mean()),
            evaluation_mean=float((ematrix['old-frozen']-ematrix['fixed-frozen']).mean()))
    dump(HERE/'summary.json',summary)
    write_csv(HERE/'training.csv',training)
    write_csv(HERE/'evaluation-means.csv',eval_means)
    if trajectories: write_csv(HERE/'search-curves.csv',trajectories)
    if evaluation: write_csv(HERE/'evaluation.csv',evaluation)
    lines=['# 固定尺度状态归一化：i2c 消融实验报告','', '## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: run + validate',
        '- Origin Date: 2026-09-28',
        '- Verification Status: '+('VERIFIED（日志、预算、状态与导出网表核验；短程恢复回归）' if (HERE/'validation.json').exists() and json.loads((HERE/'validation.json').read_text())['status']=='complete' else 'UNVERIFIED'),
        '- Version Label: fixed_state_normalization_v1','', '## 主要结果','']
    if complete:
        e=search['engineering']; ol=search['old_learning']; nl=search['fixed_learning']; dl=search['learning_change']
        old=float(matrices['old-trained'].mean()); new=float(matrices['fixed-trained'].mean())
        lines += [f'固定尺度归一化下，新训练的最佳可行 LUT 均值为 **{new:.3f}**，旧训练为 **{old:.3f}**。旧训练减新训练的平均差为 **{e["mean"]:.3f} LUT**，逐种子差值为 {e["by_seed"]}，胜/平/负为 {e["wins"]}/{e["ties"]}/{e["losses"]}。', '',
            ('达到预定筛查标准：平均至少减少 1 LUT，且至少两个种子改善，值得使用新种子进一步验证。' if threshold_pass else '未达到预定筛查标准：平均至少减少 1 LUT，且至少两个种子改善。本次没有确认值得继续投入的工程收益。'), '',
            f'搜索中的学习收益（冻结减训练）：旧归一化 **{ol["mean"]:.3f} LUT**，固定归一化 **{nl["mean"]:.3f} LUT**；学习收益变化为 **{dl["mean"]:.3f} LUT**，逐种子为 {dl["by_seed"]}。输入表示效果与权重学习效果必须分别解释。', '',
            f'独立评估中，旧训练减新训练的平均差为 **{assessment["engineering"]["mean"]:.3f} LUT**；旧、新归一化的学习收益分别为 **{assessment["old_learning"]["mean"]:.3f}** 和 **{assessment["fixed_learning"]["mean"]:.3f} LUT**，学习收益变化为 **{assessment["learning_change"]["mean"]:.3f} LUT**。', '',
            '三个种子仅支持探索性判断；固定模型评估的多条轨迹不能替代独立训练种子。']
        frozen_gain=summary['frozen_representation_gain']['search_mean']
        if e['mean']>0 and frozen_gain>0:
            lines+=['',f'新训练与新冻结同步改善，新冻结相对旧冻结平均少 {frozen_gain:.3f} LUT。冻结策略没有学习更新，因此输入表示本身改变了搜索分布；不能把新训练的全部改善归功于学习。']
        elif frozen_gain>0:
            lines+=['',f'新冻结相对旧冻结平均少 {frozen_gain:.3f} LUT，但新训练未同步改善。表示变化对冻结搜索的帮助没有转化为本次训练搜索的工程收益。']
        elif e['mean']>0:
            lines+=['','新冻结未同步改善；新训练的收益需结合学习收益变化及独立评估判断，不能仅由单个最好网表归因。']
    else:
        lines+=['实验存在未完成或不可行记录，不计算完整实验的收益结论。缺失及失败记录保留在结果目录。']
    lines += ['', '## 设置与计分口径','',
        '仅改变状态归一化。i2c、种子 0/1/2、100 轮 × 10 步、LUT6、层数上限 4、初始 strash、七个动作、九维输入。网络、Adam、γ=0.99、原奖励及逐轮回报标准化保持不变。', '',
        '六个计数特征按 x/max(|x₀|,1) 换算，三个门比例保留；x₀ 来自每轮初始 strash 后的原始状态，参考分母不随轨迹变化。训练、诊断与推理使用相同实现。', '',
        '原版训练、冻结、均匀随机的九次搜索复用 learning-effectiveness 的完整证据。其源码按基准提交核验，工具/电路/环境一致，45 个 i2c 模型相关文件哈希匹配；复用来源见 experiment.json、historical-inputs.json。旧结果未重跑，也不称为本次独立重复。', '',
        '新增六次搜索共 6,000 候选，历史对照共 9,000 候选；初始映射可选优但不计入动作候选。每种归一化的训练/冻结使用同初始权重和采样 RNG，第一轮一致；两种归一化之间允许动作不同。', '',
        '固定第 100 轮模型，用 30000–30029 评估种子，各跑 30 条无更新十步轨迹，取每条轨迹含起点的最佳可行 LUT。四类网络策略各三个模型；均匀随机仅跑一组并共享。390 条实际轨迹、3,900 候选，与训练预算分开；评估不会把偶然更优网表追加到训练成绩。', '',
        '## 训练搜索结果','', '| 组别 | seed 0 | seed 1 | seed 2 | 均值 ± 总体标准差 | 来源 |', '|---|---:|---:|---:|---:|---|']
    for group in GROUPS:
        values=matrices[group]
        avg=f'{values.mean():.3f} ± {values.std(ddof=0):.3f}' if np.isfinite(values).all() else '未汇总'
        lines.append('| '+LABELS[group]+' | '+' | '.join(str(int(v)) if np.isfinite(v) else '缺失/不可行' for v in values)+' | '+avg+' | '+('新增' if group in NEW_GROUPS else '历史复用')+' |')
    if complete:
        lines+=['', '| 比较（正值表示改善） | 逐种子差值 | 平均差 | 胜/平/负 |', '|---|---|---:|---|']
        for label,key in [('旧训练减新训练','engineering'),('旧冻结减旧训练','old_learning'),('新冻结减新训练','fixed_learning'),('新学习收益减旧学习收益','learning_change')]:
            r=search[key];lines.append(f'| {label} | {r["by_seed"]} | {r["mean"]:.3f} | {r["wins"]}/{r["ties"]}/{r["losses"]} |')
        for group in ('old-trained','fixed-trained'):
            d=matrices['uniform']-matrices[group]
            lines.append(f'| 均匀随机减{LABELS[group]} | {d.tolist()} | {d.mean():.3f} | {int((d>0).sum())}/{int((d==0).sum())}/{int((d<0).sum())} |')
    lines += ['', '![搜索曲线](search-curves.svg)', '', '## 固定模型独立评估','',
        '| 组别 | 训练种子 | 最佳可行 LUT 均值 | 可行轨迹 | 终点 LUT 均值 | 终点层数均值 |', '|---|---:|---:|---|---:|---:|']
    for row in eval_means:
        fmt=lambda x:'未汇总' if x is None else f'{x:.3f}'
        lines.append(f'| {LABELS[row["group"]]} | {row["seed"] if row["group"]!="uniform" else "共享"} | {fmt(row["mean_luts"])} | {row["feasible"]}/{row["rollouts"]} | {fmt(row["terminal_mean_luts"])} | {fmt(row["terminal_mean_levels"])} |')
    if assessment:
        lines+=['', '| 比较 | 三个模型的配对差值 | 平均差 |', '|---|---|---:|']
        for label,key in [('旧训练减新训练','engineering'),('旧冻结减旧训练','old_learning'),('新冻结减新训练','fixed_learning'),('学习收益变化','learning_change')]:
            r=assessment[key]; lines.append(f'| {label} | {[round(v,3) for v in r["by_seed"]]} | {r["mean"]:.3f} |')
    lines += ['', '## 验证、耗时与限制','',
        'validation.json 逐步复算全部搜索和评估的奖励、归一化、最好值、动作与预算；核对同种子初始权重、首轮配对、冻结不变和 Actor/Critic 更新。netlist-validation.json 记录映射及未映射最佳网表的 CEC、指标及哈希。', '',
        '测试覆盖原版路径的精确权重/Adam/RNG/动作回归、固定尺度的公式与特征顺序、零初始计数、路径无关性、生产/诊断一致性、真实工具断点恢复、非法续跑拒绝，以及推理不修改模型。短程恢复回归不等于全部长程搜索独立重跑。', '',
        '| 组别 | 三次搜索耗时总和（秒） | 来源 |', '|---|---:|---|']
    for group in GROUPS:
        times=[r['training_seconds'] for r in training if r['group']==group]
        total=f'{sum(times):.3f}' if all(t is not None for t in times) else '未汇总'
        lines.append(f'| {LABELS[group]} | {total} | '+('本次运行' if group in NEW_GROUPS else '历史运行')+' |')
    for phase in ('train','evaluate'):
        path=root/(phase+'-provenance.json')
        if path.exists():
            processes=json.loads(path.read_text())['processes']
            lines+=['',f'{phase}：{len(processes)} 个实际任务，累计子任务墙钟时间 {sum(p["elapsed_seconds"] for p in processes):.3f} 秒；每个任务的命令、起止时间、退出码及超时记录见 {phase}-provenance.json。']
    lines+=['', '三个并行进程，每进程一个 Torch 线程，单任务硬超时 30 分钟，失败不自动重复。各组并发共享机器，旧新运行时间也不同；耗时用于成本记录，不据此声称算法效率提高。', '',
        '预定门槛为平均至少少 1 LUT 且至少两种子改善，未调参、未换种子、未选择中间模型。本实验止于 i2c 三种子；不做显著性宣称、不外推到其他电路。评估种子使用共同随机数，均匀随机共享结果不计为多个独立样本。', '',
        '统计风险核查覆盖 11/11：辛普森悖论（不跨电路混合）、生态谬误（训练种子为独立单位）、选择偏差和碰撞变量（不筛成功结果）、基率忽略（报告全部可行率）、回归均值（固定种子与对照）、幸存者偏差（保留失败）、多重搜索和分析分叉（固定公式/门槛/最终模型，不作显著性选择）、相关当因果和反向因果（分开工程收益与训练减冻结收益）。', '',
        '## 交付与重放','',
        '代码、协议、CSV、汇总、图表、测试及验证记录随分支交付；evidence.tar.gz 含日志、初始/最终权重、最佳网表及 SHA256.json。完整本机输出位于 results/fixed-state-normalization/；历史输入保持哈希不变。artifact-manifest.json 记录交付文件哈希和源码指纹。AI 用于实现、核验、分析与报告撰写。', '',
        '在仓库根目录运行（已有完整结果不可覆盖）：','', '```bash',
        '.tools/conda-env/bin/python -B experiments/fixed-state-normalization/check.py',
        '.tools/conda-env/bin/python -B -u experiments/fixed-state-normalization/run.py all',
        '# 仅在中断后显式续跑，源码/协议/工具必须保持一致',
        '.tools/conda-env/bin/python -B -u experiments/fixed-state-normalization/run.py train --resume',
        '.tools/conda-env/bin/python -B -u experiments/fixed-state-normalization/run.py evaluate --resume',
        '# 分析已有数据，不重复搜索',
        '.tools/conda-env/bin/python -B experiments/fixed-state-normalization/run.py analyze',
        '```', '']
    (HERE/'report.md').write_text('\n'.join(lines))
    dump(HERE/'provenance.json', dict(**manifest, training=json.loads((root/'train-provenance.json').read_text()) if (root/'train-provenance.json').exists() else None,
                                      evaluation=json.loads((root/'evaluate-provenance.json').read_text()) if (root/'evaluate-provenance.json').exists() else None))
    print(json.dumps({'status':summary['status'],'search':search,'evaluation':assessment,'screening_passed':threshold_pass},ensure_ascii=False),flush=True)


def package(cfg):
    root, manifest = guard(cfg)
    tests=json.loads((HERE/'test-results.json').read_text())
    if not tests['successful'] or tests['skipped'] or tests['source_hashes'] != manifest['sources']:
        raise ValueError('Tested source differs from executed source, or tests did not pass.')
    if sha(HERE/'tests.log') != tests['log_sha256']:
        raise ValueError('Test log changed.')
    for name in ('validation.json','netlist-validation.json','summary.json'):
        if json.loads((HERE/name).read_text())['status'] != 'complete':
            raise ValueError('Cannot package a complete-result claim for incomplete validation.')
    for phase in ('train','evaluate'):
        if json.loads((root/(phase+'-provenance.json')).read_text())['status']!='complete':
            raise ValueError('Execution is incomplete.')
    plots=json.loads((HERE/'plot-validation.json').read_text())
    for name,digest in plots['hashes'].items():
        if sha(HERE/name)!=digest:raise ValueError('Plot input or output changed.')
    historical_count=assert_history_unchanged()
    files=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file(): continue
        relative=path.relative_to(root)
        # Candidate netlists are intermediate; all logs and all delivered best netlists are retained.
        if path.name in ('current.v','mapped.v') and 'episodes' in relative.parts: continue
        if path.suffix=='.pt' and path.name=='50.pt': continue
        files.append(path)
    hashes={str(p.relative_to(root)):sha(p) for p in files}
    dump(root/'SHA256.json',hashes)
    with tarfile.open(HERE/'evidence.tar.gz','w:gz',compresslevel=6) as archive:
        for path in files+[root/'SHA256.json']:
            archive.add(path,arcname=str(path.relative_to(root)))
    with tarfile.open(HERE/'evidence.tar.gz','r:gz') as archive:
        archived=json.loads(archive.extractfile('SHA256.json').read())
        for name,expected in archived.items():
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest()!=expected:
                raise ValueError('Evidence archive hash mismatch: '+name)
    if (HERE/'evidence.tar.gz').stat().st_size>=90_000_000:
        raise ValueError('Evidence exceeds the Git delivery size limit.')
    models={str(p.relative_to(root)):sha(p) for p in root.rglob('*.pt')}
    dump(HERE/'model-manifest.json',dict(root=str(root),hashes=models,
        portability='evidence.tar.gz 包含初始与最终模型及探针；可解包重放；中间50轮模型仅保留本机。'))
    dump(HERE/'archive-validation.json',dict(status='complete',files=len(hashes),
         archive_sha256=sha(HERE/'evidence.tar.gz'),bytes=(HERE/'evidence.tar.gz').stat().st_size))
    delivered=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='artifact-manifest.json']
    dump(HERE/'artifact-manifest.json',dict(status='complete',fingerprint=manifest['fingerprint'],
         implementation_commit=manifest['implementation_commit'],packaged_at=now(),historical_files_unchanged=historical_count,
         sources=manifest['sources'],files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in delivered}))
    print(f'Packaged {len(hashes)} evidence files; {historical_count} historical files unchanged.',flush=True)
