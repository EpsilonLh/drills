"""Chinese report and predeclared, conservative conclusion rules."""
from pathlib import Path


def number(value, digits=3):
    return '未汇总' if value is None else f'{value:.{digits}f}'


def interval(value):
    return '未汇总' if value is None else '[' + ', '.join(number(v) for v in value) + ']'


def comparison_verdict(row):
    if not row['complete']:
        return 'incomplete'
    metric, uncertainty = ('mean_luts_reduction', 'ci95') if row['complete_feasible'] else (
        'feasibility_gain_pp', 'feasibility_ci95')
    seed_metric = 'luts_reduction' if row['complete_feasible'] else 'feasibility_gain_pp'
    estimate, bounds = row[metric], row[uncertainty]
    values = [p[seed_metric] for p in row['per_seed']]
    if bounds[0] > 0 and all(v is not None and v > 0 for v in values):
        return 'supported_improvement'
    if bounds[1] < 0 and all(v is not None and v < 0 for v in values):
        return 'supported_decline'
    return 'point_improvement' if estimate > 0 else 'point_decline' if estimate < 0 else 'no_difference'


def circuit_verdict(comparisons):
    if len(comparisons) != 2 or any(comparison_verdict(r) == 'incomplete' for r in comparisons):
        return '主评估未完整完成，不判断学习收益'
    verdicts = [comparison_verdict(r) for r in comparisons]
    if all(v == 'supported_improvement' for v in verdicts):
        return '本设置下观察到一致收益'
    if 'supported_decline' in verdicts:
        return '主指标存在退化证据，未验证稳定收益'
    if any(v in ('point_improvement', 'supported_improvement') for v in verdicts):
        return '存在局部改善，未验证对两种基线的稳定收益'
    return '未验证稳定收益'


def write_report(summary, destination):
    destination = Path(destination)
    protocol = summary['protocol']
    circuits, seeds = protocol['circuits'], protocol['seeds']
    expected_training = protocol['expected_training_runs']
    expected_rollouts = protocol['expected_physical_rollouts']
    expected_banks = protocol['expected_physical_banks']
    main = f"trained-{protocol['assessment']['main_checkpoint']}"
    complete = summary['training_complete'] == expected_training and summary['physical_banks_complete'] == expected_banks
    cec = summary['netlist_validation']
    lines = [
        '# DRiLLS 200轮 × 5步：学习有效性实验报告', '',
        '## Material Passport', '',
        '- Origin Skill: academic-research-suite / experiment-agent',
        '- Origin Mode: run + validate', '- Origin Date: 2026-09-27',
        '- Verification Status: ' + ('VERIFIED（短程原实现一致性、断点恢复及完整日志/网表复核）'
            if complete and cec.get('status') == 'complete' and cec.get('protocol_complete') else 'INCOMPLETE'),
        '- Version Label: learning_effectiveness_200x5_v1', '', '## 主要结论', '',
        f"训练完成 {summary['training_complete']}/{expected_training} 次；独立评估完成 "
        f"{summary['physical_rollouts']}/{expected_rollouts} 条实际序列，"
        f"{summary['physical_banks_complete']}/{expected_banks} 个实际评估bank。", '',
    ]
    for name in circuits:
        comparisons = [r for r in summary['comparisons'] if r['circuit'] == name]
        lines.append(f'**{name}：{circuit_verdict(comparisons)}。**')
        for row in comparisons:
            baseline = '初始化/冻结' if row['baseline'] == 'initial' else '均匀随机'
            if not row['complete']:
                lines.append(f'第200轮相对{baseline}：未完成配对。')
            elif row['complete_feasible']:
                lines.append(f"第200轮相对{baseline}平均减少 {number(row['mean_luts_reduction'])} LUT，"
                    f"探索性95%区间 {interval(row['ci95'])}；逐训练种子减少量 "
                    f"{[round(p['luts_reduction'], 3) for p in row['per_seed']]}。")
            else:
                lines.append(f"第200轮可行 {row['trained_feasible']}/{len(seeds)*30}，{baseline}可行 "
                    f"{row['baseline_feasible']}/{len(seeds)*30}；可行率变化 "
                    f"{number(row['feasibility_gain_pp'])} 个百分点，探索性95%区间 "
                    f"{interval(row['feasibility_ci95'])}。存在不可行序列，不汇总筛选后的LUT收益。")
        lines.append('')
    trained = [r for r in summary['diagnostic_summary'] if r['group'] == 'trained']
    fixed = [r for r in summary['diagnostic_summary'] if r['group'] != 'trained']
    lines += [
        f"训练组Actor/Critic均改变 {sum(r['actor_drift_rms'] > 0 and r['critic_drift_rms'] > 0 for r in trained)}/{len(trained)} 次；"
        f"固定探针动作分布改变 {sum(r['policy_kl'] > 0 for r in trained)}/{len(trained)} 次；"
        f"冻结及uniform参数保持不变 {sum(r['actor_drift_rms'] == r['critic_drift_rms'] == 0 for r in fixed)}/{len(fixed)} 次。"
        '参数变化本身不作为搜索收益证据。', '',
        '本实验检验200×5设置下的策略收益。相对100×10，它同时缩短序列并增加更新次数；'
        '相对50×50还减少了动作预算。无论结果是否改善，都不能据此单独证明“原先步数太多掩盖优势”。', '',
        '## 条件与统计口径', '',
        '三组trained/frozen/uniform，训练种子0/1/2，每次200轮×5步；LUT6，'
        'int2float/i2c/max最大层数3/4/41。学习率0.001、原奖励、网络、Adam、折扣因子、'
        '状态及回报归一化均保持原设置。trained每轮更新一次；frozen及uniform停止更新。', '',
        '保存第0/100/200轮模型，第200轮是预定主结果，第100轮仅观察学习过程。'
        '各模型使用20000–20029评估种子，每条5步序列重新从原电路开始，全程无更新。'
        '初始映射允许成为最优解，但不计入动作候选。冻结经不变性验证后共享初始化评估。', '',
        '900条实际序列组成30个bank：3电路×3训练种子×3检查点，加每电路一个uniform bank。'
        'uniform在配对CSV中共享给三个训练种子，形成1,080条逻辑行；这些重复行不增加独立样本量。', '',
        '排序为可行优先；可行时先LUT再层数，不可行时先层数再LUT。'
        '只要配对含不可行序列，就报告完整可行率和可行优先胜/平/负，不计算筛选后的LUT平均收益。'
        '终点LUT、层数和奖励为辅助指标。0/0表示缺失，不表示0%可行。', '',
        '六个主比较使用配对分层bootstrap，先重采样训练种子，再共同重采样评估种子列，'
        '保留共同随机数及uniform共享bank的相关性。10,000次，统计种子20260927。'
        '只有3次独立训练，区间是探索性估计，不把90条配对序列当作90次独立训练，不宣称统计等效。', '',
        '预定结论规则：某电路对两基线的主指标区间下界均大于零，且三个训练种子均朝改善方向，'
        '才表述为“本设置下观察到一致收益”。混合可行情况下主指标是可行率；全可行情况下为LUT减少量。'
        '未规定实用收益门槛，也未进行多重比较校正；上述规则仅用于本实验的探索性描述。', '',
        '## 第200轮独立策略主比较', '',
        '| 电路 | 基线 | LUT减少量 | 探索性95%区间 | 可行率变化/百分点及区间 | 可行优先胜/平/负 |',
        '|---|---|---:|---|---|---|',
    ]
    for r in summary['comparisons']:
        score = f"{r['wins']}/{r['ties']}/{r['losses']}" if r['complete'] else '未完成配对'
        lines.append(f"| {r['circuit']} | {r['baseline']} | {number(r['mean_luts_reduction'])} | "
            f"{interval(r['ci95'])} | {number(r['feasibility_gain_pp'])} / {interval(r['feasibility_ci95'])} | {score} |")
    lines += ['', '正值表示训练更好。逐种子效应、可行数量和配对结果见comparisons.csv；逐条数据见evaluation.csv。', '',
        '## 初始化、中期与最终模型', '',
        '| 电路 | 训练seed | 策略 | 完成/30 | 最佳解可行/30 | 最佳LUT均值 | 终点可行/30 | 终点LUT均值 | 终点层数均值 | 奖励和均值 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['evaluation']:
        lines.append(f"| {r['circuit']} | {r['training_seed']} | {r['policy']} | {r['completed']}/30 | "
            f"{r['feasible']}/30 | {number(r['best_luts_mean'])} | {r['terminal_feasible']}/30 | "
            f"{number(r['terminal_luts_mean'])} | {number(r['terminal_levels_mean'])} | {number(r['reward_mean'])} |")
    lines += ['', '![固定策略评估](policy-evaluation.svg)', '', '## 训练期间搜索表现（辅助结果）', '',
        '| 电路 | 组别 | seed | 状态 | 完成轮数 | 最佳LUT/层数/可行 | 首次最佳候选 | 候选可行率 |',
        '|---|---|---:|---|---:|---|---:|---:|']
    for r in summary['training']:
        rate = 100*r['feasible_candidate_rate'] if r['feasible_candidate_rate'] is not None else None
        lines.append(f"| {r['circuit']} | {r['group']} | {r['seed']} | {r['status']} | {r['episodes_completed']} | "
            f"{r['best_luts']}/{r['best_levels']}/{r['best_feasible']} | {r['best_candidate']} | {number(rate)}% |")
    lines += ['', '![训练搜索曲线](search-curves.svg)', '', '## 相同1,000候选预算的历史比较', '',
        '| 电路 | 组别 | seed | 旧100×10 LUT/层数/可行 | 新200×5 LUT/层数/可行 |',
        '|---|---|---:|---|---|']
    for r in summary['training']:
        old = next(x for x in summary['old_budget'] if (x['group'],x['circuit'],x['seed']) == (r['group'],r['circuit'],r['seed']))
        old_text = f"{old['best_luts']}/{old['best_levels']}/{old['best_feasible']}" if old['status'] == 'complete' else '历史数据未提供'
        new_text = f"{r['best_luts']}/{r['best_levels']}/{r['best_feasible']}" if r['status'] == 'complete' else '新预算未完成'
        lines.append(f"| {r['circuit']} | {r['group']} | {r['seed']} | {old_text} | {new_text} |")
    lines += ['', '历史数据只读，来源哈希见analysis-provenance.json。旧模型不参与新实验运行或独立评估。'
        '相同动作数不等于相同序列空间、更新次数或运行时间，历史差值不能单独归因于步数。', '',
        '## 更新诊断、完成情况与限制', '',
        '| 电路 | 组别 | seed | 保存轮次 | Actor漂移RMS | Critic漂移RMS | 探针KL | 熵 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary['diagnostic_summary']:
        lines.append(f"| {r['circuit']} | {r['group']} | {r['seed']} | {r['episode']} | "
            f"{number(r['actor_drift_rms'],6)} | {number(r['critic_drift_rms'],6)} | {number(r['policy_kl'],6)} | {number(r['entropy'],6)} |")
    lines += ['', '探针固定为首轮归一化输入，诊断不消耗采样RNG；探针变化不能证明整个状态空间的策略质量。'
        '奖励与归一化机制本轮未干预，不将它们解释成已验证的收益原因。', '']
    for r in summary['failures']:
        reason = '30分钟硬超时' if r['timed_out'] else f"退出码{r['exit_code']}"
        lines.append(f"- {r['phase']}/{r['label']}：{reason}，耗时{r['elapsed_seconds']:.1f}秒。")
    for r in summary['skipped_evaluation']:
        lines.append(f"- {r['policy']}/{r['circuit']}/seed-{r['seed']}：{r['reason']}。")
    if not summary['failures'] and not summary['skipped_evaluation']:
        lines.append('已记录的执行任务无失败或超时；完整性以顶部完成数量及验证记录为准。')
    lines += ['', '3个workers、每进程1个Torch线程，每分钟监控，单任务30分钟硬超时。'
        '耗时包含诊断开销与共享资源并发，仅按候选预算比较搜索质量，不据此给算法速度排名。', '']
    for phase, r in summary['execution'].items():
        seconds = r.get('elapsed_seconds')
        lines.append(f"- {phase}：{number(seconds/60 if seconds is not None else None)}分钟；状态{r.get('status')}。")
    lines += ['', '只涉及三个训练电路及三个训练种子，不是未见电路泛化验证，也未证明5步或全局最优。'
        '核查了聚合反转、推断单位、选择/碰撞偏差、基率、回归均值、幸存者偏差、多重比较、'
        '分析分叉、相关当因果和反向因果共11类统计风险（11/11）；种子少、探索性区间及跨设置混杂仍限制结论。', '',
        '## 验证、复现与代码交付', '',
        'check.py保存5步原实现一致性、冻结、uniform、RNG隔离、断点恢复、快照、缺失和不可行处理测试，'
        '以及原回归测试。log-validation.json重放原始日志，核对27,000训练动作、奖励、最优值、可行率、'
        '4,500评估动作及评估模型哈希。', '',
        f"网表复核状态：{cec.get('status','尚未生成')}；检查{cec.get('exports_checked',0)}份导出结果，"
        f"通过{cec.get('netlists_verified',0)}个映射/未映射网表的CEC。详细命令、工具与网表哈希见netlist-validation.json。", '',
        '完整训练每个种子只运行一次；短程精确回归、恢复测试和日志复核不等于所有完整实验都做了独立重复。'
        '源码、工具、输入及配置指纹见provenance.json，测试记录见test-results.json。', '',
        '本套件代码、协议、CSV、图表、报告和evidence.tar.gz提交到本地learning-effectiveness分支。'
        '原始权重及完整运行输出位于results/learning-effectiveness-200x5/，不纳入Git；'
        'model-manifest.json记录每份权重哈希。其他机器固定模型重放需复制这些权重，或重新训练。', '',
        '在仓库根目录执行，先按本机环境设置protocol.yml工具路径：', '', '```bash',
        '.tools/conda-env/bin/python -B experiments/learning-effectiveness-200x5/check.py',
        '.tools/conda-env/bin/python -B -u experiments/learning-effectiveness-200x5/run.py',
        '.tools/conda-env/bin/python -B -u experiments/learning-effectiveness-200x5/evaluate.py',
        '.tools/conda-env/bin/python -B experiments/learning-effectiveness-200x5/verify.py',
        '.tools/conda-env/bin/python -B experiments/learning-effectiveness-200x5/evaluate.py --report-only',
        '# plot.py使用带ReportLab和PyMuPDF的Python环境；随后使用训练环境运行package.py',
        '.tools/conda-env/bin/python -B experiments/learning-effectiveness-200x5/package.py',
        '```', '', '相同指纹中断续跑：run.py --resume / evaluate.py --resume。首次重跑使用干净结果目录，'
        '源码或协议变化后不能复用旧检查点。仅重新分析已有证据使用--report-only。'
        '本套件不依赖历史模型；历史CSV缺失时只省去历史比较。', '',
        'AI用于实验代码实现、日志核对、统计、图表和报告撰写。',
    ]
    (destination/'report.md').write_text('\n'.join(lines)+'\n')
