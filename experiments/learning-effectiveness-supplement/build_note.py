"""Prepare the Obsidian supplement inside the writable repository for review."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUITE = ROOT / 'experiments/learning-effectiveness'
NOTE = Path('/Users/zhan/Documents/Obsidian Vault/DRiLLS/改进实验/学习有效性实验.md')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    validation = json.loads((HERE / 'validation.json').read_text())
    summary = json.loads((HERE / 'recomputed-summary.json').read_text())
    if validation['status'] != 'complete' or len(validation['inference_replays']) != 12:
        raise ValueError('Complete evidence audit and inference replay required.')
    original = HERE / 'original-note.md'
    if not original.exists():
        shutil.copyfile(NOTE, original)
    if sha(NOTE) != sha(original):
        raise ValueError('The source Obsidian note changed; rebuild against its latest contents.')
    text = original.read_text()
    index = {(r['circuit'], r['policy']): r for r in summary['aggregate']}
    policies = ('initial', 'trained-50', 'trained-100', 'uniform')
    table = ['| 电路 | 初始化/冻结 | 第50轮模型 | 第100轮模型 | 均匀随机 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name in ('int2float', 'i2c'):
        table.append('| ' + name + ' | ' + ' | '.join(f'{index[name, p]["best_luts_mean"]:.3f}' for p in policies) + ' |')
    table.append('| max：找到可行解的比例 | 32/90（35.6%） | 32/90（35.6%） | 32/90（35.6%） | 10/30（33.3%） |')
    comparisons = ['| 电路 | 对照 | 第100轮平均LUT减少量 | 探索性95%区间 | 可行率变化/百分点及区间 | 可行优先胜/平/负 |',
                   '| --- | --- | ---: | --- | --- | --- |']
    for row in summary['comparisons']:
        value = f'{row["mean_luts_reduction"]:.3f}' if row['mean_luts_reduction'] is not None else '不汇总'
        interval = '[' + ', '.join(f'{v:.3f}' for v in row['ci95']) + ']' if row['ci95'] is not None else '不汇总'
        feasible_interval = '[' + ', '.join(f'{v:.3f}' for v in row['feasibility_ci95']) + ']'
        baseline = '初始化/冻结' if row['baseline'] == 'initial' else '均匀随机'
        comparisons.append(f'| {row["circuit"]} | {baseline} | {value} | {interval} | {row["feasibility_gain_pp"]:.3f} / {feasible_interval} | {row["wins"]}/{row["ties"]}/{row["losses"]} |')
    seed_table = ['| 电路 | 对照 | seed 0 | seed 1 | seed 2 |',
                  '| --- | --- | ---: | ---: | ---: |']
    for row in summary['comparisons']:
        if row['mean_luts_reduction'] is None:
            continue
        baseline = '初始化/冻结' if row['baseline'] == 'initial' else '均匀随机'
        seed_table.append('| ' + row['circuit'] + ' | ' + baseline + ' | ' +
                          ' | '.join(f'{s["luts_reduction"]:.3f}' for s in row['per_seed']) + ' |')
    supplement = '''## 独立策略评估补充（2026-09-27核对）

### Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：validate + 既有报告补充
- Verification Status：VERIFIED（既有原始证据与统计复核，12条固定策略序列精确重放）
- Version Label：learning_effectiveness_100x10_note_supplement_v1

本地仓库的100×10实验实际上已经完成固定策略独立评估，原Obsidian笔记未收录这部分。原评估完成于2026-09-26 UTC（日本时间2026-09-27），本次补充复用了该批证据，并重新核对全部评估日志和统计结果；另重放12条序列验证可复现性。本次没有重新训练，也没有重新执行整批评估。

### 评估方案与“独立”的含义

训练种子仍为0、1、2。固定第0、50、100轮模型，使用新采样种子10000–10029，各执行30条10步序列。每条序列从原电路开始，全程不更新权重；归一化、动作集合、LUT6及深度限制保持原设置。第100轮为最终主比较，第50轮仅用于观察过程。

独立指评估序列未参与训练更新。模型仍按状态给出的动作概率采样，评估种子控制采样随机数；权重保持不变。这里仍评估原有三个电路，不能解释为对未见电路的泛化测试。冻结模型经网络及Adam不变性检查后，复用初始化模型评估。

完整记录为36/36个评估任务、1080/1080条实际执行序列、10800个动作。旧评估程序对uniform的三个训练种子各执行了一份bank，但同一电路的三份30条序列完全相同；共享随机基线每电路只有30条独立序列。合并重复uniform bank后，相当于900个模型/电路/评估种子组合，重复运行不增加独立样本量。配对分层bootstrap保留这些相关性。

### 初始化、中期与最终模型

int2float、i2c的数字是每条10步序列中“最佳可行LUT”的平均值，包括第0步初始映射，越小越好；每个模型策略汇总3个训练种子×30条序列，均为90/90可行。uniform为每电路共享的30条基线。max存在不可行序列，完整报告可行率，不筛掉失败后计算LUT均值。

''' + '\n'.join(table) + '''

![[学习有效性-100x10-独立评估.svg]]

### 第100轮模型对两种基线的主比较

''' + '\n'.join(comparisons) + '''

正值表示训练模型更好。胜/平/负保留所有序列，采用可行优先排序：可行时先比较LUT，再比较层数；两者均不可行时先比较层数，再比较LUT。表中的胜负也可能反映层数差异，不能全部解释为LUT收益。

区间复用原方案：配对分层bootstrap，先重采样3个训练种子，再共同重采样30个评估种子列，保留共同随机数和共享uniform基线的相关性；重复10000次，统计种子20260927。只有3次独立训练，区间为探索性估计，六个比较未作多重比较校正，也未预设实用收益门槛。

逐训练种子的LUT减少量如下，正值表示改善，负值表示退化：

''' + '\n'.join(seed_table) + '''

### 逐电路解释

**int2float：未验证稳定收益。** 第100轮相对初始化平均少0.100 LUT，相对随机少0.078 LUT，幅度小；两个95%区间都包含零。第50轮和第100轮汇总均值也只相差0.044 LUT。不能据此证明模型无效或三组等效。

**i2c：平均有局部改善，但训练种子之间不一致。** 第100轮相对初始化平均少0.911 LUT（约0.28%），相对随机少1.289 LUT（约0.40%），两个95%区间均包含零。seed 0相对初始化/随机分别少2.700/4.767 LUT；seed 1反而多0.100/2.233 LUT；seed 2只有较小改善。均值受到seed 0影响，当前证据不足以表述为一致收益。

**max：未验证约束满足能力稳定提高。** 第100轮与初始化同为32/90可行，逐训练种子分别为11/30、10/30、11/30，对应初始化10/30、11/30、11/30。相对随机10/30可行，点估计高2.222个百分点，但区间为[-2.222, 8.889]，包含零；可行率仍约三分之一。

这里的主指标与“第10步终点”不同。终点作为辅助指标：int2float初始化/第100轮/随机终点可行率分别为76/90、80/90、23/30；i2c全部终点可行，终点平均LUT分别为322.111、321.944、323.900；max终点可行数与上述序列最佳可行数相同。不能把序列中找到的较好解当作最后一步必然保持的结果。

### 本次核对与证据位置

本次逐条核对1080份rollout.json及逐步log.csv，重算动作、奖励、最佳可行值和终点，并与evaluation.csv逐项对齐；六个bootstrap主比较与原summary.json完全一致。训练源码、配置、工具、输入指纹一致；135份模型及探针文件哈希一致，冻结网络及Adam不变。37份原交付文件当前哈希一致；根README因后续实验扩展已更新，其原版本通过Git历史哈希核对。

原CEC验证覆盖1110份导出结果、2217个映射/未映射网表；本次核对全部网表、验证日志的哈希及通过记录，并校验证据压缩包内10687份文件。另对9个第100轮模型和3个uniform电路基线各重放评估种子10000的一条10步序列，12/12的动作、奖励、最佳值和终点精确一致，权重、检查点及全局Torch随机数状态保持不变。这不等于全部训练或全部评估都另做了一次独立重复。

统计解释风险核对11/11：逐电路及逐种子效应、独立训练单位、选择偏差、可行筛选导致的碰撞偏差、完整可行基率、均值回归、幸存者偏差、多重比较、分析分叉、相关当因果和反向因果。少训练种子和探索性区间的限制仍然存在。

- [原始逐序列CSV](/Users/zhan/Downloads/test/DRiLLS/experiments/learning-effectiveness/evaluation.csv)
- [原独立评估完整报告](/Users/zhan/Downloads/test/DRiLLS/experiments/learning-effectiveness/report.md)
- [本次重算汇总JSON](/Users/zhan/Downloads/test/DRiLLS/experiments/learning-effectiveness-supplement/recomputed-summary.json)
- [本次验证记录](/Users/zhan/Downloads/test/DRiLLS/experiments/learning-effectiveness-supplement/validation.json)
- [只读核对及抽样重放脚本](/Users/zhan/Downloads/test/DRiLLS/experiments/learning-effectiveness-supplement/validate.py)

与200×5比较时需要区分评估预算：本报告每条评估序列为10步，200×5报告为5步，评估种子集合也不同。两份报告可以各自判断学习相对基线的收益，不能直接将绝对LUT均值或max可行率差异归因于训练轮数或单轮步数。

'''
    marker = '## 数据来源与核对口径'
    if text.count(marker) != 1:
        raise ValueError('Unexpected source note structure.')
    text = text.replace(marker, supplement + marker)
    text = text.replace('![[search-curves.svg]]', '![[学习有效性-100x10-搜索曲线.svg]]')
    conclusion = '''## 总结性结论

**补入独立评估后，最终结论仍是：训练发生了参数和动作概率更新，但100轮×10步设置下尚未验证稳定、明确的收益。**

训练搜索累计最好值方面，训练相对冻结在int2float平均多0.333 LUT、i2c平均少1.000 LUT、max平均多2.000 LUT；训练相对uniform分别持平、平均少0.667 LUT、平均少2.333 LUT。这些差异随电路及训练种子变化，没有形成对两种不学习基线的持续优势。达到自身最佳值较早，也不能单独说明策略更好。

固定第100轮模型的独立评估方面，int2float相对两基线只改善约0.1 LUT；i2c平均少0.911/1.289 LUT，但seed 1退化、收益集中在seed 0；max相对初始化可行率没有提高。六个主比较的探索性95%区间都包含零，三个电路均未形成稳定收益证据。

**可以表述为“学习更新发生了，存在局部改善，但当前证据不足以确认稳定收益”。** 不能表述为“证明学习无效”“三组等效”或“已经接近10步最优解”；也不能用这次结果证明单轮步数太多就是优势不明显的原因。只有3个独立训练种子，评估仍限于训练电路。

AI用于补充报告整理、证据核对、统计重算及脚本实现。

---

'''
    if text.count('## 总结性结论') != 1:
        raise ValueError('Unexpected conclusion section count.')
    text = text[:text.index('## 总结性结论')] + conclusion
    text = '# DRiLLS 100轮 × 10步学习有效性实验\n\n2026-09-27补充：已收录固定策略独立评估，并复核原始证据。**最终未验证稳定收益；i2c存在部分种子的局部改善。**\n\n' + text
    target = HERE / NOTE.name
    target.write_text(text)
    figures = {'policy-evaluation.svg': '学习有效性-100x10-独立评估.svg',
               'search-curves.svg': '学习有效性-100x10-搜索曲线.svg'}
    for source, filename in figures.items():
        shutil.copyfile(SUITE / source, HERE / filename)
    receipt = dict(target=str(NOTE), original_sha256=sha(original), prepared_sha256=sha(target),
                   files={filename: sha(HERE / filename) for filename in [NOTE.name, *figures.values()]})
    (HERE / 'prepared-manifest.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(f'Prepared report: {target}')


if __name__ == '__main__':
    main()
