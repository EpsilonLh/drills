"""Compare the prescribed runs against the preserved local nine-feature baseline."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_recorded_source(project, source_hashes):
    """Verify historical execution code against current files or immutable Git history."""
    verified = {}
    for name, expected in source_hashes.items():
        if (project / name).is_file() and hashlib.sha256((project / name).read_bytes()).hexdigest() == expected:
            verified[name] = 'working-tree'
            continue
        revisions = subprocess.check_output(['git', 'log', '--format=%H', '--', name],
                                            cwd=project, text=True).splitlines()
        for revision in revisions:
            saved = subprocess.run(['git', 'show', f'{revision}:{name}'], cwd=project,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if saved.returncode == 0 and hashlib.sha256(saved.stdout).hexdigest() == expected:
                verified[name] = revision
                break
        else:
            raise ValueError(f'Cannot verify recorded experiment source: {name}')
    return verified


def summarize_run(root, circuit, seed, protocol):
    directory = root / circuit / f'seed-{seed}'
    result = read_json(directory / 'result.json')
    budget = protocol['episodes']
    if (result['status'] != 'complete' or result['episodes_completed'] != budget
            or result['circuit'] != circuit or result['seed'] != seed):
        raise ValueError(f'Incomplete or mismatched run: {directory}')
    best = result['best']
    if not best or best != read_json(directory / 'best.json'):
        raise ValueError(f'Best candidate record mismatch: {directory}')
    maximum = protocol['circuits'][circuit]['max_levels']
    if best['feasible'] != (best['levels'] <= maximum):
        raise ValueError(f'Incorrect feasibility label: {directory}')
    cec = (directory / 'equivalence.log').read_text()
    metrics = re.findall(r'\bnd\s*=\s*(\d+)[^\n]*?\blev\s*=\s*(\d+)', cec)
    if ('Networks are equivalent' not in cec or not metrics
            or tuple(map(int, metrics[-1])) != (best['luts'], best['levels'])):
        raise ValueError(f'Final netlist verification missing or inconsistent: {directory}')
    eligible_best = None
    feasible_episodes = 0
    for episode in range(1, budget + 1):
        with (directory / 'episodes' / str(episode) / 'log.csv').open(newline='') as handle:
            steps = list(csv.DictReader(handle))
        if [int(step['iteration']) for step in steps] != list(range(protocol['iterations'] + 1)):
            raise ValueError(f'Incomplete trajectory: {directory}, episode {episode}')
        feasible_episodes += any(int(step['levels']) <= maximum for step in steps)
        for step in steps:
            iteration, luts, levels = (int(step[key]) for key in ['iteration', 'luts', 'levels'])
            if iteration == 0 and not protocol['evaluation']['include_initial']:
                continue
            rank = (0, luts, levels) if levels <= maximum else (1, levels, luts)
            if eligible_best is None or rank < eligible_best[0]:
                eligible_best = (rank, luts, levels, episode, iteration)
    if tuple(best[key] for key in ['luts', 'levels', 'episode', 'iteration']) != eligible_best[1:]:
        raise ValueError(f'Saved best does not match the complete search log: {directory}')
    return dict(luts=best['luts'], levels=best['levels'], feasible=best['feasible'],
                episode=best['episode'], iteration=best['iteration'], sequence=best['sequence'],
                training_seconds=result['training_seconds'], feasible_episodes=feasible_episodes,
                feasible_episode_percent=100 * feasible_episodes / budget)


def compare(baseline, improved, report):
    baseline, improved, report = baseline.resolve(), improved.resolve(), report.resolve()
    preserved = read_json(improved / 'baseline-provenance.json')
    project = Path(preserved['config']['runtime']['output_dir']).parent
    for name, expected in preserved['result_file_sha256'].items():
        if hashlib.sha256((project / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Original baseline file changed: {name}')
    if hashlib.sha256((project / 'params.yml').read_bytes()).hexdigest() != preserved['source_hashes']['params.yml']:
        raise ValueError('Original params.yml changed.')
    original = read_json(baseline / 'experiment.json')['config']
    augmented = read_json(improved / 'experiment.json')['config']
    if original != preserved['config']:
        raise ValueError('Baseline configuration changed.')
    method = dict(augmented['method'])
    if method.pop('performance_features', None) != ['lut_ratio', 'depth_margin']:
        raise ValueError('Unexpected performance feature definition.')
    if method != original['method'] or any(augmented[key] != original[key] for key in ['protocol', 'environment']):
        raise ValueError('The experimental conditions differ beyond the two added features.')
    for key in ['abc_binary', 'yosys_binary', 'workers']:
        if augmented['runtime'][key] != original['runtime'][key]:
            raise ValueError(f'Runtime differs: {key}')
    protocol = original['protocol']
    if protocol['evaluation']['std_ddof'] != 0:
        raise ValueError('This comparison uses population standard deviation (ddof=0).')
    rows, summaries = [], {}
    for circuit in protocol['circuits']:
        group = []
        for seed in protocol['seeds']:
            before = summarize_run(baseline, circuit, seed, protocol)
            after = summarize_run(improved, circuit, seed, protocol)
            delta = after['luts'] - before['luts']
            comparable = before['feasible'] and after['feasible']
            row = dict(circuit=circuit, seed=seed, baseline=before, improved=after,
                       lut_delta=delta, improvement_percent=100 * (-delta) / before['luts'] if comparable else None,
                       outcome=('improved' if delta < 0 else ('tied' if delta == 0 else 'regressed'))
                       if comparable else 'incomparable')
            rows.append(row)
            group.append(row)
        summary = dict(required_seeds=len(group), wins=sum(r['outcome'] == 'improved' for r in group),
                       ties=sum(r['outcome'] == 'tied' for r in group), losses=sum(r['outcome'] == 'regressed' for r in group),
                       incomparable=sum(r['outcome'] == 'incomparable' for r in group))
        for variant in ['baseline', 'improved']:
            values = [r[variant] for r in group]
            complete_feasible = all(value['feasible'] for value in values)
            summary[variant] = dict(feasible_seeds=sum(v['feasible'] for v in values),
                                    luts_mean=statistics.mean(v['luts'] for v in values) if complete_feasible else None,
                                    luts_std=statistics.pstdev(v['luts'] for v in values) if complete_feasible else None,
                                    levels_mean=statistics.mean(v['levels'] for v in values) if complete_feasible else None,
                                    levels_std=statistics.pstdev(v['levels'] for v in values) if complete_feasible else None,
                                    training_seconds_mean=statistics.mean(v['training_seconds'] for v in values),
                                    feasible_episode_percent=statistics.mean(v['feasible_episode_percent'] for v in values))
        before_mean, after_mean = summary['baseline']['luts_mean'], summary['improved']['luts_mean']
        summary['improvement_percent'] = 100 * (before_mean - after_mean) / before_mean if before_mean and after_mean is not None else None
        summaries[circuit] = summary
    provenance = read_json(improved / 'run-provenance.json')
    status = read_json(improved / 'run-status.json')
    validation = read_json(improved / 'validation.json')
    if validation['exit_code'] != 0 or validation['tests_passed'] != 9:
        raise ValueError('State compatibility and integration checks did not pass.')
    verified_sources = verify_recorded_source(project, provenance['source_hashes'])
    if status['status'] != 'complete' or status['exit_code'] != 0 or status['completed_runs'] != len(rows):
        raise ValueError('Full experiment did not finish successfully.')
    payload = dict(material_passport=dict(origin_skill='academic-research-suite/experiment-agent',
                                         origin_mode='run', origin_date=datetime.now(timezone.utc).date().isoformat(),
                                         verification_status='VERIFIED', version_label='new_state_comparison_v1'),
                   baseline_commit=preserved['baseline_commit'], baseline_source_hashes=preserved['source_hashes'],
                   baseline_directory=str(baseline), improved_directory=str(improved),
                   original_config=original, improved_config=augmented, provenance=provenance,
                   run_status=status, validation=validation, verified_sources=verified_sources,
                   baseline_files_unchanged=True, rows=rows, summary=summaries)
    payload['interpretation_checks'] = [
        dict(name="Simpson's paradox", finding='Each circuit and seed is reported separately; no pooled cross-circuit score.'),
        dict(name='Ecological fallacy', finding='Means are not claimed to describe every seed; per-seed outcomes are shown.'),
        dict(name="Berkson's paradox", finding='All fixed benchmark circuits are included; no claim about the population of circuits.'),
        dict(name='Collider bias', finding='Not applicable: no covariate-adjusted inference.'),
        dict(name='Base rate neglect', finding='Feasible seed and episode counts include the full prescribed denominators.'),
        dict(name='Regression to the mean', finding='No selection of baseline extreme seeds; fixed seeds 0/1/2 are reused.'),
        dict(name='Survivorship bias', finding='All prescribed runs must be complete; infeasible runs prevent feasible-score aggregation.'),
        dict(name='Look-elsewhere effect', finding='All circuits and seeds are reported; no significance tests or significance-based selection.'),
        dict(name='Garden of forking paths', finding='Exploratory experiment follows the user-approved budget and metrics; no tuning or extra seed selection.'),
        dict(name='Correlation versus causation', finding='Two features and input-layer parameters change together; no isolated causal contribution is claimed.'),
        dict(name='Reverse causality', finding='Feature definitions are fixed before training; no reversed directional inference from aggregate scores.'),
    ]
    (improved / 'comparison.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    flat = []
    for row in rows:
        item = {key: row[key] for key in ['circuit', 'seed', 'lut_delta', 'improvement_percent', 'outcome']}
        for variant in ['baseline', 'improved']:
            item.update({f'{variant}_{key}': value for key, value in row[variant].items() if key != 'sequence'})
        flat.append(item)
    with (improved / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ['# 新增映射性能状态的本地对照实验', '', '## Material Passport', '',
             '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: run',
             f'- Origin Date: {payload["material_passport"]["origin_date"]}', '- Verification Status: VERIFIED',
             '- Version Label: new_state_comparison_v1', '',
             '## 汇总结果', '',
             '原版采用之前本地运行的九维状态结果；改进版新增 LUT 比例和层数余量。正改善率表示 LUT 减少。标准差使用 ddof=0。', '',
             '| 电路 | 原版 LUT 均值 ± 标准差 | 改进版 LUT 均值 ± 标准差 | LUT 改善率 | 原版 / 改进版 levels 均值 | 可行种子 原版 / 改进版 | 改善 / 持平 / 退步 |',
             '|---|---:|---:|---:|---:|---|---|']
    def formatted(value, key):
        return '未汇总' if value[key + '_mean'] is None else f'{value[key + "_mean"]:.3f} ± {value[key + "_std"]:.3f}'
    for circuit, summary in summaries.items():
        before, after = summary['baseline'], summary['improved']
        change = '未汇总' if summary['improvement_percent'] is None else f'{summary["improvement_percent"]:+.3f}%'
        lines.append(f'| {circuit} | {formatted(before, "luts")} | {formatted(after, "luts")} | {change} | '
                     f'{before["levels_mean"]} / {after["levels_mean"]} | '
                     f'{before["feasible_seeds"]}/{summary["required_seeds"]} / {after["feasible_seeds"]}/{summary["required_seeds"]} | '
                     f'{summary["wins"]} / {summary["ties"]} / {summary["losses"]} |')
    lines += ['', '## 逐种子结果', '', 'LUT 差值 = 改进版 − 原版，负值表示改善。找到最佳解的位置写作“回合 / 步骤”。', '',
              '| 电路 | 种子 | 原版 LUT | 改进版 LUT | LUT 差值 | 原版 levels | 改进版 levels | 最佳位置 原版 → 改进版 |',
              '|---|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        before, after = row['baseline'], row['improved']
        lines.append(f'| {row["circuit"]} | {row["seed"]} | {before["luts"]} | {after["luts"]} | {row["lut_delta"]:+d} | '
                     f'{before["levels"]} | {after["levels"]} | {before["episode"]} / {before["iteration"]} → '
                     f'{after["episode"]} / {after["iteration"]} |')
    lines += ['', '## 可行回合与耗时', '',
              '“可行回合”指该回合第 0–10 步至少出现一次满足层数约束的候选；不是整条轨迹始终可行，也不是最终一步可行。',
              '两组不是同期运行，耗时仅供参考，不作为加速结论。训练计时包含环境初始化、工具调用和网络更新，排除检查点导出、模型初始化、resyn2 与最终 CEC。', '',
              '| 电路 | 种子 | 可行回合 原版 | 可行回合 改进版 | 原版训练秒数 | 改进版训练秒数 |',
              '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        before, after = row['baseline'], row['improved']
        lines.append(f'| {row["circuit"]} | {row["seed"]} | {before["feasible_episodes"]}/{protocol["episodes"]} | '
                     f'{after["feasible_episodes"]}/{protocol["episodes"]} | {before["training_seconds"]:.3f} | {after["training_seconds"]:.3f} |')
    lines += ['', '| 电路 | 原版可行回合比例 | 改进版可行回合比例 | 原版平均训练秒数 | 改进版平均训练秒数 |',
              '|---|---:|---:|---:|---:|']
    for circuit, summary in summaries.items():
        before, after = summary['baseline'], summary['improved']
        lines.append(f'| {circuit} | {before["feasible_episode_percent"]:.2f}% | {after["feasible_episode_percent"]:.2f}% | '
                     f'{before["training_seconds_mean"]:.3f} | {after["training_seconds_mean"]:.3f} |')
    lines += ['', '## 实现与验证', '',
              '- 新状态为 `[原九维结构特征, 当前 LUT / 第 0 步 LUT, (层数上限 − 当前映射 levels) / 层数上限]`。',
              '- 原九维仍按每回合 Welford 标准化；新增两维直接拼接，保留余量为零的约束边界。',
              '- Actor 和 Critic 输入由 9 变为 11，隐藏层保持不变；总参数从 878 增至 938。',
              '- 9 项自动检查通过，覆盖关闭特征时与原版精确一致、新特征输入、无额外工具调用及断点续训。',
              '- 改进版 9 次训练完成；最终网表指标复核与 CEC 均通过。对比脚本逐回合重算最佳候选并核对导出记录。',
              f'- 对原版 {len(preserved["result_file_sha256"])} 个结果文件的 SHA-256 检查通过；原 params.yml 未修改。', '',
              '## 实验设置与复现', '',
              '三个电路：int2float、i2c、max；种子 0/1/2；100 回合 × 10 步；LUT6；层数上限 3/4/41；三个工作进程，各一个 PyTorch 线程。奖励、动作、网络隐藏层、优化器及最终选解规则保持原设置。', '',
              f'原版提交：`{preserved["baseline_commit"]}`；改进分支：`{provenance["branch"]}`，未提交工作区的文件哈希记录在 JSON 中。',
              f'运行环境：Python {provenance["python"].split()[0]}，PyTorch {provenance["torch"]}，NumPy {provenance["numpy"]}；{provenance["platform"]}。',
              f'Yosys：`{provenance["yosys"]}`；ABC：`{provenance["abc"].splitlines()[-1]}`。',
              f'改进版整次作业墙钟耗时：{status["elapsed_seconds"]:.3f} 秒，含并行训练与基线及最终验证。', '',
              '```bash', '.tools/conda-env/bin/python -m unittest discover -s tests -v',
              '.tools/conda-env/bin/python drills.py train fpga params-new-state.yml',
              '.tools/conda-env/bin/python compare_new_state.py', '```', '',
              '重跑训练需为配置指定新的输出目录；原结果目录已有数据时不会被覆盖。续训仅适用于相同特征定义。',
              f'完整数据：`{improved / "comparison.csv"}`、`{improved / "comparison.json"}`；训练日志：`{improved / "training.log"}`。', '',
              '## 结果解释', '']
    for circuit, summary in summaries.items():
        change = summary['improvement_percent']
        if change is None:
            sentence = f'{circuit}：至少一组存在不可行种子，不能计算全部预定种子的可行 LUT 汇总。'
        else:
            label = '减少' if change > 0 else ('增加' if change < 0 else '不变')
            sentence = f'{circuit}：LUT 均值{label}' + (f' {abs(change):.3f}%。' if change else '。')
        lines.append('- ' + sentence)
    totals = {key: sum(summary[key] for summary in summaries.values()) for key in ['wins', 'ties', 'losses']}
    lines += ['', f'逐种子共 {totals["wins"]} 次改善、{totals["ties"]} 次持平、{totals["losses"]} 次退步。']
    for circuit, summary in summaries.items():
        if summary['improved']['luts_std'] is not None and summary['improved']['luts_std'] > summary['baseline']['luts_std']:
            lines.append(f'{circuit} 的种子间 LUT 标准差由 {summary["baseline"]["luts_std"]:.3f} 增至 '
                         f'{summary["improved"]["luts_std"]:.3f}，需要同时看待均值和波动。')
    lines += ['', '这组结果比较的是训练期间搜到的最佳可行映射，不是冻结模型的独立测试或跨电路泛化。',
              '每个电路只有三个种子，仅作描述性比较，不计算显著性或声称稳定提升。两维同时加入，且增加了输入层参数；不能单独归因于某一维或证明性能变化只来自信息增量。',
              '每个模型仍按各自相同种子初始化；输入维度变化会改变初始化随机数消耗，不能理解为两组共有参数或动作轨迹完全相同。',
              '本实验未做超参数调优、单维消融或补跑挑选种子。AI 协助实现、运行、核对日志及整理结果。', '']
    lines += ['已检查 11/11 类统计解释问题；完整检查记录在 comparison.json 中。这里的验证表示执行、日志及汇总一致，不代表已经证明该方法普遍有效。', '']
    report.write_text('\n'.join(lines), encoding='utf-8')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('results'))
    parser.add_argument('--improved', type=Path, default=Path('results/new-state'))
    parser.add_argument('--report', type=Path, default=Path('new-state-comparison.md'))
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.improved, args.report), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
