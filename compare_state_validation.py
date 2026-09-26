"""Report matched-dimension information ablation with prespecified paired analysis."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess

import numpy as np

from compare_new_state import read_json, summarize_run, verify_recorded_source


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results/state-validation'


def paired_intervals(differences, plan):
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(plan['bootstrap_seed'])
    means = values[rng.integers(0, len(values), size=(plan['bootstrap_samples'], len(values)))].mean(axis=1)
    intervals = {}
    for key, coverage in [('descriptive_95', plan['descriptive_interval_percent']),
                          ('familywise', plan['familywise_interval_percent'])]:
        tail = (100 - coverage) / 200
        intervals[key] = np.quantile(means, [tail, 1 - tail]).tolist()
    return intervals


def validate_history():
    preserved = read_json(OUTPUT / 'preserved-history.json')
    for name, expected in {**preserved['files'], **preserved['configs']}.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Historical result or configuration changed: {name}')
    initialization = read_json(OUTPUT / 'initialization-check.json')
    if initialization['inputs'] != 11 or initialization['parameters'] != 938:
        raise ValueError('The matched control must use the same eleven-dimensional network.')
    if [item['seed'] for item in initialization['seeds']] != list(range(10)):
        raise ValueError('All ten prespecified initializations must be checked.')
    for item in initialization['seeds']:
        if not item['exact_match'] or len({item['zero'], item['performance'], item['historical_performance']}) != 1:
            raise ValueError(f'Initial weights differ for seed {item["seed"]}.')
    validation = read_json(OUTPUT / 'validation.json')
    if validation['exit_code'] != 0 or validation['tests_passed'] != 15:
        raise ValueError('State mask integration checks did not pass.')
    return preserved, initialization, validation


def validate_stage(label):
    directory = OUTPUT / label
    provenance = read_json(directory / 'provenance.json')
    status = read_json(directory / 'status.json')
    if status['status'] != 'complete' or status['exit_code'] != 0 or status['complete_runs'] != status['required_runs']:
        raise ValueError(f'Stage incomplete: {label}')
    for name, expected in provenance['source_sha256'].items():
        if hashlib.sha256((directory / 'source' / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Execution source snapshot mismatch: {label}/{name}')
    return dict(label=label, provenance=provenance, status=status)


def compare(stage):
    preserved, initialization, validation = validate_history()
    historical_provenance = read_json(ROOT / 'results/new-state/run-provenance.json')
    historical_config = read_json(ROOT / 'results/new-state/experiment.json')['config']
    benchmark_hashes = {}
    for circuit in historical_config['protocol']['circuits'].values():
        path = Path(circuit['file'])
        expected = historical_provenance['benchmark_hashes'][path.name]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Benchmark differs from the historical run: {path}')
        benchmark_hashes[str(path)] = actual
    plan = read_json(OUTPUT / 'analysis-plan.json')
    zero_root = OUTPUT / 'zero'
    if stage == 'first':
        zero_stage = validate_stage('stage1-zero')
        zero_config = zero_stage['provenance']['config']
        performance_root = ROOT / 'results/new-state'
        performance_config = read_json(performance_root / 'experiment.json')['config']
        performance_provenance = read_json(performance_root / 'run-provenance.json')
        historical_source = verify_recorded_source(ROOT, performance_provenance['source_hashes'])
        execution = [zero_stage]
        seeds = [0, 1, 2]
    else:
        first_report = json.loads(subprocess.check_output(
            ['git', 'show', 'HEAD:state-validation-first.json'], cwd=ROOT, text=True))
        if plan != first_report['analysis_plan']:
            raise ValueError('The analysis plan differs from the committed first-stage report.')
        zero_stage = validate_stage('stage2-zero')
        performance_stage = validate_stage('stage2-performance')
        zero_config = zero_stage['provenance']['config']
        performance_root = OUTPUT / 'performance'
        performance_config = performance_stage['provenance']['config']
        historical_source = verify_recorded_source(ROOT, read_json(ROOT / 'results/new-state/run-provenance.json')['source_hashes'])
        execution = [validate_stage('stage1-zero'), zero_stage, performance_stage]
        seeds = plan['fixed_seeds']
    for item in execution:
        for key in ['python', 'torch', 'numpy', 'yosys', 'abc']:
            if item['provenance'][key] != historical_provenance[key]:
                raise ValueError(f'Execution environment differs: {item["label"]}/{key}')
    method = dict(zero_config['method'])
    zero_mask = method.pop('performance_feature_mask', [1, 1])
    performance_method = dict(performance_config['method'])
    performance_mask = performance_method.pop('performance_feature_mask', [1, 1])
    if zero_mask != [0, 0] or performance_mask != [1, 1] or method != performance_method:
        raise ValueError('The methods differ beyond the prespecified feature mask.')
    historical_method = dict(historical_config['method'])
    historical_method.pop('performance_feature_mask', None)
    if method != historical_method or zero_config['environment'] != historical_config['environment']:
        raise ValueError('Training settings differ from the reused historical seeds.')
    for key, expected in historical_config['protocol'].items():
        if key != 'seeds' and zero_config['protocol'][key] != expected:
            raise ValueError(f'Protocol differs from the reused historical seeds: {key}')
    for key in ['protocol', 'environment']:
        if zero_config[key] != performance_config[key]:
            raise ValueError(f'Unequal experimental settings: {key}')
    for key in ['abc_binary', 'yosys_binary', 'workers']:
        if zero_config['runtime'][key] != performance_config['runtime'][key]:
            raise ValueError(f'Unequal runtime settings: {key}')
    protocol = zero_config['protocol']
    if protocol['seeds'] != seeds or protocol['episodes'] != 100 or protocol['iterations'] != 10:
        raise ValueError('The fixed seed set or training budget changed.')
    rows, summary = [], {}
    for circuit in protocol['circuits']:
        group = []
        for seed in seeds:
            zero = summarize_run(zero_root, circuit, seed, protocol)
            performance = summarize_run(performance_root, circuit, seed, protocol)
            comparable = zero['feasible'] and performance['feasible']
            gain = zero['luts'] - performance['luts'] if comparable else None
            row = dict(circuit=circuit, seed=seed, zero=zero, performance=performance, lut_reduction=gain,
                       outcome=('improved' if gain > 0 else ('tied' if gain == 0 else 'regressed')) if comparable else 'incomparable',
                       reused_performance_run=seed in [0, 1, 2])
            rows.append(row)
            group.append(row)
        entry = dict(required_seeds=len(seeds), wins=sum(r['outcome'] == 'improved' for r in group),
                     ties=sum(r['outcome'] == 'tied' for r in group), losses=sum(r['outcome'] == 'regressed' for r in group),
                     incomparable=sum(r['outcome'] == 'incomparable' for r in group))
        for variant in ['zero', 'performance']:
            values = [r[variant] for r in group]
            valid = all(v['feasible'] for v in values)
            entry[variant] = dict(feasible_seeds=sum(v['feasible'] for v in values),
                                 luts_mean=statistics.mean(v['luts'] for v in values) if valid else None,
                                 luts_std=statistics.pstdev(v['luts'] for v in values) if valid else None,
                                 levels_mean=statistics.mean(v['levels'] for v in values) if valid else None,
                                 training_seconds_mean=statistics.mean(v['training_seconds'] for v in values),
                                 feasible_episode_percent=statistics.mean(v['feasible_episode_percent'] for v in values))
        gains = [r['lut_reduction'] for r in group]
        valid = all(gain is not None for gain in gains)
        entry['paired_mean_reduction'] = statistics.mean(gains) if valid else None
        entry['paired_median_reduction'] = statistics.median(gains) if valid else None
        entry['improvement_percent'] = 100 * entry['paired_mean_reduction'] / entry['zero']['luts_mean'] if valid else None
        entry['intervals'] = paired_intervals(gains, plan) if valid and stage == 'ten' else None
        entry['supports_information_gain'] = bool(entry['intervals'] and entry['intervals']['familywise'][0] > 0)
        summary[circuit] = entry
    gate = stage == 'ten' and any(entry['supports_information_gain'] for entry in summary.values()) and all(
        entry['performance']['feasible_seeds'] >= entry['zero']['feasible_seeds'] for entry in summary.values())
    payload = dict(material_passport=dict(origin_skill='academic-research-suite/experiment-agent', origin_mode='validate',
                                         origin_date=datetime.now(timezone.utc).date().isoformat(),
                                         verification_status='VERIFIED', version_label=f'matched_state_{stage}_v1'),
                   stage=stage, seeds=seeds, zero_config=zero_config, performance_config=performance_config,
                   rows=rows, summary=summary, initialization=initialization, validation=validation,
                   analysis_plan=plan, execution=execution, historical_commit=preserved['commit'],
                   analysis_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   plan_matches_first_stage=stage == 'ten',
                   benchmark_sha256=benchmark_hashes,
                   analysis_environment=dict(platform=platform.platform(), machine=platform.machine()),
                   historical_source=historical_source, preserved_history_files=len(preserved['files']),
                   historical_files_unchanged=True, ablation_gate_met=gate)
    stem = ROOT / f'state-validation-{stage}'
    stem.with_suffix('.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    flat = []
    for row in rows:
        item = {key: row[key] for key in ['circuit', 'seed', 'lut_reduction', 'outcome', 'reused_performance_run']}
        for variant in ['zero', 'performance']:
            item.update({f'{variant}_{key}': value for key, value in row[variant].items() if key != 'sequence'})
        flat.append(item)
    with stem.with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(flat)
    title = '第一步：三种子十一维零特征对照' if stage == 'first' else '第二步：十种子同维度性能信息验证'
    lines = [f'# {title}', '', '## Material Passport', '',
             '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: validate',
             f'- Origin Date: {payload["material_passport"]["origin_date"]}', '- Verification Status: VERIFIED', '',
             '## 结果', '',
             '两组均为十一维输入、938 个参数，相同种子的初始权重完全一致。零特征组输入 `[原九维, 0, 0]`；性能特征组输入 `[原九维, LUT比例, 层数余量]`。',
             '正 LUT 减少量表示性能特征组更好。统计对象是每次训练期间找到的最佳可行映射。', '',
             '| 电路 | 零特征 LUT 均值 ± 标准差 | 性能特征 LUT 均值 ± 标准差 | 平均减少 LUT | 改善率 | 改善 / 持平 / 退步 |',
             '|---|---:|---:|---:|---:|---|']
    def value(number, precision=3):
        return '未汇总' if number is None else f'{number:.{precision}f}'
    for circuit, entry in summary.items():
        zero, performance = entry['zero'], entry['performance']
        lines.append(f'| {circuit} | {value(zero["luts_mean"])} ± {value(zero["luts_std"])} | '
                     f'{value(performance["luts_mean"])} ± {value(performance["luts_std"])} | '
                     f'{value(entry["paired_mean_reduction"])} | {value(entry["improvement_percent"])}% | '
                     f'{entry["wins"]} / {entry["ties"]} / {entry["losses"]} |')
    if stage == 'ten':
        lines += ['', '## 配对差值区间', '',
                  '按同一种子的两组结果计算“零特征 LUT − 性能特征 LUT”，对种子对整体重采样 100000 次，固定分析种子 20260926。',
                  '95% 区间用于描述；另给出对三个预定电路使用 Bonferroni 调整后的约 98.333% 单电路区间。两者都是小样本 percentile bootstrap 的近似区间。', '',
                  '| 电路 | 配对差值中位数 | 95% 区间 | 调整后区间 | 区间是否完全为正 |',
                  '|---|---:|---|---|---|']
        for circuit, entry in summary.items():
            intervals = entry['intervals']
            formatted = [f'[{bounds[0]:.3f}, {bounds[1]:.3f}]' for bounds in intervals.values()] if intervals else ['未汇总', '未汇总']
            lines.append(f'| {circuit} | {value(entry["paired_median_reduction"])} | {formatted[0]} | {formatted[1]} | '
                         f'{"是" if entry["supports_information_gain"] else "否"} |')
    lines += ['', '## 逐种子结果', '',
              '| 电路 | 种子 | 零特征 LUT / levels | 性能特征 LUT / levels | LUT 减少量 | 最佳位置 零特征 → 性能特征 |',
              '|---|---:|---|---|---:|---|']
    for row in rows:
        zero, performance = row['zero'], row['performance']
        lines.append(f'| {row["circuit"]} | {row["seed"]} | {zero["luts"]} / {zero["levels"]} | '
                     f'{performance["luts"]} / {performance["levels"]} | {row["lut_reduction"]} | '
                     f'{zero["episode"]}/{zero["iteration"]} → {performance["episode"]}/{performance["iteration"]} |')
    lines += ['', '## 可行性与耗时', '',
              '| 电路 | 可行种子 零特征 / 性能特征 | 可行回合比例 零特征 / 性能特征 | 平均训练秒数 零特征 / 性能特征 |',
              '|---|---|---|---|']
    for circuit, entry in summary.items():
        zero, performance = entry['zero'], entry['performance']
        lines.append(f'| {circuit} | {zero["feasible_seeds"]}/{len(seeds)} / {performance["feasible_seeds"]}/{len(seeds)} | '
                     f'{zero["feasible_episode_percent"]:.2f}% / {performance["feasible_episode_percent"]:.2f}% | '
                     f'{zero["training_seconds_mean"]:.3f} / {performance["training_seconds_mean"]:.3f} |')
    lines += ['', '可行回合指第 0–10 步至少出现一个达标候选；不表示整条轨迹或最终一步均可行。耗时仅作参考，性能特征组的种子 0/1/2 复用了之前的本地结果。', '',
              '## 核验与复现', '',
              '- 固定三个电路、种子集合及 100×10 预算；LUT6、奖励、动作集、隐藏层、归一化和优化器保持一致。',
              '- 十个种子的两组初始权重及历史十一维模型初始化完全一致；15 项状态、掩码及续训检查通过。',
              '- 配对分析方案固定在第一步提交中；十种子分析开始前核对方案一致，分析脚本哈希随 JSON 保存。',
              '- 三个电路输入的 SHA-256，以及 Python、PyTorch、NumPy、ABC 和 Yosys 版本，与历史性能特征运行一致。',
              '- 所有预定种子均完整纳入；每次训练最终网表经过指标复核和 CEC；报告逐回合复核最优解及保存位置。',
              f'- 历史 {len(preserved["files"])} 个结果文件与原配置哈希保持不变；本阶段执行代码快照、命令和进程状态保存于 results/state-validation/。',
              '- 只改变新增信息是否可见；输入维度和初始参数相同。零输入使对应连接的梯度为零，这是对照的预期行为。', '',
              '```bash', f'.tools/conda-env/bin/python -B run_state_validation.py {"first" if stage == "first" else "ten"}',
              f'.tools/conda-env/bin/python -B compare_state_validation.py {stage}', '```', '',
              '运行器不会覆盖已有阶段记录。重跑需创建新的实验输出位置；不可直接覆盖当前已完成结果。', '',
              '## 解释与下一步', '']
    if stage == 'first':
        lines += ['三个种子只作描述性对照，不用于判断统计显著性。无论本阶段均值方向如何，下一步都按预先固定的 0–9 十种子方案完成验证，避免根据中途结果选择样本数量。']
    elif gate:
        lines += ['至少一个电路的调整后差值区间完全为正，且最终可行种子数未下降，达到预先约定的单维消融条件。下一步可分别屏蔽 LUT 比例与层数余量，同时保持十一维网络。']
    else:
        lines += ['本阶段未达到预先约定的单维消融条件：当前数据不足以确认性能信息带来稳定收益。先保留这一结论，不继续增加单维实验或挑选种子追求正结果。']
    lines += ['', '区间跨零表示本次数据尚不能确定平均差值方向，不等于证明两种状态等效。即使区间完全为正，也仅支持这些电路、预算和种子分布下的初步结论。',
              '没有跨电路泛化或冻结模型独立测试；比较的是搜索期间最优可行结果。全部种子独立从头训练，第二阶段仅复用已完成的种子，不增加其预算。',
              '复用的三个性能特征种子在方案制定前已观察过；区间和消融门槛用于探索性判断，不作为完全独立的确认性检验。',
              'AI 协助实现、执行、核验与描述性统计。结果未通过超参数调整、追加种子或挑选最好结果来优化。', '']
    stem.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    for extension in ['csv', 'json']:
        (OUTPUT / f'comparison-{stage}.{extension}').write_bytes(stem.with_suffix(f'.{extension}').read_bytes())
    return summary, gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['first', 'ten'])
    args = parser.parse_args()
    summary, gate = compare(args.stage)
    print(json.dumps(dict(stage=args.stage, summary=summary, ablation_gate_met=gate), indent=2))


if __name__ == '__main__':
    main()
