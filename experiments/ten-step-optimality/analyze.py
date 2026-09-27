"""Check observed state/transition consistency before considering structural deduplication."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def main():
    summary = json.loads((HERE / 'summary.json').read_text())
    if summary['status'] != 'complete' or not summary['complete_sequence_coverage']:
        raise ValueError('Only complete exact enumeration supports this analysis.')
    rows = list(csv.DictReader((HERE / 'candidates.csv').open()))
    if len(rows) != summary['expected_candidates']:
        raise ValueError('Candidate count changed.')
    by_sequence = {}
    for row in rows:
        row['sequence'] = tuple(json.loads(row['actions']))
        row['metrics'] = (int(row['luts']), int(row['levels']), row['mapped_structural_sha256'])
        key = (row['circuit'], row['sequence'])
        if key in by_sequence:
            raise ValueError('Duplicate action sequence.')
        by_sequence[key] = row
    result = []
    for name in summary['experiment']['protocol']['circuits']:
        outputs, transitions = collections.defaultdict(set), collections.defaultdict(set)
        for row in rows:
            if row['circuit'] != name:
                continue
            outputs[row['structural_sha256']].add(row['metrics'])
            sequence = row['sequence']
            if sequence:
                parent = by_sequence[(name, sequence[:-1])]
                transitions[(parent['structural_sha256'], sequence[-1])].add(
                    (row['structural_sha256'], *row['metrics']))
        changed_outputs = {key: sorted(value) for key, value in outputs.items() if len(value) > 1}
        changed_transitions = {str(key): sorted(value) for key, value in transitions.items() if len(value) > 1}
        result.append(dict(circuit=name, observed_structures=len(outputs),
            structures_with_different_mapped_outputs=len(changed_outputs),
            observed_structure_action_pairs=len(transitions),
            structure_action_pairs_with_different_successors=len(changed_transitions),
            different_outputs=changed_outputs, different_transitions=changed_transitions))
    metadata = dict(command=[sys.executable, '-B', str(Path(__file__).resolve())],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        candidates_sha256=hashlib.sha256((HERE / 'candidates.csv').read_bytes()).hexdigest(),
        inference='Only observed 0–4-step transitions are checked; this does not certify future state merging.',
        checks=result)
    (HERE / 'transition-checks.json').write_text(json.dumps(metadata, indent=2) + '\n')
    report = (HERE / 'report.md').read_text().split('\n## 重复结构的行为核对')[0]
    if '## Material Passport' not in report:
        passport = ('## Material Passport\n\n- Origin Skill: academic-research-suite / experiment-agent\n'
                    '- Origin Mode: run + validate\n- Origin Date: 2026-09-27\n'
                    '- Verification Status: VERIFIED（仅0–4步完整枚举；10步及全局最优未验证）\n'
                    '- Version Label: bounded_optimality_v1\n\n')
        title, body = report.split('\n\n', 1)
        report = title + '\n\n' + passport + body
    lines = ['', '## 重复结构的行为核对', '',
        '| 电路 | 同结构但映射结果不同的结构数 | 同结构+动作但后继不同的数量 |', '|---|---:|---:|']
    for row in result:
        lines.append(f'| {row["circuit"]} | {row["structures_with_different_mapped_outputs"]} | '
                     f'{row["structure_action_pairs_with_different_successors"]} |')
    lines += ['', '第一项比较映射 LUT、层数及映射网表；第二项比较执行同一动作后的网表结构和映射结果。'
              '若出现不一致，仅凭该结构哈希合并状态就可能漏解。即使全部一致，也只是已枚举层的经验核对，'
              '没有证明 5–10 步下工具内部状态和后续行为完全由这个哈希决定。', '',
              '当前 LUT 较差或当前层数超限也不能直接剪枝，后续优化动作可能使其改善。'
              '基于这些条件删除分支会失去精确最优性保证。', '']
    original = json.loads((HERE.parent / 'learning-effectiveness/summary.json').read_text())['training']
    trained_max = min(r['best']['luts'] for r in original if r['circuit'] == 'max' and r['group'] == 'trained')
    frozen_max = min(r['best']['luts'] for r in original if r['circuit'] == 'max' and r['group'] == 'frozen')
    final = [r for r in summary['results'] if r['maximum_depth'] == summary['max_depth']]
    validation = json.loads((HERE / 'reference-validation.json').read_text())
    if not validation['all_matched']:
        raise ValueError('Reference validation is incomplete.')
    lines += ['', '## 对学习有效性问题的解释', '',
        f'现有训练结果不能直接称为10步最优。max 的训练组最好为{trained_max} LUT，冻结组已找到'
        f'同预算内的{frozen_max} LUT，至少存在{trained_max - frozen_max} LUT的已知改善空间。'
        f'冻结组的{frozen_max} LUT自身是否接近10步极限，尚未证明。', '',
        '4步精确最优44/312/781均高于已有10步最好值43/300/763，说明更深的动作组合能改善结果。'
        '这不说明10步还一定存在巨大空间；目前没有10步最优的有效下界，无法判断距离极限有多近。', '',
        '0–4步累计不同结构为' + '、'.join(str(r['distinct_structures']) for r in final) + '，最后一层仍新增' +
        '、'.join(str(r['new_structures']) for r in final) + '个结构。重复结构可以减少计算，但目前未获得'
        '可证明安全的完整状态合并方法，不能直接据此声称已覆盖10步。', '',
        '所有10步序列的枚举若完成，证明的也是固定起点、7个动作、ABC映射器及约束下的最优。'
        '它不覆盖所有等价电路结构与所有LUT映射；ABC面积优化包含启发式过程，参见 '
        '[ABC官方文档](https://people.eecs.berkeley.edu/~alanmi/abc/abc.htm)。', '',
        f'本次完整枚举耗时{summary["experiment"]["wall_seconds"]:.1f}秒，3个workers；无剪枝、无失败、无超时。'
        f'{validation["comparisons"]}次原工具路径对照与{validation["recorded_tool_commands_verified"]}条工具命令/指标复核通过，'
        '最佳解6个网表通过CEC。独立核验脚本首次从JSON读取奖励表时发生整数键类型错误，已修复为原YAML读取；'
        '失败原因和后续成功记录保存于validation-attempts.json，此错误不影响只使用协议与ABC的枚举结果。', '',
        '报告重建：先运行 enumerate.py（已有输出使用 --resume），再运行 check.py 和 analyze.py。'
        '代码、数据、网表、工具命令与证据压缩包的哈希分别见 provenance.json、reference-validation.json、'
        'transition-checks.json、evidence-hashes.json 和 artifact-manifest.json。', '']
    (HERE / 'report.md').write_text(report + '\n'.join(lines))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
