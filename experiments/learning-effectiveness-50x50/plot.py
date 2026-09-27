"""ReportLab scientific curves, runnable without NumPy, Torch, or the ML environment."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import reportlab
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Line, String
from reportlab.lib.colors import HexColor
import pymupdf as fitz

HERE = Path(__file__).resolve().parent
GROUPS = ('trained', 'frozen', 'uniform')
COLORS = [HexColor('#2563eb'), HexColor('#6b7280'), HexColor('#ea580c')]
PREFIXES = (5, 10, 20, 30, 40, 50)


def table(name):
    with (HERE / name).open() as stream:
        return list(csv.DictReader(stream))


def legend(drawing, labels, x=570, y=823, spacing=160):
    for i, label in enumerate(labels):
        offset = x + i * spacing
        drawing.add(Line(offset, y, offset + 22, y, strokeColor=COLORS[i], strokeWidth=2))
        drawing.add(String(offset + 30, y - 4, label, fontSize=11))


def line_chart(drawing, series, x, y, width, height, bounds, xticks, ylabel):
    populated = [(i, values) for i, values in enumerate(series) if values]
    if not populated:
        drawing.add(String(x, y + height / 2, 'No completed evidence', fontSize=12, fillColor=HexColor('#6b7280')))
        return
    chart = LinePlot()
    chart.x, chart.y, chart.width, chart.height = x, y, width, height
    chart.data = [values for _, values in populated]
    chart.xValueAxis.valueMin, chart.xValueAxis.valueMax = xticks[0], xticks[-1]
    chart.xValueAxis.valueSteps = xticks
    low, high = bounds
    chart.yValueAxis.valueMin, chart.yValueAxis.valueMax = low, high
    chart.yValueAxis.valueSteps = sorted(set(round(low + (high - low) * i / 4, 2) for i in range(5)))
    chart.xValueAxis.labels.fontSize = chart.yValueAxis.labels.fontSize = 9
    chart.yValueAxis.visibleGrid = True
    chart.yValueAxis.gridStrokeColor = HexColor('#e5e7eb')
    for index, (color_index, _) in enumerate(populated):
        chart.lines[index].strokeColor, chart.lines[index].strokeWidth = COLORS[color_index], 1.5
    drawing.add(chart)
    drawing.add(String(x, y + height + 9, ylabel, fontSize=10, fillColor=HexColor('#475569')))


def save(drawing, label):
    svg, png = HERE / f'{label}.svg', HERE / f'{label}.png'
    renderSVG.drawToFile(drawing, str(svg))
    with fitz.open(stream=renderPDF.drawToString(drawing), filetype='pdf') as document:
        document[0].get_pixmap(matrix=fitz.Matrix(120 / 72, 120 / 72)).save(str(png))
    return [svg, png]


def search_curves(data, summary):
    budget = summary['protocol']['episodes'] * summary['protocol']['iterations']
    drawing = Drawing(1120, 850)
    drawing.add(String(35, 820, 'Best feasible LUTs vs search candidates', fontName='Helvetica-Bold', fontSize=19))
    drawing.add(String(35, 796, f'{summary["protocol"]["episodes"]} episodes x {summary["protocol"]["iterations"]} actions; '
        'same initialization, mapping and constraints', fontSize=11))
    legend(drawing, GROUPS)
    for row, name in enumerate(summary['protocol']['circuits']):
        all_values = [int(r['luts']) for r in data if r['circuit'] == name and r['feasible'] == 'True']
        bounds = (min(all_values) - 1, max(all_values) + 1) if all_values else (0, 1)
        for column, seed in enumerate(summary['protocol']['seeds']):
            x, y = 65 + column * 365, 570 - row * 240
            series = [[(int(r['candidate']), int(r['luts'])) for r in data if r['circuit'] == name
                       and r['seed'] == str(seed) and r['group'] == group and r['feasible'] == 'True'] for group in GROUPS]
            line_chart(drawing, series, x, y, 305, 165, bounds, [0, budget / 4, budget / 2, budget * 3 / 4, budget], 'LUTs (lower better)')
            drawing.add(String(x, y + 195, f'{name} / seed {seed}', fontName='Helvetica-Bold', fontSize=12))
            drawing.add(String(x + 90, y - 35, 'Action candidates', fontSize=10))
    drawing.add(String(35, 18, 'Step 0 is eligible but is not counted as an action. Infeasible candidates are retained in CSV, not plotted as LUTs.', fontSize=10))
    drawing.add(String(35, 5, 'Partial runs stop at the last committed candidate; their missing tail is not extrapolated.', fontSize=9))
    return save(drawing, 'search-curves')


def old_budget(data, summary):
    drawing = Drawing(1120, 375)
    drawing.add(String(35, 340, 'Original 100 x 10 experiment: common candidate budgets', fontName='Helvetica-Bold', fontSize=19))
    drawing.add(String(35, 316, 'All three seeds retained; use feasibility if any compared run is infeasible', fontSize=11))
    legend(drawing, GROUPS, x=590, y=302, spacing=160)
    budgets = sorted(set(int(r['budget']) for r in data))
    for column, name in enumerate(summary['protocol']['circuits']):
        records = [r for r in data if r['circuit'] == name]
        use_feasibility = any(r['status'] == 'complete' and r['best_feasible'] != 'True' for r in records)
        series = []
        for group in GROUPS:
            points = []
            for budget in budgets:
                bank = [r for r in records if r['group'] == group and int(r['budget']) == budget]
                if len(bank) != 3 or any(r['status'] != 'complete' for r in bank):
                    continue
                value = 100 * sum(r['best_feasible'] == 'True' for r in bank) / 3 if use_feasibility else sum(int(r['best_luts']) for r in bank) / 3
                points.append((budget, value))
            series.append(points)
        values = [value for points in series for _, value in points]
        bounds = (0, 100) if use_feasibility else ((min(values) - 1, max(values) + 1) if values else (0, 1))
        x = 65 + column * 365
        line_chart(drawing, series, x, 78, 305, 172, bounds, budgets,
                   'Feasible % (higher better)' if use_feasibility else 'Mean best LUTs (lower better)')
        drawing.add(String(x, 277, name, fontName='Helvetica-Bold', fontSize=12))
        drawing.add(String(x + 85, 41, 'Action candidates', fontSize=10))
    drawing.add(String(35, 15, 'This reuses old evidence; it does not rerun or change the original experiment.', fontSize=10))
    return save(drawing, 'old-budget')


def evaluation_prefixes(data, summary):
    drawing = Drawing(1120, 850)
    drawing.add(String(35, 820, 'Independent fixed-policy evaluation vs sequence budget', fontName='Helvetica-Bold', fontSize=19))
    drawing.add(String(35, 796, 'Nested prefixes of 30 held-out 50-action sequences; no learning during evaluation', fontSize=11))
    labels = ('trained-50', 'initial/frozen', 'uniform')
    legend(drawing, labels)
    for row, name in enumerate(summary['protocol']['circuits']):
        records = [r for r in data if r['circuit'] == name and r['policy'] in ('trained-50', 'initial', 'uniform')]
        use_feasibility = any(r['status'] == 'complete' and r['best_feasible'] != 'True' for r in records)
        circuits_series = []
        for seed in summary['protocol']['seeds']:
            series = []
            for policy in ('trained-50', 'initial', 'uniform'):
                points = []
                for prefix in PREFIXES:
                    bank = [r for r in records if r['training_seed'] == str(seed) and r['policy'] == policy and int(r['steps']) == prefix]
                    if len(bank) != 30 or any(r['status'] != 'complete' or r['task_status'] != 'complete' for r in bank):
                        continue
                    value = (100 * sum(r['best_feasible'] == 'True' for r in bank) / 30 if use_feasibility
                             else sum(int(r['best_luts']) for r in bank) / 30)
                    points.append((prefix, value))
                series.append(points)
            circuits_series.append(series)
        all_values = [value for series in circuits_series for points in series for _, value in points]
        bounds = (0, 100) if use_feasibility else ((min(all_values) - 0.5, max(all_values) + 0.5) if all_values else (0, 1))
        for column, (seed, series) in enumerate(zip(summary['protocol']['seeds'], circuits_series)):
            x, y = 65 + column * 365, 570 - row * 240
            line_chart(drawing, series, x, y, 305, 165, bounds, list(PREFIXES),
                       'Feasible % (higher better)' if use_feasibility else 'Mean best LUTs (lower better)')
            drawing.add(String(x, y + 195, f'{name} / training seed {seed}', fontName='Helvetica-Bold', fontSize=12))
            drawing.add(String(x + 70, y - 35, 'Actions per evaluation sequence', fontSize=10))
    drawing.add(String(35, 18, 'Uniform is one shared 30-sequence bank per circuit. Repeated columns and prefixes do not add independent samples.', fontSize=10))
    drawing.add(String(35, 5, 'Missing or failed banks are omitted; a missing policy is not plotted as 0% feasible.', fontSize=9))
    return save(drawing, 'evaluation-prefixes')


def main():
    summary = json.loads((HERE / 'summary.json').read_text())
    curves, budgets, evaluation = table('trajectories.csv'), table('old-budget.csv'), table('evaluation.csv')
    artifacts = search_curves(curves, summary) + old_budget(budgets, summary) + evaluation_prefixes(evaluation, summary)
    sources = [HERE / 'trajectories.csv', HERE / 'old-budget.csv', HERE / 'evaluation.csv', HERE / 'summary.json', HERE / 'plot.py']
    records = dict(command=[sys.executable, str(Path(__file__).resolve())], python=sys.version,
        platform=platform.platform(), reportlab=reportlab.Version, trajectory_rows=len(curves), old_budget_rows=len(budgets),
        evaluation_prefix_rows=len(evaluation), missing_evaluation_rows=sum(r['status'] != 'complete' for r in evaluation),
        uniform_shared=True, physical_rollouts=summary['physical_rollouts'], logical_rollouts=summary['logical_rollouts'],
        hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources + artifacts})
    (HERE / 'plot-validation.json').write_text(json.dumps(records, indent=2) + '\n')
    print('Charts: ' + ', '.join(p.name for p in artifacts))


if __name__ == '__main__':
    main()
