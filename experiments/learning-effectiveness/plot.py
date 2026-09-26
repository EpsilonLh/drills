"""Standalone ReportLab charts, using the bundled document runtime (no ML dependencies)."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import reportlab
from reportlab.graphics import renderSVG, renderPDF
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String, Line
from reportlab.lib.colors import HexColor
import pymupdf as fitz

HERE = Path(__file__).resolve().parent
COLORS = [HexColor('#2563eb'), HexColor('#6b7280'), HexColor('#ea580c')]


def main():
    data = list(csv.DictReader((HERE / 'trajectories.csv').open()))
    drawing = Drawing(1120, 850)
    drawing.add(String(35, 820, 'Best feasible LUTs vs search candidates', fontName='Helvetica-Bold', fontSize=19))
    drawing.add(String(35, 796, '100 episodes x 10 actions; same initial mapping and constraints', fontSize=11))
    for i, group in enumerate(('trained', 'frozen', 'uniform')):
        x = 500 + i * 165
        drawing.add(Line(x, 823, x + 22, 823, strokeColor=COLORS[i], strokeWidth=2))
        drawing.add(String(x + 30, 819, group, fontSize=12))
    for row, name in enumerate(('int2float', 'i2c', 'max')):
        all_values = [int(r['luts']) for r in data if r['circuit'] == name and r['feasible'] == 'True']
        if not all_values:
            continue
        low, high = min(all_values) - 1, max(all_values) + 1
        for column, seed in enumerate((0, 1, 2)):
            x, y = 65 + column * 365, 570 - row * 240
            plot = LinePlot()
            plot.x, plot.y, plot.width, plot.height = x, y, 305, 175
            plot.data = [[(int(r['candidate']), int(r['luts'])) for r in data
                          if r['circuit'] == name and r['seed'] == str(seed) and r['group'] == group and r['feasible'] == 'True']
                         for group in ('trained', 'frozen', 'uniform')]
            plot.xValueAxis.valueMin, plot.xValueAxis.valueMax = 0, 1000
            plot.xValueAxis.valueSteps = [0, 250, 500, 750, 1000]
            plot.yValueAxis.valueMin, plot.yValueAxis.valueMax = low, high
            plot.yValueAxis.valueSteps = sorted(set(round(low + (high - low) * k / 4) for k in range(5)))
            plot.xValueAxis.labels.fontSize = plot.yValueAxis.labels.fontSize = 9
            plot.yValueAxis.visibleGrid = True
            plot.yValueAxis.gridStrokeColor = HexColor('#e5e7eb')
            for index, color in enumerate(COLORS):
                plot.lines[index].strokeColor, plot.lines[index].strokeWidth = color, 1.4
            drawing.add(plot)
            drawing.add(String(x, y + 192, f'{name} / seed {seed}', fontName='Helvetica-Bold', fontSize=12))
            drawing.add(String(x + 90, y - 35, 'Candidates', fontSize=10))
    drawing.add(String(35, 18, 'Step 0 is eligible, but excluded from the action-candidate count. Infeasible prefixes are omitted.', fontSize=10))
    svg = HERE / 'search-curves.svg'
    renderSVG.drawToFile(drawing, str(svg))
    png = HERE / 'search-curves.png'
    with fitz.open(stream=renderPDF.drawToString(drawing), filetype='pdf') as document:
        document[0].get_pixmap(matrix=fitz.Matrix(120 / 72, 120 / 72)).save(str(png))
    evaluation = list(csv.DictReader((HERE / 'evaluation.csv').open()))
    policy_chart = Drawing(1120, 360)
    policy_chart.add(String(35, 330, 'Independent no-update policy evaluation', fontName='Helvetica-Bold', fontSize=19))
    policy_chart.add(String(35, 307, '30 fresh sampling seeds per model; LUT mean only when all rollouts are feasible', fontSize=11))
    palette = [HexColor('#2563eb'), HexColor('#0f766e'), HexColor('#9333ea'), COLORS[2]]
    for i, label in enumerate(('seed 0', 'seed 1', 'seed 2', 'uniform')):
        x = 570 + i * 125
        policy_chart.add(Line(x, 333, x + 18, 333, strokeColor=palette[i], strokeWidth=2))
        policy_chart.add(String(x + 23, 329, label, fontSize=11))
    for column, name in enumerate(('int2float', 'i2c', 'max')):
        chart = LinePlot()
        chart.x, chart.y, chart.width, chart.height = 65 + column * 365, 70, 305, 195
        use_feasibility = any(r['best_feasible'] != 'True' for r in evaluation if r['circuit'] == name)
        def value(rows):
            if len(rows) != 30:
                raise ValueError('Plot requires complete evaluation.')
            return (100 * sum(r['best_feasible'] == 'True' for r in rows) / len(rows) if use_feasibility else
                    sum(int(r['best_luts']) for r in rows) / len(rows))
        series = []
        for seed in (0, 1, 2):
            points = []
            for episode, policy in [(0, 'initial'), (50, 'trained-50'), (100, 'trained-100')]:
                rows = [r for r in evaluation if r['circuit'] == name and r['policy'] == policy and r['training_seed'] == str(seed)]
                points.append((episode, value(rows)))
            series.append(points)
        rows = [r for r in evaluation if r['circuit'] == name and r['policy'] == 'uniform' and r['training_seed'] == '0']
        uniform = value(rows)
        series.append([(0, uniform), (100, uniform)])
        chart.data = series
        chart.xValueAxis.valueMin, chart.xValueAxis.valueMax = 0, 100
        chart.xValueAxis.valueSteps = [0, 50, 100]
        values = [value for points in series for _, value in points]
        low, high = (0, 100) if use_feasibility else (min(values) - 0.5, max(values) + 0.5)
        chart.yValueAxis.valueMin, chart.yValueAxis.valueMax = low, high
        chart.yValueAxis.valueSteps = [round(low + (high - low) * i / 4, 2) for i in range(5)]
        chart.xValueAxis.labels.fontSize = chart.yValueAxis.labels.fontSize = 9
        chart.yValueAxis.visibleGrid = True
        chart.yValueAxis.gridStrokeColor = HexColor('#e5e7eb')
        for i, color in enumerate(palette):
            chart.lines[i].strokeColor, chart.lines[i].strokeWidth = color, 1.6
        chart.lines[3].strokeDashArray = [4, 3]
        policy_chart.add(chart)
        label = name + (' / feasible % (higher better)' if use_feasibility else ' / LUTs (lower better)')
        policy_chart.add(String(chart.x, 280, label, fontName='Helvetica-Bold', fontSize=11))
        policy_chart.add(String(chart.x + 65, 35, 'Training episodes completed', fontSize=10))
    evaluation_svg, evaluation_png = HERE / 'policy-evaluation.svg', HERE / 'policy-evaluation.png'
    renderSVG.drawToFile(policy_chart, str(evaluation_svg))
    with fitz.open(stream=renderPDF.drawToString(policy_chart), filetype='pdf') as document:
        document[0].get_pixmap(matrix=fitz.Matrix(120 / 72, 120 / 72)).save(str(evaluation_png))
    records = dict(command=[sys.executable, str(Path(__file__).resolve())], python=sys.version,
                   platform=platform.platform(), reportlab=reportlab.Version, rows=len(data),
                   evaluation_rows=len(evaluation),
                   hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [HERE / 'trajectories.csv', HERE / 'evaluation.csv', svg, png, evaluation_svg, evaluation_png]})
    (HERE / 'plot-validation.json').write_text(json.dumps(records, indent=2) + '\n')
    print(svg)


if __name__ == '__main__':
    main()
