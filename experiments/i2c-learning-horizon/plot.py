"""Plot audited best-LUT trajectories with ReportLab, independently of training."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import pymupdf
import reportlab
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Line, String
from reportlab.lib import colors

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    validation = json.loads((HERE / 'validation.json').read_text())
    data = HERE / 'trajectories.csv.gz'
    if sha256(data) != validation['trajectories_sha256']:
        raise ValueError('Trajectory data differs from validated results.')
    with gzip.open(data, 'rt', newline='') as stream:
        rows = list(csv.DictReader(stream))
    groups = ['trained-30', 'frozen-30', 'trained-10', 'frozen-10']
    palette = [colors.HexColor('#2563eb'), colors.HexColor('#d97706')] * 2
    minimum = math.floor(min(int(r['running_best_feasible_luts']) for r in rows) / 5) * 5 - 5
    maximum = math.ceil(max(int(r['running_best_feasible_luts']) for r in rows) / 5) * 5 + 5
    drawing = Drawing(1030, 375)
    drawing.add(String(35, 350, 'Best feasible LUTs during search', fontName='Helvetica-Bold', fontSize=17))
    drawing.add(String(35, 331, 'Original table rewards | standardize | 100 episodes | seeds 0/1/2', fontSize=11))
    for i, group in enumerate(groups):
        x = 50 + i * 240
        drawing.add(Line(x, 305, x + 28, 305, strokeColor=palette[i], strokeWidth=2,
                         strokeDashArray=[4, 3] if i >= 2 else None))
        drawing.add(String(x + 36, 301, group, fontSize=11))
    for seed in [0, 1, 2]:
        chart = LinePlot()
        chart.x, chart.y, chart.width, chart.height = 55 + seed * 330, 65, 280, 205
        series = []
        for group in groups:
            selected = [r for r in rows if r['group'] == group and int(r['seed']) == seed]
            previous = int(selected[0]['running_best_feasible_luts'])
            points = [(int(selected[0]['actions_completed']), previous)]
            for row in selected[1:]:
                value, action = int(row['running_best_feasible_luts']), int(row['actions_completed'])
                if value != previous:
                    points.extend([(action, previous), (action, value)])
                    previous = value
            points.append((int(selected[-1]['actions_completed']), previous))
            series.append(points)
        chart.data = series
        chart.xValueAxis.valueMin, chart.xValueAxis.valueMax = 0, 3000
        chart.xValueAxis.valueSteps = [0, 1000, 2000, 3000]
        chart.yValueAxis.valueMin, chart.yValueAxis.valueMax = minimum, maximum
        chart.yValueAxis.valueSteps = list(range(minimum, maximum + 1, 10))
        for axis in [chart.xValueAxis, chart.yValueAxis]:
            axis.labels.fontSize = 9
            axis.visibleGrid = True
            axis.gridStrokeColor = colors.HexColor('#e5e7eb')
        for i in range(4):
            chart.lines[i].strokeColor = palette[i]
            chart.lines[i].strokeWidth = 1.6
            if i >= 2:
                chart.lines[i].strokeDashArray = [4, 3]
        drawing.add(chart)
        drawing.add(String(chart.x + 115, 283, f'Seed {seed}', fontName='Helvetica-Bold', fontSize=12))
        drawing.add(String(chart.x + 75, 30, 'Search actions completed', fontSize=10))
    drawing.add(String(35, 10, '10-step runs stop at 1,000 actions; 30-step runs stop at 3,000. Lower LUTs are better.', fontSize=10))
    svg, png = HERE / 'best-trajectories.svg', HERE / 'best-trajectories.png'
    renderSVG.drawToFile(drawing, str(svg))
    with pymupdf.open(stream=renderPDF.drawToString(drawing), filetype='pdf') as document:
        document[0].get_pixmap(dpi=150).save(png)
    record = dict(command=[sys.executable, '-B', str(Path(__file__).relative_to(HERE.parents[1]))],
        reportlab=reportlab.Version, pymupdf=pymupdf.VersionBind,
        source_sha256=sha256(Path(__file__)), data_sha256=sha256(data),
        outputs={p.name: sha256(p) for p in [svg, png]})
    (HERE / 'plot-validation.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Saved validated SVG and PNG trajectories.')


if __name__ == '__main__':
    main()
