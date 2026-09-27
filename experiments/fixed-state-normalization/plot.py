"""Standard ReportLab search charts; run with the bundled document Python."""
import csv
import hashlib
import json
from pathlib import Path
import sys

from reportlab.graphics import renderSVG,renderPDF
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing,String,Line
from reportlab.lib.colors import HexColor
import pymupdf as fitz

HERE=Path(__file__).resolve().parent
GROUPS=['old-trained','old-frozen','fixed-trained','fixed-frozen','uniform']
COLORS=[HexColor(c) for c in ['#2563eb','#94a3b8','#dc2626','#fb923c','#16a34a']]


def main():
    rows=list(csv.DictReader((HERE/'search-curves.csv').open()))
    drawing=Drawing(1160,390)
    drawing.add(String(30,365,'i2c: best feasible LUTs vs search candidates',fontName='Helvetica-Bold',fontSize=18))
    drawing.add(String(30,342,'100 episodes x 10 actions; seeds 0/1/2; initial mapping included at candidate 0',fontSize=10))
    for i,(group,color) in enumerate(zip(GROUPS,COLORS)):
        x=35+i*225
        drawing.add(Line(x,320,x+22,320,strokeColor=color,strokeWidth=2))
        drawing.add(String(x+30,316,group,fontSize=10))
    values=[int(r['luts']) for r in rows]
    low,high=min(values)-2,max(values)+2
    for i,seed in enumerate((0,1,2)):
        plot=LinePlot();plot.x=60+i*375;plot.y=65;plot.width=310;plot.height=210
        plot.data=[[(int(r['candidate']),int(r['luts'])) for r in rows if r['group']==g and int(r['seed'])==seed] for g in GROUPS]
        plot.xValueAxis.valueMin=0;plot.xValueAxis.valueMax=1000;plot.xValueAxis.valueSteps=[0,250,500,750,1000]
        plot.yValueAxis.valueMin=low;plot.yValueAxis.valueMax=high
        plot.yValueAxis.valueSteps=sorted(set(round(low+(high-low)*k/4) for k in range(5)))
        plot.yValueAxis.visibleGrid=True;plot.yValueAxis.gridStrokeColor=HexColor('#e5e7eb')
        for j,color in enumerate(COLORS):
            plot.lines[j].strokeColor=color;plot.lines[j].strokeWidth=1.5
            if 'frozen' in GROUPS[j]:plot.lines[j].strokeDashArray=[4,3]
        drawing.add(plot)
        drawing.add(String(plot.x,290,f'Seed {seed}',fontName='Helvetica-Bold',fontSize=12))
        drawing.add(String(plot.x+110,35,'Candidates',fontSize=10))
    renderSVG.drawToFile(drawing,str(HERE/'search-curves.svg'))
    with fitz.open(stream=renderPDF.drawToString(drawing),filetype='pdf') as document:
        document[0].get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(str(HERE/'search-curves.png'))
    paths=[HERE/'search-curves.csv',HERE/'search-curves.svg',HERE/'search-curves.png',Path(__file__)]
    (HERE/'plot-validation.json').write_text(json.dumps({'command':[sys.executable,'-B',str(Path(__file__))],
        'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2)+'\n')


if __name__=='__main__':main()
