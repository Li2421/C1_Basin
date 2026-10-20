"""Export recorded training curves; no rollout generation or task-label reads."""
from pathlib import Path
import json,hashlib
from html import escape
import numpy as np

HERE=Path(__file__).resolve().parent
COLORS=('#1768ac','#d95d39','#237b4b','#854d9b','#ad7c13','#5f6b72')


def panel(svg,xs,ys,x0,y0,title,xlabel,ylabel,reference=None):
    w,h=475,240;px0,py0=x0+70,y0+50
    lo,hi=float(min(ys)),float(max(ys));pad=max((hi-lo)*.08,1e-6);lo-=pad;hi+=pad
    xmax=float(max(xs));xmin=float(min(xs));span=max(xmax-xmin,1.)
    x=lambda v:px0+(v-xmin)/span*w;y=lambda v:py0+h-(v-lo)/(hi-lo)*h
    def text(xx,yy,value,size=12,extra=''):svg.append(f'<text x="{xx:.2f}" y="{yy:.2f}" font-size="{size}" {extra}>{escape(str(value))}</text>')
    text(x0+40,y0+20,title,16)
    for v in np.linspace(lo,hi,5):
        yy=y(v);svg.append(f'<line x1="{px0}" y1="{yy}" x2="{px0+w}" y2="{yy}" stroke="#ddd"/>');text(px0-9,yy+4,f'{v:.3g}',11,'text-anchor="end"')
    for v in np.linspace(xmin,xmax,5):text(x(v),py0+h+20,f'{v/1e6:.1f}M' if xmax>=1e6 else f'{v:.0f}',11,'text-anchor="middle"')
    if reference is not None:
        xx=x(reference);svg.append(f'<line x1="{xx}" y1="{py0}" x2="{xx}" y2="{py0+h}" stroke="#888" stroke-dasharray="4 4"/>')
    pts=' '.join(f'{x(a):.2f},{y(b):.2f}' for a,b in zip(xs,ys))
    svg.append(f'<polyline points="{pts}" fill="none" stroke="#1768ac" stroke-width="1.8"/>')
    text(px0+w/2,py0+h+46,xlabel,12,'text-anchor="middle"')
    text(x0+18,py0+h/2,ylabel,12,f'transform="rotate(-90 {x0+18} {py0+h/2})" text-anchor="middle"')


def figure(family,metric,ylabel):
    rows=[];files={}
    for i in range(6):
        name=f'{family}{i}';path=HERE/'policies'/name/'TRAINING_HISTORY.json'
        if not path.exists():continue
        receipt=path.parent/('TRAINING_COMPLETE.receipt.json' if family=='ring_ippo' else 'COMPLETE.receipt.json')
        for source,digest in json.loads(receipt.read_text())['files'].items():
            assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==digest
        history=json.loads(path.read_text());files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        step='robot_steps' if family=='ring_ippo' else 'step'
        if metric=='training_success_pct':
            values=[100*r['successes']/r['completed_episodes'] if r['completed_episodes'] else 0. for r in history]
        else:values=[r[metric] for r in history]
        selected=None if family=='ring_ippo' else min(history,key=lambda r:r['dev_loss'])['step']
        rows.append(dict(controller=name,x=[r[step] for r in history],y=values,selected_step=selected))
    if not rows:return
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1130" viewBox="0 0 1200 1130">',
        '<rect width="100%" height="100%" fill="white"/>','<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#17202a}</style>',
        f'<text x="30" y="30" font-size="20">{family}: recorded {escape(ylabel)}</text>']
    for i,row in enumerate(rows):panel(svg,row['x'],row['y'],600*(i%2),65+350*(i//2),
        row['controller']+(f' (selected step {row["selected_step"]})' if row['selected_step'] is not None else ''),
        'Robot steps' if family=='ring_ippo' else 'Optimizer steps',ylabel,4000000 if family=='ring_ippo' else row['selected_step'])
    svg.append('</svg>');folder=HERE/'figures';folder.mkdir(exist_ok=True)
    (folder/f'{family}_{metric}.svg').write_text('\n'.join(svg)+'\n')
    (folder/f'{family}_{metric}_data.json').write_text(json.dumps(dict(files=files,curves=rows,
        task_evaluation_labels_used=False,ppo_curriculum_boundary_robot_steps=4000000 if family=='ring_ippo' else None),indent=2)+'\n')


if __name__=='__main__':
    for family in ('chunking','recurrent','cvae','implicit'):figure(family,'dev_loss','Validation objective')
    figure('ring_ippo','mean_reward','Mean training reward')
    figure('ring_ippo','training_success_pct','Training episode success (%)')
    print('Training curves exported from retained histories only')
