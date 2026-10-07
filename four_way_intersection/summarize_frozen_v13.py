"""Analysis-only summary and failure-location SVG from a completed frozen run.

It consumes the already emitted diagnostics JSON rather than reopening any
test archive, and performs no training, collection, or policy selection.
"""
from __future__ import annotations
import argparse, collections, json
from html import escape
from pathlib import Path
import numpy as np
from .environment import Config
from .scenario import AGENT_NAMES
from .visualization import COLORS


def _failure_svg(rows, output):
    cfg=Config(); W=H=640.; m=45.; h=cfg.world_half_extent
    def xy(point): return m+(point[0]+h)/(2*h)*(W-2*m), H-m-(point[1]+h)/(2*h)*(H-2*m)
    center=xy((0.,0.)); r=.7/(2*h)*(W-2*m)
    body=['<rect width="100%" height="100%" fill="white"/>',f'<rect x="{m}" y="{m}" width="{W-2*m}" height="{H-2*m}" fill="#fbfbfb" stroke="#222" stroke-width="3"/>',f'<circle cx="{center[0]:.1f}" cy="{center[1]:.1f}" r="{r:.1f}" fill="#eef6dd" stroke="#aaa" stroke-dasharray="4 4"/>']
    failures=[row for row in rows if not row['success']]
    for row in failures:
        symbol='×' if row['termination']=='collision' else '•'
        for agent,point in enumerate(row['failure_location']):
            x,y=xy(point);body.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="{COLORS[agent]}" font-size="16" text-anchor="middle" dominant-baseline="middle">{symbol}</text>')
    legend='; '.join(f'{name}={color}' for name,color in zip(AGENT_NAMES,COLORS))
    body += [f'<text x="{m}" y="24" font-size="18" font-weight="bold">Frozen v13 final locations: timeout •, collision ×</text>',f'<text x="{m}" y="{H-14}" font-family="monospace" font-size="12">{escape(legend)}; {len(failures)} non-success episodes</text>']
    Path(output).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'+''.join(body)+'</svg>\n')


def run(summary, output):
    summary, output=Path(summary),Path(output)
    if output.exists(): raise FileExistsError(output)
    payload=json.loads(summary.read_text());rows=payload['rollouts']
    overall=collections.Counter(row['mode_signature'] for row in rows)
    by_terminal={terminal:collections.Counter(row['mode_signature'] for row in rows if row['termination']==terminal) for terminal in ('success','timeout','collision')}
    output.mkdir(parents=True)
    report={'source_summary':str(summary),'analysis_only':True,'test_archives_reopened':False,'rollouts':len(rows),
            'mode_counts':dict(sorted(overall.items())),'mode_counts_by_terminal':{key:dict(sorted(value.items())) for key,value in by_terminal.items()},
            'terminal_counts':dict(collections.Counter(row['termination'] for row in rows)),
            'distinct_modes':len(overall),'success_distinct_modes':len(by_terminal['success'])}
    (output/'crossing_order_statistics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    _failure_svg(rows,output/'frozen_failure_locations.svg')
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('summary');p.add_argument('output');a=p.parse_args();print(json.dumps(run(a.summary,a.output),indent=2))
