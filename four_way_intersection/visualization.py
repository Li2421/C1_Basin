"""Dependency-free SVG visualisations for the open intersection."""
from __future__ import annotations
from pathlib import Path
from html import escape
import numpy as np
from .scenario import AGENT_NAMES

COLORS=("#0072B2", "#D55E00", "#009E73", "#CC79A7")

def save_trajectory_svg(env, positions, output_path, *, title="Four-Way Intersection", terminal_reason="", order_signature=""):
    p=np.asarray(positions,dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (4,2): raise ValueError("positions must be [T,4,2]")
    W=H=640.; m=45.; h=env.config.world_half_extent
    def xy(v): return m+(v[0]+h)/(2*h)*(W-2*m), H-m-(v[1]+h)/(2*h)*(H-2*m)
    def pts(a): return " ".join(f"{xy(x)[0]:.2f},{xy(x)[1]:.2f}" for x in a)
    cxy=xy((0,0)); r=.7/(2*h)*(W-2*m)
    body=['<rect width="100%" height="100%" fill="white"/>',f'<rect x="{m}" y="{m}" width="{W-2*m}" height="{H-2*m}" fill="#fbfbfb" stroke="#222" stroke-width="3"/>',f'<circle cx="{cxy[0]:.1f}" cy="{cxy[1]:.1f}" r="{r:.1f}" fill="#eef6dd" stroke="#aaa" stroke-dasharray="4 4"/>']
    for i,(name,color) in enumerate(zip(AGENT_NAMES,COLORS)):
        body.append(f'<polyline points="{pts(p[:,i])}" fill="none" stroke="{color}" stroke-width="3"/>')
        sx,sy=xy(p[0,i]);gx,gy=xy(env.goals[i]); body.extend((f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="7" fill="white" stroke="{color}" stroke-width="2"/>',f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="7" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="3 2"/>',f'<text x="{sx+8:.1f}" y="{sy-8:.1f}" fill="{color}" font-size="14">{name}</text>'))
    body += [f'<text x="{m}" y="24" font-size="18" font-weight="bold">{escape(title)}</text>',f'<text x="{m}" y="{H-14}" font-family="monospace" font-size="13">terminal={escape(str(terminal_reason))} | crossing={escape(str(order_signature))}</text>']
    out=Path(output_path);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'+''.join(body)+'</svg>\n',encoding='utf-8');return out
