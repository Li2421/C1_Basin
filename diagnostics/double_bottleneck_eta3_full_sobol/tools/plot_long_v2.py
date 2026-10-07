#!/usr/bin/env python3
"""Dependency-free deterministic R1-R4 figures for long v2."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[3]; STUDY=ROOT/"diagnostics/double_bottleneck_eta3_full_sobol"; LONG=STUDY/"long_run_v2"; FIG=LONG/"figures"
LOW=np.asarray((.5,-.5,0.)); HIGH=np.asarray((1.25,.5,.75)); WIDTH=HIGH-LOW


def start(w,h,title): return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>',f'<text x="{w/2}" y="27" text-anchor="middle" font-family="sans-serif" font-size="17" font-weight="600">{escape(title)}</text>']


def main():
    FIG.mkdir(exist_ok=True); sel=json.loads((LONG/"representative_selection.json").read_text()); data=json.loads((LONG/"timeout_basin_matrix.json").read_text()); rows={r["episode_id"]:r for r in data["episodes"]}; theta=np.asarray([p["theta"] for p in data["eta_points"]]); unit=(theta-LOW)/WIDTH; roles=("R1","R2","R3","R4")
    # Three coordinate projections for each deterministic representative.
    w,h=1050,800; parts=start(w,h,"R1–R4 P0-3D basin projections on the common 256 points"); projections=((0,1,"eta1","eta2"),(0,2,"eta1","eta3"),(1,2,"eta2","eta3"))
    for ri,role in enumerate(roles):
        eid=sel[role]; mask=np.zeros(256,dtype=bool); mask[rows[eid]["successful_eta_indices"]]=True; parts.append(f'<text x="16" y="{105+ri*170}" font-family="sans-serif" font-size="10" font-weight="600">{role}: {escape(eid)}</text>')
        for ci,(a,b,xl,yl) in enumerate(projections):
            x0,y0,pw,ph=285+ci*245,58+ri*175,205,135; parts.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#bbb"/>')
            for point,success in zip(unit,mask):
                x=x0+point[a]*pw; y=y0+(1-point[b])*ph; parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{3.2 if success else 1.45}" fill="{"#d7301f" if success else "#9ecae1"}" fill-opacity="{.9 if success else .5}"/>')
            parts += [f'<text x="{x0+pw/2}" y="{y0+ph+13}" text-anchor="middle" font-family="sans-serif" font-size="9">{xl}</text>',f'<text x="{x0+4}" y="{y0+11}" font-family="sans-serif" font-size="9">{yl}</text>']
    parts.append('</svg>'); (FIG/"R1_R4_eta_2d_projections.svg").write_text("\n".join(parts)+"\n")
    # Fixed isometric 3D view.
    w,h=1000,650; parts=start(w,h,"R1–R4 P0-3D success/failure scatter")
    for idx,role in enumerate(roles):
        eid=sel[role]; mask=np.zeros(256,dtype=bool); mask[rows[eid]["successful_eta_indices"]]=True; col,row=idx%2,idx//2; x0,y0,pw,ph=60+col*485,55+row*285,410,220; parts += [f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#bbb"/>',f'<text x="{x0+7}" y="{y0+16}" font-family="sans-serif" font-size="10" font-weight="600">{role}: {escape(eid)}</text>']
        for i in np.argsort(unit[:,2]):
            a,b,c=unit[i]; x=x0+40+(.72*a+.28*c)*(pw-80); y=y0+30+(1-(.72*b+.28*c))*(ph-60); success=mask[i]; parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{3.3 if success else 1.5}" fill="{"#d7301f" if success else "#9ecae1"}" fill-opacity="{.9 if success else .48}"/>')
    parts.append('</svg>'); (FIG/"R1_R4_eta_3d_scatter.svg").write_text("\n".join(parts)+"\n")
    # Local fixed-radius samples around R1-R3 representatives.
    local=[]
    for path in sorted((LONG/"raw").glob("local_shard*.jsonl")): local.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    w,h=930,340; parts=start(w,h,"Fixed 32-point local robustness around R1–R3 eta representatives")
    for ci,role in enumerate(("R1","R2","R3")):
        eid=sel[role]; subset=[r for r in local if r["episode_id"]==eid]; center=np.asarray(rows[eid]["eta_rep"]["theta"]); cu=(center-LOW)/WIDTH; x0,y0,pw,ph=55+ci*300,60,250,220; parts += [f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#bbb"/>',f'<text x="{x0+pw/2}" y="{y0+16}" text-anchor="middle" font-family="sans-serif" font-size="10">{role}: {escape(eid)}</text>']
        for r in subset:
            u=(np.asarray(r["theta"])-LOW)/WIDTH; dx,dy=(u[0]-cu[0])/.05,(u[1]-cu[1])/.05; x=x0+pw/2+dx*pw*.38; y=y0+ph/2-dy*ph*.38; parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{"#238b45" if r["success"] else "#cb181d"}"/>')
        parts.append(f'<circle cx="{x0+pw/2}" cy="{y0+ph/2}" r="5" fill="#2171b5"/>')
    parts.append('</svg>'); (FIG/"R1_R3_local_robustness.svg").write_text("\n".join(parts)+"\n")
    print(json.dumps({"figures":3},sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
