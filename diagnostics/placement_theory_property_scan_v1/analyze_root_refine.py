"""Uniform narrow three-point root estimate for state-perturbed Ring slices."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent

def stem(uid,chain,h,sign,offset):
    return f'{uid}_{chain}_h{h:g}_s{sign}_o{offset:+g}'

def main():
    p=json.loads((HERE/'protocol.json').read_text());rows=[]
    for uid in ('hard_confirm_ring_1_920004','hard_confirm_ring_1_920005'):
        for chain in ('TT','FF'):
            for h in (0.,.001,.003,.01):
                for sign in ((0,) if h==0 else (-1,1)):
                    vals=[];term=[];active=[]
                    for offset in (-1e-5,0.,1e-5):
                        x=json.loads((HERE/'rollout_results'/(stem(uid,chain,h,sign,offset)+'.json')).read_text())
                        vals.append(x['rho']);term.append(x['termination']);active.append(x['active_margin'])
                    left,center,right=vals
                    if left is None or center is None or right is None:root=None
                    elif left*center<=0 and left!=center:root=-1e-5-left*1e-5/(center-left)
                    elif center*right<=0 and right!=center:root=-center*1e-5/(right-center)
                    else:root=None
                    rows.append(dict(uid=uid,chain=chain,radius=h,sign=sign,rho=vals,termination=term,
                        active_margin=active,active_same=all(x==active[0] for x in active),
                        bracketed=root is not None,root_offset=root,
                        local_slope=(right-left)/(2e-5) if left is not None and right is not None else None))
    for r in rows:
        base=next(x for x in rows if x['uid']==r['uid'] and x['chain']==r['chain'] and x['radius']==0)
        r['root_shift']=r['root_offset']-base['root_offset'] if r['bracketed'] and base['bracketed'] else None
        r['root_quotient']=abs(r['root_shift'])/r['radius'] if r['root_shift'] is not None and r['radius'] else None
    stats=dict(total=len(rows),bracketed=sum(r['bracketed'] for r in rows),
        active_same=sum(r['active_same'] for r in rows),
        base_root_offset=[r['root_offset'] for r in rows if r['radius']==0],
        by_radius={str(h):dict(bracketed=sum(r['bracketed'] for r in rows if r['radius']==h),
            max_abs_shift=max([abs(r['root_shift']) for r in rows if r['radius']==h and r['root_shift'] is not None],default=None),
            median_quotient=float(np.median([r['root_quotient'] for r in rows if r['radius']==h and r['root_quotient'] is not None]))
                if any(r['radius']==h and r['root_quotient'] is not None for r in rows) else None)
            for h in (.001,.003,.01)},
        note='Root is linearly interpolated only inside a 2e-5 wide bracket; no root asserted outside it.')
    (HERE/'state_boundary_refined_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (HERE/'state_boundary_refined_summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(5.2,3.5))
    for chain,color in (('TT','#1f77b4'),('FF','#d95f02')):
        rr=[r for r in rows if r['chain']==chain and r['radius'] and r['root_shift'] is not None]
        ax.scatter([r['radius'] for r in rr],[abs(r['root_shift']) for r in rr],label=chain,color=color)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('$||\\Delta s_0||$');ax.set_ylabel('refined boundary shift');ax.legend()
    fig.tight_layout();fig.savefig(HERE/'figures/state_boundary_refined.pdf');fig.savefig(HERE/'figures/state_boundary_refined.png',dpi=180);plt.close(fig)
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
