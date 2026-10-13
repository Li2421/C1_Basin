"""Initial-state margin continuity and matched local boundary probes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent

def label(j):
    return f"{j['uid']}_{j['chain']}_h{j['radius']:g}_s{j['sign']}_o{j['line_offset']:+g}"

def local_root(points):
    vals=sorted((float(k),float(v['rho'])) for k,v in points.items() if v['rho'] is not None)
    candidates=[]
    for (a,ya),(b,yb) in zip(vals,vals[1:]):
        if ya*yb<=0 and ya!=yb:candidates.append(a-ya*(b-a)/(yb-ya))
    return candidates[0] if len(candidates)==1 else None

def main():
    rp=json.loads((HERE/'rollout_protocol.json').read_text())
    jobs=[j for j in rp['jobs'] if j['kind']=='state_margin']
    records={};rows=[]
    for j in jobs:
        x=json.loads((HERE/'rollout_results'/(label(j)+'.json')).read_text())
        records[(j['uid'],j['chain'],j['radius'],j['sign'],j['line_offset'])]=x
    for j in jobs:
        if j['radius']==0:continue
        x=records[(j['uid'],j['chain'],j['radius'],j['sign'],j['line_offset'])]
        b=records[(j['uid'],j['chain'],0.,0,j['line_offset'])]
        diff=(x['rho']-b['rho']) if x['rho'] is not None and b['rho'] is not None else None
        rows.append(dict(uid=j['uid'],scene=j['scene'],chain=j['chain'],radius=j['radius'],
             sign=j['sign'],eta_line_offset=j['line_offset'],base_rho=b['rho'],perturbed_rho=x['rho'],
             margin_difference=diff,margin_quotient=abs(diff)/j['radius'] if diff is not None else None,
             termination_base=b['termination'],termination_perturbed=x['termination'],
             termination_switch=b['termination']!=x['termination'],
             active_base=b['active_margin'],active_perturbed=x['active_margin'],
             active_switch=b['active_margin']!=x['active_margin'],
             success_base=b['success'],success_perturbed=x['success']))
    roots=[]
    for uid in ('hard_confirm_ring_1_920004','hard_confirm_ring_1_920005'):
        for chain in ('TT','FF'):
            base={j['line_offset']:records[(uid,chain,0.,0,j['line_offset'])]
                  for j in jobs if j['uid']==uid and j['chain']==chain and j['radius']==0}
            r0=local_root(base)
            for h in (0.,.001,.003,.01):
                for sign in ((0,) if h==0 else (-1,1)):
                    pts={j['line_offset']:records[(uid,chain,h,sign,j['line_offset'])]
                         for j in jobs if j['uid']==uid and j['chain']==chain and j['radius']==h and j['sign']==sign}
                    rr=local_root(pts)
                    roots.append(dict(uid=uid,chain=chain,radius=h,sign=sign,root_offset=rr,
                        base_root_offset=r0,root_displacement=rr-r0 if rr is not None and r0 is not None else None,
                        normalized_displacement=abs(rr-r0)/h if h and rr is not None and r0 is not None else None,
                        bracketed=rr is not None,
                        rho_offsets={str(k):v['rho'] for k,v in pts.items()},
                        term_offsets={str(k):v['termination'] for k,v in pts.items()},
                        active_offsets={str(k):v['active_margin'] for k,v in pts.items()}))
    def summaries(scene):
        rr=[r for r in rows if r['scene']==scene]
        out=dict(n=len(rr),termination_switches=sum(r['termination_switch'] for r in rr),
                 active_margin_switches=sum(r['active_switch'] for r in rr),
                 by_radius={})
        for h in (.001,.003,.01):
            aa=[r for r in rr if r['radius']==h and r['margin_quotient'] is not None]
            out['by_radius'][str(h)]=dict(n=len(aa),median_margin_quotient=float(np.median([r['margin_quotient'] for r in aa])),
                max_margin_quotient=float(np.max([r['margin_quotient'] for r in aa])),
                termination_switch_fraction=float(np.mean([r['termination_switch'] for r in aa])),
                active_switch_fraction=float(np.mean([r['active_switch'] for r in aa])))
        return out
    out=dict(ring=summaries('ring_exchange'),toy=summaries('toy_give_way'),
             root_probe=dict(total=len(roots),bracketed=sum(r['bracketed'] for r in roots),
                 by_radius={str(h):dict(bracketed=sum(r['bracketed'] for r in roots if r['radius']==h),
                     median_normalized_displacement=float(np.median([r['normalized_displacement'] for r in roots if r['radius']==h and r['normalized_displacement'] is not None]))
                         if any(r['radius']==h and r['normalized_displacement'] is not None for r in roots) else None)
                     for h in (.001,.003,.01)}),
             note='Three-point linear root interpolation, not a high-precision root; missing bracket is censored.')
    for name,value in [('state_margin_rows.json',rows),('state_boundary_roots.json',roots),('state_margin_summary.json',out)]:
        (HERE/name).write_text(json.dumps(value,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(9,3.4))
    for scene,c in (('ring_exchange','#1f77b4'),('toy_give_way','#d95f02')):
        rr=[r for r in rows if r['scene']==scene and r['margin_difference'] is not None]
        axs[0].scatter([r['radius'] for r in rr],[abs(r['margin_difference']) for r in rr],color=c,label=scene,alpha=.55)
    axs[0].set_xscale('log');axs[0].set_yscale('log');axs[0].set_xlabel('$||\\Delta s_0||$');axs[0].set_ylabel('$|\\Delta\\rho|$');axs[0].legend()
    for chain,c in (('TT','#1f77b4'),('FF','#d95f02')):
        rr=[r for r in roots if r['chain']==chain and r['root_displacement'] is not None and r['radius']]
        axs[1].scatter([r['radius'] for r in rr],[abs(r['root_displacement']) for r in rr],color=c,label=chain,alpha=.7)
    axs[1].set_xscale('log');axs[1].set_yscale('log');axs[1].set_xlabel('$||\\Delta s_0||$');axs[1].set_ylabel('boundary line-fraction shift');axs[1].legend()
    fig.tight_layout();fig.savefig(HERE/'figures/state_margin_boundary_continuity.pdf');fig.savefig(HERE/'figures/state_margin_boundary_continuity.png',dpi=180);plt.close(fig)
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
