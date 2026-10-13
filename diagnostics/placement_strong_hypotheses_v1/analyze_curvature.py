"""Estimate regular-branch tangent curvature with radius and bootstrap checks."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=ROOT/'diagnostics/placement_boundary_mechanism_audit_v1'


def signature(r):
    which='goal' if r['goal_ever_margin']<=r['safety_always_margin'] else 'safety'
    step=r['goal_active_step'] if which=='goal' else r['safety_active_step']
    return which,step


def fit_curvature(center,records,radius,seed):
    g=np.asarray(center['gradient']);gn=np.linalg.norm(g)
    f0=float(center['center_rho'])
    pts=[(0.,0.,f0)]
    for r in records:
        a,b=r['uv'];pts.append((radius*a,radius*b,r['rho']))
    arr=np.asarray(pts)
    X=np.stack([np.ones(len(arr)),arr[:,0],arr[:,1],.5*arr[:,0]**2,arr[:,0]*arr[:,1],.5*arr[:,1]**2],axis=1)
    y=arr[:,2]
    coef=np.linalg.lstsq(X,y,rcond=None)[0]
    pred=X@coef
    h=np.array([[coef[3],coef[4]],[coef[4],coef[5]]])
    curv,ev=np.linalg.eigh(-h/gn)
    ambient_dirs=np.asarray(center['tangent'])@ev
    rng=np.random.default_rng(seed)
    boots=[]
    for _ in range(1000):
        idx=np.r_[0,rng.integers(1,len(arr),size=len(arr)-1)]
        if np.linalg.matrix_rank(X[idx])<6:continue
        b=np.linalg.lstsq(X[idx],y[idx],rcond=None)[0]
        hb=np.array([[b[3],b[4]],[b[4],b[5]]])
        boots.append(np.linalg.eigvalsh(-hb/gn))
    boots=np.asarray(boots)
    return dict(principal_curvatures=curv.tolist(),principal_directions=ambient_dirs.T.tolist(),
        mean_curvature=float(np.mean(curv)),
        gaussian_curvature=float(np.prod(curv)),
        fit_rmse=float(np.sqrt(np.mean((pred-y)**2))),
        fit_relative_rmse=float(np.linalg.norm(pred-y)/max(np.linalg.norm(y-y.mean()),1e-12)),
        bootstrap_valid=len(boots),bootstrap_95=np.quantile(boots,[.025,.975],axis=0).tolist() if len(boots)>20 else None,
        gradient_tangent_fit=coef[1:3].tolist(),center_rho=f0)


def main():
    design=json.loads((HERE/'curvature_protocol.json').read_text());rows=[]
    for c in design['centers']:
        prior=json.loads((PRIOR/'boundary_normal_protocol.json').read_text())
        old=next(x for x in prior if x['uid']==c['uid'] and x['chain']==c['chain'])
        import sys
        sys.path.insert(0,str(ROOT))
        from shared_rollout_db.src.rollout_db import eta_identity
        eid=eta_identity(c['center_eta'])[0]
        base=json.loads((PRIOR/'boundary_normal_results'/f"{c['uid']}_{c['chain']}_{eid}.json").read_text())
        base_sig=signature(base)
        for radius in design['radii']:
            jobs=[x for x in design['jobs'] if x['uid']==c['uid'] and x['chain']==c['chain'] and x['radius']==radius]
            records=[]
            for j in jobs:
                stem=f"{j['uid']}_{j['chain']}_r{j['radius']}_{j['offset_index']}"
                r=json.loads((HERE/'curvature_results'/(stem+'.json')).read_text())
                records.append(r)
            sigs=[signature(r) for r in records]
            same=all(x==base_sig for x in sigs)
            fit=fit_curvature(c,records,radius,20261008) if all(r['rho'] is not None for r in records) else None
            rows.append(dict(uid=c['uid'],chain=c['chain'],radius=radius,
                             branch_stable=same,center_signature=base_sig,probe_signatures=sigs,
                             termination_set=sorted({r['termination'] for r in records}),
                             gradient_stability=c['gradient_stability'],fit=fit))
    pairs=[]
    for uid in sorted({c['uid'] for c in design['centers']}):
        a,b=[c for c in design['centers'] if c['uid']==uid]
        g=np.asarray(a['gradient']);h=np.asarray(b['gradient'])
        angle=float(np.degrees(np.arccos(np.clip(np.dot(g,h)/(np.linalg.norm(g)*np.linalg.norm(h)),-1,1))))
        pairs.append(dict(uid=uid,orientation_angle_deg=angle,TT_gradient_stability=a['gradient_stability'],
                          FF_gradient_stability=b['gradient_stability']))
    out=dict(centers=rows,orientation=pairs,
             note='Curvature based on local rho Hessian projected onto tangent basis; reject radii with active factor switches.')
    (HERE/'curvature_analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for i,uid in enumerate(sorted({x['uid'] for x in rows})):
        ax=axes[i]
        for chain,color in (('TT','#1f77b4'),('FF','#d62728')):
            rr=[x for x in rows if x['uid']==uid and x['chain']==chain]
            for k,marker in ((0,'o'),(1,'s')):
                ax.plot([x['radius'] for x in rr],[x['fit']['principal_curvatures'][k] for x in rr],
                        marker=marker,color=color,label=f'{chain} k{k+1}' if i==0 else None)
            for x in rr:
                if not x['branch_stable']:ax.axvspan(x['radius']*.98,x['radius']*1.02,color='gray',alpha=.15)
        ax.set_title(uid.split('_')[-1]);ax.set_xlabel('tangent radius');ax.set_ylabel('estimated principal curvature')
    axes[0].legend();fig.tight_layout();(HERE/'figures').mkdir(exist_ok=True)
    fig.savefig(HERE/'figures/curvature_radius.pdf');fig.savefig(HERE/'figures/curvature_radius.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(orientation=pairs,regular=[{k:r[k] for k in ('uid','chain','radius','branch_stable')} for r in rows]),indent=2))


if __name__=='__main__':main()
