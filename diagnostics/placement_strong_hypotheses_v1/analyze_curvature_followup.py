"""Follow-up orientation/curvature with active agent checks."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_curvature import fit_curvature
HERE=Path(__file__).resolve().parent
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')


def signature(r,arrays,goals):
    kind='goal' if r['goal_ever_margin']<=r['safety_always_margin'] else 'safety'
    t=r['goal_active_step'] if kind=='goal' else r['safety_active_step']
    agent=int(np.argmax(np.linalg.norm(arrays['positions'][t]-goals,axis=1))) if kind=='goal' else None
    return kind,t,agent


def main():
    p=json.loads((HERE/'curvature_followup_protocol.json').read_text())
    states={x['uid']:x for x in json.loads((HARD/'confirmation_manifest.json').read_text())['states']}
    rows=[]
    for c in p['centers']:
        uid,chain=c['uid'],c['chain'];goals=np.asarray(states[uid]['physical']['goals'])
        stem=f'{uid}_root{chain}_eval{chain}_center_e0.0_aNone_sNone'
        base=json.loads((HERE/'h1_gradient_results'/(stem+'.json')).read_text())
        with np.load(HERE/'h1_gradient_results'/(stem+'.npz')) as z:ba={k:z[k].copy() for k in z.files}
        sig0=signature(base,ba,goals)
        for radius in p['radii']:
            jobs=[x for x in p['jobs'] if x['uid']==uid and x['chain']==chain and x['radius']==radius]
            records=[];signatures=[]
            for j in jobs:
                stem=f"{uid}_{chain}_r{radius}_{j['offset_index']}"
                r=json.loads((HERE/'curvature_followup_results'/(stem+'.json')).read_text())
                with np.load(HERE/'curvature_followup_results'/(stem+'.npz')) as z:a={k:z[k].copy() for k in z.files}
                signatures.append(signature(r,a,goals));records.append(r)
            fit=fit_curvature(c,records,radius,20261008)
            rows.append(dict(uid=uid,chain=chain,radius=radius,center_signature=sig0,
                             branch_stable=all(x==sig0 for x in signatures),probe_signatures=signatures,
                             gradient_stability=c['gradient_stability'],fit=fit))
    normals=[]
    for uid in sorted({x['uid'] for x in p['centers']}):
        a,b=[x for x in p['centers'] if x['uid']==uid]
        g=np.asarray(a['gradient']);h=np.asarray(b['gradient'])
        angle=float(np.degrees(np.arccos(np.clip(g@h/(np.linalg.norm(g)*np.linalg.norm(h)),-1,1))))
        normals.append(dict(uid=uid,own_boundary_normal_angle_deg=angle,
                            TT_gradient_stability=a['gradient_stability'],FF_gradient_stability=b['gradient_stability']))
    (HERE/'curvature_followup_analysis.json').write_text(json.dumps(dict(rows=rows,orientation=normals),indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,uid in zip(axes,sorted({x['uid'] for x in rows})):
        for ch,color in (('TT','#1f77b4'),('FF','#d62728')):
            rr=[x for x in rows if x['uid']==uid and x['chain']==ch]
            for k,marker in ((0,'o'),(1,'s')):
                ax.plot([x['radius'] for x in rr],[x['fit']['principal_curvatures'][k] for x in rr],
                        color=color,marker=marker,label=f'{ch} k{k+1}')
        ax.set_title(uid[-6:]);ax.set_xlabel('tangent radius');ax.set_ylabel('principal curvature')
    axes[0].legend();fig.tight_layout();(HERE/'figures').mkdir(exist_ok=True)
    fig.savefig(HERE/'figures/curvature_followup.pdf');fig.savefig(HERE/'figures/curvature_followup.png',dpi=180);plt.close(fig)
    print(json.dumps(normals,indent=2))


if __name__=='__main__':main()
