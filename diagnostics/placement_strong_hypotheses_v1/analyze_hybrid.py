"""Compare exact finite-response transfer swaps against original TT/FF boundaries."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=ROOT/'diagnostics/placement_boundary_mechanism_audit_v1'


def crossing(rows):
    rows=sorted(rows,key=lambda x:x['fraction'])
    roots=[]
    for a,b in zip(rows[:-1],rows[1:]):
        if (a['rho']>0)!=(b['rho']>0):
            t=a['fraction']+(b['fraction']-a['fraction'])*(-a['rho'])/(b['rho']-a['rho'])
            roots.append(dict(t=float(t),bracket=[a['fraction'],b['fraction']],rho=[a['rho'],b['rho']]))
    return roots


def main():
    p=json.loads((HERE/'hybrid_protocol.json').read_text());old=json.loads((PRIOR/'boundary_normal_protocol.json').read_text())
    results=[]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for i,line in enumerate(p['lines']):
        uid=line['uid'];entry=dict(uid=uid,normal_roots={r['chain']:r['fraction'] for r in old if r['uid']==uid},modes={})
        for mode,color in (('FF_base_TT_corr','#2ca02c'),('TT_base_FF_corr','#9467bd')):
            jobs=[j for j in p['jobs'] if j['uid']==uid and j['mode']==mode]
            rows=[]
            for j in jobs:
                r=json.loads((HERE/'hybrid_results'/f"{uid}_{mode}_{j['index']:02d}.json").read_text())
                rows.append(r)
            roots=crossing(rows)
            closure=np.array([x['closure_mean'] for x in rows]);donor=np.array([x['donor_correction_mean'] for x in rows])
            nonzero=np.array([x['closure_nonzero_steps']/max(x['steps'],1) for x in rows])
            entry['modes'][mode]=dict(roots=roots,rows=rows,median_closure_to_donor=float(np.median(closure/np.maximum(donor,1e-12))),
                median_closure_nonzero_step_fraction=float(np.median(nonzero)),
                max_instant_closure=float(max(x['closure_max'] for x in rows)),
                rho_success_sign_mismatches=sum((r['rho']>0)!=r['success'] for r in rows))
            axes[i].plot([x['fraction'] for x in rows],[x['rho'] for x in rows],'-o',color=color,label=mode)
        axes[i].axhline(0,color='black',lw=.8)
        for ch,linestyle in (('TT','--'),('FF',':')):
            axes[i].axvline(entry['normal_roots'][ch],color='black',linestyle=linestyle,label=ch+' normal root')
        axes[i].set_title(uid.split('_')[-1]);axes[i].set_xlabel('line fraction');axes[i].set_ylabel('diagnostic rho')
        results.append(entry)
    axes[0].legend(fontsize=7);fig.tight_layout();(HERE/'figures').mkdir(exist_ok=True)
    fig.savefig(HERE/'figures/action_family_swap.pdf');fig.savefig(HERE/'figures/action_family_swap.png',dpi=180);plt.close(fig)
    (HERE/'hybrid_analysis.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ('uid','normal_roots')}|{'hybrid_roots':{m:[x['t'] for x in v['roots']] for m,v in r['modes'].items()},
                       'median_projection_fraction':{m:v['median_closure_to_donor'] for m,v in r['modes'].items()}} for r in results],indent=2))


if __name__=='__main__':main()
