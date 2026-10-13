"""Reconstruct the active goal-margin chain rule from saved full trajectories."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
STRONG=HERE.parent/'placement_strong_hypotheses_v1'
PRIOR=HERE.parent/'placement_boundary_mechanism_audit_v1'
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')

def main():
    states={s['uid']:s for s in json.loads((HARD/'confirmation_manifest.json').read_text())['states']}
    h1=json.loads((STRONG/'h1_gradient_analysis.json').read_text())['rows']
    rows=[];trans=[]
    for r in h1:
        uid=r['uid'];root=r['root_chain'];chain=r['eval_chain']
        name=f'{uid}_root{root}_eval{chain}_center_e0.0_aNone_sNone.npz'
        with np.load(STRONG/'h1_gradient_results'/name) as z:positions=z['positions'].copy()
        kind,t,agent=r['center_signature']
        if kind!='goal':continue
        pos=positions[t,agent];goal=np.asarray(states[uid]['physical']['goals'],float)[agent]
        direction=(goal-pos)/np.linalg.norm(goal-pos)
        for eps,est in r['estimates'].items():
            S=np.asarray(est['S_active'],float)[:,agent,:]
            reconstructed=S@direction
            direct=np.asarray(est['rho_gradient'],float)
            old=np.asarray(est['goal_chain_gradient'],float)
            rows.append(dict(uid=uid,root_chain=root,eval_chain=chain,epsilon=float(eps),
                 center_rho=r['center_rho'],active_signature=r['center_signature'],
                 active_stable=est['all_signatures_same'],
                 direct_gradient=direct.tolist(),reconstructed_gradient=reconstructed.tolist(),
                 source_chain_gradient=old.tolist(),
                 relative_error=float(np.linalg.norm(reconstructed-direct)/max(np.linalg.norm(direct),1e-12)),
                 cosine=float(np.dot(reconstructed,direct)/max(np.linalg.norm(reconstructed)*np.linalg.norm(direct),1e-12)),
                 signed_component_error=(reconstructed-direct).tolist()))
        if root==chain:
            e1=np.asarray(r['estimates']['0.0005']['rho_gradient'],float)
            e2=np.asarray(r['estimates']['0.002']['rho_gradient'],float)
            trans.append(dict(uid=uid,chain=chain,norm=float(np.linalg.norm(e1)),
                scale_change=float(np.linalg.norm(e1-e2)/max(np.linalg.norm(e1),1e-12)),
                active_stable=r['estimates']['0.0005']['all_signatures_same'] and r['estimates']['0.002']['all_signatures_same'],
                origin='regular_followup'))
    old=json.loads((PRIOR/'boundary_normals.json').read_text())
    for r in old:
        a=np.asarray(r['gradient_eps0005'],float);b=np.asarray(r['gradient_eps002'],float)
        trans.append(dict(uid=r['uid'],chain=r['chain'],norm=float(np.linalg.norm(a)),
             scale_change=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12)),
             active_stable=all(len({x['minus_goal_step'],x['plus_goal_step']})==1
                               for e in r['side_signatures'].values() for x in e),
             origin='previous_preselected'))
    stats=dict(chain_rows=len(rows),active_stable_rows=sum(r['active_stable'] for r in rows),
        median_relative_chain_error=float(np.median([r['relative_error'] for r in rows])),
        max_relative_chain_error=float(np.max([r['relative_error'] for r in rows])),
        min_cosine=float(np.min([r['cosine'] for r in rows])),
        own_boundary_points=len(trans),own_boundary_gradient_norm_quantiles=np.quantile([r['norm'] for r in trans],[0,.25,.5,.75,1]).tolist(),
        near_zero_count_below_001=sum(r['norm']<.01 for r in trans),
        near_zero_count_below_005=sum(r['norm']<.05 for r in trans),
        regular_points_scale_change_below_1pct=sum(r['scale_change']<.01 for r in trans),
        note='The active goal margin has one nonzero position-gradient at its limiting timestep; safety-active branches not covered.')
    for name,obj in [('chain_rule_rows.json',rows),('boundary_transversality_rows.json',trans),('boundary_property_summary.json',stats)]:
        (HERE/name).write_text(json.dumps(obj,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(9,3.3))
    axs[0].hist([r['norm'] for r in trans],bins=8,color='#1f77b4')
    axs[0].set_xlabel('$||\\nabla_\\eta\\rho||$');axs[0].set_ylabel('boundary points')
    axs[1].scatter([r['epsilon'] for r in rows],[r['relative_error'] for r in rows],color='#d95f02')
    axs[1].set_xscale('log');axs[1].set_yscale('log');axs[1].set_xlabel('finite-difference epsilon');axs[1].set_ylabel('chain relative error')
    fig.tight_layout();fig.savefig(HERE/'figures/chain_transversality.pdf');fig.savefig(HERE/'figures/chain_transversality.png',dpi=180);plt.close(fig)
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
