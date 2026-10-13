"""Does the controller family excite the unchanged three eta directions?"""
import numpy as np
from .study import HERE,read,write

def run():
    import jax
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime
    from diagnostics.controller_response_families_v1.runtime import EndpointMatchedField
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    from .runtime import FlowEndpointMatchedField
    p=read(HERE/'protocol.json');bases={};rows=[];fields=[]
    for state in p['states']:
        sc=state['scenario']
        if sc not in bases:bases[sc]=native_runtime(sc)
        base=bases[sc];jax.config.update('jax_enable_x64',False)
        for name,cls in [('goal',EndpointMatchedField),('flow',FlowEndpointMatchedField)]:
            rt=cls(base,'isotropic');env=rt.make_env(state)
            for seed in p['seeds']:
                key=key_for(state,seed,0);_,_,_,safe,_,_=rt.prepare(env,key,np.zeros(3))
                b=np.stack(rt.basis.compute(env.positions,env.goals,safe,rt.config.max_speed).values,-1)
                norms=np.linalg.norm(b.reshape(-1,3),axis=0);sv=np.linalg.svd(b.reshape(-1,3),compute_uv=False)
                rows.append(dict(scene=sc,state_uid=state['state_uid'],reference=name,seed=seed,
                    column_norms=norms.tolist(),singular_values=sv.tolist(),relative_min_singular=float(sv[-1]/max(sv[0],1e-12)),
                    flow_perp_to_goal=float(norms[1]/max(norms[0],1e-12)),numerical_rank=int((sv>sv[0]*1e-6).sum())))
                fields.append(np.pad(b,((0,4-len(b)),(0,0),(0,0))))
    summary=[]
    for sc in sorted(bases):
        for name in ('goal','flow'):
            rr=[r for r in rows if r['scene']==sc and r['reference']==name]
            summary.append(dict(scene=sc,reference=name,states=8,seed_queries=16,
                median_perp_to_goal=float(np.median([r['flow_perp_to_goal'] for r in rr])),
                median_relative_min_singular=float(np.median([r['relative_min_singular'] for r in rr])),
                rank_deficient_queries=sum(r['numerical_rank']<3 for r in rr)))
    np.savez_compressed(HERE/'basis_fields.npz',fields=np.asarray(fields))
    write(HERE/'basis_excitation_audit.json',dict(rows=rows,summary=summary,physical_steps=0,new_rollouts=0,
        limitation='Initial-state intervention-direction support, not a proof that this alone causes the entire long-horizon Q difference. Definition and dimensionality of eta remain unchanged.'))
    for r in summary:print(r)

if __name__=='__main__':run()
