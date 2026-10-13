"""Exact known-input correction reconstruction where nominal is already safe.

This is a read-only sufficiency/localization calculation, not a new Q model.
All scenes are retained; Toy's preparatory-safety discrepancy is explicit.
"""
import numpy as np
from .study import HERE,read,write


def run():
    d=np.load(HERE/'dataset.npz');p=read(HERE/'protocol.json')
    b=np.load(HERE/'goal_frame_basis_truth.npz');lookup={s:t for s,t in zip(b['state_uid'],b['basis'])}
    labels=np.load(HERE/'motion_raw.npz')['vector_motion'];rows=[];outputs={}
    for i,sid in enumerate(d['state_uid']):
        n=int(d['entity_agent_mask'][i].sum());v=float(d['entity_globals'][i,4])*.82
        goal=np.column_stack((np.minimum(d['entity_agents'][i,:n,0].astype(float)*4.,v)/v,np.zeros(n)))
        pair=-d['entity_pairs'][i,:n,:n,:2].astype(float)*4.
        rel=(pair*np.minimum(1.,v/np.maximum(np.linalg.norm(pair,axis=-1,keepdims=True),1e-30))).sum(1)/((n-1)*v)
        nominal=d['feature'][i,:n,:2].astype(float)
        eps=1e-6
        perp=3.303687238760696*(nominal-(nominal*goal).sum(-1,keepdims=True)/(np.sum(goal*goal,-1,keepdims=True)+eps*eps)*goal)
        basis=np.stack((goal,perp,rel),-1)
        value=np.einsum('nfk,ek->enf',basis,d['eta'])
        selected=value[[0,5,10,15]].reshape(4,2*n)
        truth=(2*labels[i,[0,5,10,15],0,:2*n]-1)*p['correction_bound_vmax']
        rows.append(dict(scene=str(d['scene'][i]),state_uid=str(sid),controller=str(d['controller'][i]),
            max_delta_error_vmax=float(abs(selected-truth).max()),
            max_perp_basis_error=float(abs(perp-lookup[sid][:n,:,1]).max())))
        outputs[str(i)]=value
    summary=[]
    for sc in sorted(set(d['scene'])):
        rr=[r for r in rows if r['scene']==sc]
        summary.append(dict(scene=sc,state_controller_rows=len(rr),physical_states=len(set(r['state_uid'] for r in rr)),
            matching_rows_1e_6=sum(r['max_delta_error_vmax']<1e-6 for r in rr),max_delta_error_vmax=max(r['max_delta_error_vmax'] for r in rr)))
    write(HERE/'algebra_audit_results.json',dict(rows=rows,summary=summary,new_fit=0,new_query=0,new_rollout=0,
        history='Earlier operator andbasisexcitationaudits didnotreconstructentireactualdeltafromexistingcriticinputs. Usesonlyalreadycachedfeatures/targets; no newexperiment fitting.',
        boundary='Formula usesnominal currentreference descriptor forBperp. Exactonlywherepreparatorysafetydoesnotchangeitsperpendicularcomponent; Toyfailuresretained. Mathematical input-sufficiencywitness for matchingpanels, NOTa learned model, universalQproof, orFF/Qrepair.'))
    np.savez_compressed(HERE/'algebra_audit_predictions.npz',**outputs)
    print(summary)


if __name__=='__main__':run()
