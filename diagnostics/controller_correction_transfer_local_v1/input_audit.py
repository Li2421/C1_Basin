"""Check existing input retains the deterministic physical basis primitives."""
import numpy as np
from .study import HERE,read,write


def run():
    d=np.load(HERE/'dataset.npz');p=read(HERE/'protocol.json');states={s['state_uid']:s for s in p['states']}
    rows=[];raw={};ids=[]
    for sid in sorted(states):
        i=int(np.flatnonzero(d['state_uid']==sid)[0]);n=int(d['entity_agent_mask'][i].sum())
        z=np.load(HERE/'basis'/(sid+'.npz'));v=float(z['vmax'])
        np.testing.assert_allclose(d['entity_globals'][i,4]*.82,v,atol=1e-7,rtol=0)
        # These quantities use existing physical inputs only, no policy query.
        a=d['entity_agents'][i,:n].astype(float);pair=d['entity_pairs'][i,:n,:n].astype(float)
        goal=np.column_stack((np.minimum(a[:,0]*4.,v)/v,np.zeros(n)))
        delta=-pair[...,:2]*4.;mag=np.linalg.norm(delta,axis=-1,keepdims=True)
        rel=(delta*np.minimum(1.,v/np.maximum(mag,1e-30))).sum(1)/((n-1)*v)
        s=states[sid]['physical'];e=np.asarray(s['goals'])-np.asarray(s['positions'])
        e/=np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12)
        frame=np.stack((e,np.stack((-e[:,1],e[:,0]),-1)),-1)
        truth=np.einsum('ndk,ndf->nfk',z['basis_world'],frame)/v
        gerr=float(abs(goal-truth[...,0]).max());rerr=float(abs(rel-truth[...,2]).max())
        np.testing.assert_allclose(goal,truth[...,0],atol=1e-6,rtol=0)
        np.testing.assert_allclose(rel,truth[...,2],atol=1e-6,rtol=0)
        rows.append(dict(state_uid=sid,scene=str(d['scene'][i]),split=str(d['split'][i]),goal_reconstruction_error=gerr,relation_reconstruction_error=rerr,
            safe_perp_basis_max=float(abs(truth[...,1]).max())))
        raw[sid]=np.pad(truth,((0,4-n),(0,0),(0,0)));ids.append(sid)
    np.savez_compressed(HERE/'goal_frame_basis_truth.npz',state_uid=np.asarray(ids),basis=np.asarray([raw[s] for s in ids]))
    # This is an analysis inversion on fixed eta coordinates, never learned.
    design=np.column_stack((np.ones(4),d['eta'][[0,5,10,15]]))
    result=dict(rows=rows,primitive_input_sufficiency_verified=True,
        max_goal_error=max(r['goal_reconstruction_error'] for r in rows),
        max_relation_error=max(r['relation_reconstruction_error'] for r in rows),
        eta_affine_design_condition=float(np.linalg.cond(design)),
        boundary='Only physicalBgoal/Brel recovery established. Safety-projected nominal perpendicularbasis stilldepends onfixedreference/safety; no universalcontroller/Qsufficiency claim. No newquery/fit/rollout.')
    write(HERE/'input_audit.json',result);print({k:v for k,v in result.items() if k!='rows'})


if __name__=='__main__':run()
