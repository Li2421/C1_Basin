"""Forward parity and independent finite-difference audits of the frozen VJP."""
import json
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from single_integrator.c1.risk.joint_frozen import geometry,trajectory
from single_integrator.cbf import barrier_constraints,CBFConfig
from single_integrator.environment import GiveWayEnv,Config
from audit_c1_candidate_coverage import metrics

OUT=ROOT/'results/c1_jax_training_audit';OUT.mkdir(exist_ok=True)

def main():
    jax.config.update('jax_enable_x64',True);env=GiveWayEnv(Config(corridor_half_length=1.3));cfg=CBFConfig();rows=[];local=[]
    for rid in [100000,100003,100019,100034,100037,100043,100046,100058]:
        s=json.loads((ROOT/f'results/c1_frozen_unseen_64/states/{rid}.json').read_text())
        for cid in ['zero',s['selected']['cid']]:
            with np.load(ROOT/f'results/c1_frozen_unseen_64/traces/{rid}/{cid}.npz') as data:z={k:data[k] for k in data.files}
            aa=[];bb=[]
            for x in z['positions_before'][100:300]:
                a,b,_=barrier_constraints(dict(positions=x,walls=env.walls,config=env.config.to_dict()),cfg);aa.append(a);bb.append(b)
            A=jnp.asarray(np.array(aa));b=jnp.asarray(np.array(bb));v=jnp.asarray(z['candidate'][100:300])
            g=geometry(v,A,b);ge=float(np.max(np.abs(np.asarray(g)-z['g'])))
            T=len(z['success']);pad=max(0,500-T)
            before=np.concatenate([z['positions_before'][:500],np.repeat(z['positions_after'][-1][None],pad,axis=0)])
            after=np.concatenate([z['positions_after'][:500],np.repeat(z['positions_after'][-1][None],pad,axis=0)])
            candidate=np.concatenate([z['candidate'][:500],np.zeros((pad,4))]);success=np.r_[z['success'][:500],np.ones(pad,bool)]
            terms=trajectory(jnp.asarray(before),jnp.asarray(after),jnp.asarray(candidate),g,jnp.asarray(z['goals']),jnp.asarray(success))
            ref=metrics(z)[0];expected=np.array([ref['P'],ref['g'],ref['Stwo'],ref['score']]);err=float(np.max(np.abs(terms-expected)))
            rows.append(dict(rid=rid,cid=cid,max_g_error=ge,max_terms_error=err));print(rows[-1],flush=True)
            for k in [0,80,160]:
                rng=np.random.default_rng(rid+k);dv=rng.normal(size=4);dv/=np.linalg.norm(dv)
                fn=lambda v:geometry(v[None],A[k:k+1],b[k:k+1])[0]
                grad=jax.grad(fn)(v[k]);ad=float(jnp.dot(grad,dv));fds=[]
                for h in [1e-4,1e-5,1e-6]:fds.append(float((fn(v[k]+h*dv)-fn(v[k]-h*dv))/(2*h)))
                local.append(dict(rid=rid,cid=cid,step=100+k,gradient_norm=float(jnp.linalg.norm(grad)),ad=ad,finite_differences=fds,
                    relative_error=abs(ad-fds[-1])/max(1.,abs(ad),abs(fds[-1]))))
    result=dict(forward=rows,local_gradients=local,max_forward_error=max(r['max_terms_error'] for r in rows),
        max_gradient_relative_error=max(r['relative_error'] for r in local))
    (OUT/'forward_local_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(result['max_forward_error'],result['max_gradient_relative_error'])
    assert result['max_forward_error']<1e-8

if __name__=='__main__':main()
