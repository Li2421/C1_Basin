"""One analytic no-step kinematic probe following η0 executed action."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent

import jax
import numpy as np
from diagnostics.controller_q_diversity_v1.runtime import native_runtime
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
from diagnostics.controller_q_transfer_round3_v1.placement_runtime import make_program


def static_after_action(rt,s):
    env=rt.make_env(s);p=env.positions.copy();v=env.velocities.copy();zero=np.zeros(3)
    k0=key_for(s,0,0);a0=rt.action(env,k0,zero,'field')[0]
    env.positions[:]=p+env.config.dt*a0;env.velocities[:]=a0
    k1=key_for(s,0,1)
    def action(eta):return rt.action(env,k1,eta,'field')[0].ravel()
    base=action(zero);eps=.001
    J=np.stack([(action(zero+np.eye(3)[j]*eps)-action(zero-np.eye(3)[j]*eps))/(2*eps) for j in range(3)],axis=1)
    return np.r_[a0.ravel(),np.zeros(24)],np.r_[base,J.ravel()]


def main():
    assert not json.loads((HERE/'P2PRIME_RESULTS.json').read_text())['primary_confirmed']
    p=json.loads((HERE/'P2PRIME_PROTOCOL.json').read_text());jax.config.update('jax_enable_x64',False)
    rt=native_runtime('ring_exchange');jax.config.update('jax_enable_x64',False)
    from ring_exchange.environment import RingExchangeEnv
    original=RingExchangeEnv.step
    def forbidden(*a,**kw):raise RuntimeError('P3 static query called env.step')
    RingExchangeEnv.step=forbidden
    folder=HERE/'p3_features';folder.mkdir(exist_ok=True);rows=[];t=time.time()
    try:
        for name in p['source_controllers']+p['held_controllers']+['FF_historical512']:
            dest=folder/f'{name}.npz'
            if name=='FF_historical512':
                old=json.loads((ROOT/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/protocol.json').read_text())
                states=old['states'][:512];program=make_program(rt,'FF')
                cached=np.load(HERE/'p2prime_features'/'FF_historical512.npz')['R4_action']
            else:
                ids=list(range(288)) if name in p['source_controllers'] else list(range(256,352))
                states=[p['states'][i] for i in ids];program=make_program(rt,name)
                cached=np.load(HERE/'p2prime_features'/f'{name}.npz')['R4_action']
            if dest.exists():
                z=np.load(dest);assert z['R4_openloop'].shape==(len(states),64)
                rows.append(dict(name=name,reused=True,errors=[]));continue
            features=[];errors=[]
            for i,s in enumerate(states):
                try:
                    _,future=static_after_action(program,s)
                    z=np.r_[cached[i,:32],future].astype(np.float32)
                    assert z.shape==(64,) and np.isfinite(z).all()
                except Exception as exc:
                    errors.append(dict(state_index=i,type=type(exc).__name__,message=str(exc)[:300]))
                    z=np.full(64,np.nan,np.float32)
                features.append(z)
                if (i+1)%64==0:print(name,i+1,'elapsed',round(time.time()-t,1),flush=True)
            np.savez_compressed(dest,R4_openloop=np.asarray(features),state_uids=np.array([s['state_uid'] for s in states]))
            rows.append(dict(name=name,reused=False,errors=errors,elapsed_seconds=time.time()-t))
    finally:
        RingExchangeEnv.step=original
    (HERE/'P3_FEATURE_RECEIPT.json').write_text(json.dumps(dict(status='COMPLETED',physical_steps=0,outcome_used_for_features=False,rows=rows,
                                                                 elapsed_seconds=time.time()-t),indent=2)+'\n')


if __name__=='__main__':main()
