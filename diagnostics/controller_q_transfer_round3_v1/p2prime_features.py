"""Outcome-free source/held controller features for conditional P2′."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
BASE=Path(__file__).resolve().parent

import jax
import numpy as np
from diagnostics.controller_q_diversity_v1.runtime import native_runtime,controller_context
from diagnostics.controller_q_diversity_v1.features import field_context
from diagnostics.controller_q_diversity_v1.train import compact_field
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
from diagnostics.controller_q_transfer_round3_v1.placement_runtime import make_program


def static_r3(rt,s):
    query=type('Query',(),{'__getattr__':lambda self,k:getattr(rt,k)})()
    nominal=controller_context(rt,s);field=field_context(query,s)
    x=np.asarray(compact_field(np.concatenate((nominal,field),axis=-1)),dtype=np.float32).reshape(-1)
    assert x.shape==(256,) and np.isfinite(x).all()
    return x


def static_r4(rt,s):
    env=rt.make_env(s);positions=env.positions.copy();velocity=env.velocities.copy();goal=env.goals.copy()
    direction=goal-positions;direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-12)
    key=key_for(s,0,0);rows=[]
    for delta in (0.,.1):
        env.positions[:]=positions+delta*direction;env.velocities[:]=velocity
        def act(eta):return rt.action(env,key,eta,'field')[0].ravel()
        zero=np.zeros(3);base=act(zero);eps=.001
        J=np.stack([(act(zero+np.eye(3)[j]*eps)-act(zero-np.eye(3)[j]*eps))/(2*eps) for j in range(3)],axis=1)
        rows.extend((base,J.ravel()))
    out=np.concatenate(rows).astype(np.float32)
    assert out.shape==(64,) and np.isfinite(out).all()
    return out


def main():
    protocol=json.loads((BASE/'P2PRIME_PROTOCOL.json').read_text())
    assert not json.loads((BASE/'P1_RESULTS.json').read_text())['gate']['p2_qualified']
    jax.config.update('jax_enable_x64',False)
    rt=native_runtime('ring_exchange');jax.config.update('jax_enable_x64',False)
    from ring_exchange.environment import RingExchangeEnv
    original=RingExchangeEnv.step
    def forbidden(*args,**kwargs):raise RuntimeError('feature query cannot call env.step')
    RingExchangeEnv.step=forbidden
    sources=protocol['source_controllers'];held=protocol['held_controllers'];rows=[];cached_r3={};t0=time.time()
    folder=BASE/'p2prime_features';folder.mkdir(exist_ok=True)
    try:
        for name in sources+held:
            dest=folder/f'{name}.npz'
            indices=list(range(288)) if name in sources else list(range(256,352))
            if dest.exists():
                data=np.load(dest);assert data['R3'].shape==(len(indices),256)
                if name in ('FF','TF'):
                    for i,x in zip(indices,data['R3']):cached_r3[protocol['states'][i]['state_uid']]=x
                rows.append(dict(name=name,reused=True,errors=[]));continue
            program=make_program(rt,name);r3=[];r4=[];errors=[]
            for i in indices:
                s=protocol['states'][i];uid=s['state_uid']
                try:
                    if name in ('FF','TT','TF','FT') and uid in cached_r3:
                        a=cached_r3[uid]
                    else:
                        a=static_r3(program,s)
                        if name in ('FF','TF'):cached_r3[uid]=a
                    b=static_r4(program,s)
                except Exception as exc:
                    errors.append(dict(state_index=i,type=type(exc).__name__,message=str(exc)[:300]))
                    a=np.full(256,np.nan,np.float32);b=np.full(64,np.nan,np.float32)
                r3.append(a);r4.append(b)
            np.savez_compressed(dest,R3=np.asarray(r3),R4_action=np.asarray(r4),indices=np.array(indices),
                                state_uids=np.array([protocol['states'][i]['state_uid'] for i in indices]))
            rows.append(dict(name=name,reused=False,errors=errors,seconds=time.time()-t0))
            print(name,'errors',len(errors),'elapsed',round(time.time()-t0,1),flush=True)
        # Historical native FF Q512 source labels need the same static input;
        # their state identities are preserved rather than regenerated.
        oldproto=json.loads((ROOT/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/protocol.json').read_text())
        dest=folder/'FF_historical512.npz'
        if dest.exists():
            prior=np.load(dest);assert prior['R3'].shape==(512,256)
            rows.append(dict(name='FF_historical512',reused=True,errors=[]))
        else:
            program=make_program(rt,'FF');r3=[];r4=[];errors=[]
            for i,s in enumerate(oldproto['states'][:512]):
                try:a=static_r3(program,s);b=static_r4(program,s)
                except Exception as exc:
                    errors.append(dict(state_index=i,type=type(exc).__name__,message=str(exc)[:300]))
                    a=np.full(256,np.nan,np.float32);b=np.full(64,np.nan,np.float32)
                r3.append(a);r4.append(b)
                if (i+1)%64==0:print('FF_historical512',i+1,'elapsed',round(time.time()-t0,1),flush=True)
            np.savez_compressed(dest,R3=np.asarray(r3),R4_action=np.asarray(r4),
                                state_uids=np.array([s['state_uid'] for s in oldproto['states'][:512]]))
            rows.append(dict(name='FF_historical512',reused=False,errors=errors,seconds=time.time()-t0))
    finally:
        RingExchangeEnv.step=original
    (BASE/'P2PRIME_FEATURE_RECEIPT.json').write_text(json.dumps(dict(status='COMPLETED',physical_steps=0,source_outcome_reads=0,
                                                                       rows=rows,elapsed_seconds=time.time()-t0),indent=2)+'\n')


if __name__=='__main__':main()
