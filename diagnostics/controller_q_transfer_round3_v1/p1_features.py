"""Deployment-time static R3/R4_action features for P1, with no env.step."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent

import numpy as np
import jax
from diagnostics.controller_q_diversity_v1.runtime import controller_context
from diagnostics.controller_q_diversity_v1.features import field_context
from diagnostics.controller_q_diversity_v1.train import compact_field
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter


def feature(rt,state):
    query=type('Query',(),{'__getattr__':lambda self,key:getattr(rt,key)})()
    nominal=controller_context(rt,state)
    field=field_context(query,state)
    compact=np.asarray(compact_field(np.concatenate((nominal,field),axis=-1)),dtype=np.float32).reshape(-1)
    assert compact.shape==(256,)
    env=rt.make_env(state)
    p=env.positions.copy();v=env.velocities.copy();g=env.goals.copy()
    direction=g-p;direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-12)
    key=key_for(state,0,0)
    blocks=[]
    for offset in (0.,.1):
        env.positions[:]=p+offset*direction
        env.velocities[:]=v
        def action(eta):return rt.action(env,key,eta,'field')[0].ravel()
        eta=np.zeros(3)
        a=action(eta)
        eps=.001
        J=np.stack([(action(eta+np.eye(3)[j]*eps)-action(eta-np.eye(3)[j]*eps))/(2*eps) for j in range(3)],axis=1)
        blocks.extend((a,J.ravel()))
    env.positions[:]=p;env.velocities[:]=v
    desc=np.concatenate(blocks).astype(np.float32)
    assert desc.shape==(64,) and np.isfinite(desc).all()
    return compact,desc


def main():
    from ring_exchange.environment import RingExchangeEnv
    p=json.loads((HERE/'P1_PROTOCOL.json').read_text())
    jax.config.update('jax_enable_x64',False)
    original=RingExchangeEnv.step
    def forbidden(*a,**kw):raise RuntimeError('static feature query called env.step')
    RingExchangeEnv.step=forbidden
    receipts=[];start=time.time()
    try:
        for name in p['controller_order']:
            dest=HERE/'p1_features'/f'{name}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                prior=np.load(dest)
                assert prior['R3'].shape==(24,256) and prior['R4_action'].shape==(24,64)
                receipts.append(dict(name=name,reused=True,errors=[]));continue
            profile=p['controllers']['ring_exchange'][name]['payload'];manifest=json.loads(Path(profile['adapter_manifest_path']).read_text())
            jax.config.update('jax_enable_x64',False)
            rt=RingFlowAdapter(profile['flow_checkpoint_path'],manifest,profile['observation_frame'])
            r3=[];r4=[];errors=[]
            for i,s in enumerate(p['states']):
                try:
                    a,b=feature(rt,s)
                except Exception as exc:
                    errors.append(dict(state=i,type=type(exc).__name__,message=str(exc)[:300]))
                    a=np.full(256,np.nan,np.float32);b=np.full(64,np.nan,np.float32)
                r3.append(a);r4.append(b)
            np.savez_compressed(dest,R3=np.asarray(r3),R4_action=np.asarray(r4),state_uids=np.array([s['state_uid'] for s in p['states']]))
            receipts.append(dict(name=name,reused=False,errors=errors,seconds=time.time()-start))
            print(name,'errors',len(errors),'elapsed',round(time.time()-start,1),flush=True)
    finally:
        RingExchangeEnv.step=original
    (HERE/'P1_FEATURE_RECEIPT.json').write_text(json.dumps(dict(status='COMPLETED',rows=receipts,physical_steps=0,outcome_reads=0,total_seconds=time.time()-start),indent=2)+'\n')


if __name__=='__main__':main()
