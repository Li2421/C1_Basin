"""Static real-Flow descriptors; no simulator stepping or outcome access."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent

import jax
import numpy as np
from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
from diagnostics.controller_q_diversity_v1.runtime import native_runtime
from diagnostics.controller_q_transfer_round3_v1.p1_features import feature as prior_feature
from diagnostics.controller_q_transfer_round3_v1.p3_features import static_after_action
from ring_exchange.environment import RingExchangeEnv

def read(p):return json.loads(Path(p).read_text())

def run(index):
    p=read(HERE/'protocol.json');name=p['controller_order'][index]
    profile=p['controllers']['ring_exchange'][name]['payload']
    assert profile['jax_enable_x64'] is False
    jax.config.update('jax_enable_x64',False)
    original=RingExchangeEnv.step
    def forbidden(*args,**kwargs):raise RuntimeError('static feature query called env.step')
    RingExchangeEnv.step=forbidden
    t0=time.time();errors=[]
    try:
        rt=(native_runtime('ring_exchange') if name=='base_u_v10_local_macflow' else
            RingFlowAdapter(profile['flow_checkpoint_path'],read(profile['adapter_manifest_path']),profile['observation_frame']))
        dest=HERE/'static_features'/f'{name}.npz';dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            r3=[];r4=[];openloop=[]
            for i,s in enumerate(p['states']):
                try:
                    a,b=prior_feature(rt,s)
                    _,future=static_after_action(rt,s)
                    c=np.r_[b[:32],future].astype(np.float32)
                    assert c.shape==(64,) and np.isfinite(c).all()
                except Exception as exc:
                    errors.append(dict(state_index=i,type=type(exc).__name__,message=str(exc)[:300]))
                    a=np.full(256,np.nan,np.float32);b=np.full(64,np.nan,np.float32);c=np.full(64,np.nan,np.float32)
                r3.append(a);r4.append(b);openloop.append(c)
                if (i+1)%32==0:print(name,i+1,'seconds',round(time.time()-t0,1),flush=True)
            np.savez_compressed(dest,R3=np.asarray(r3),R4_action=np.asarray(r4),R4_openloop=np.asarray(openloop),
                state_uids=np.array([s['state_uid'] for s in p['states']]))
        validation=read(ROOT/'diagnostics/controller_q_transfer_round3_v1/P1_EXECUTION_PROTOCOL.json')['states']
        vd=HERE/'static_features'/f'{name}_validation.npz'
        if not vd.exists():
            old=np.load(ROOT/'diagnostics/controller_q_transfer_round3_v1/p1_features'/f'{name}.npz')
            assert list(old['state_uids'])==[s['state_uid'] for s in validation]
            v3=old['R3'];v4=old['R4_action'];vo=[]
            for i,s in enumerate(validation):
                try:
                    _,future=static_after_action(rt,s)
                    c=np.r_[v4[i,:32],future].astype(np.float32)
                    assert c.shape==(64,) and np.isfinite(c).all()
                except Exception as exc:
                    errors.append(dict(validation_index=i,type=type(exc).__name__,message=str(exc)[:300]))
                    c=np.full(64,np.nan,np.float32)
                vo.append(c)
            np.savez_compressed(vd,R3=v3,R4_action=v4,R4_openloop=np.asarray(vo),state_uids=old['state_uids'])
    finally:RingExchangeEnv.step=original
    rec=dict(controller=name,states=len(p['states']),validation_states=24,physical_steps=0,outcome_reads=0,
        jax_enable_x64=bool(jax.config.x64_enabled),seconds=time.time()-t0,errors=errors)
    (HERE/'static_features'/f'{name}_receipt.json').write_text(json.dumps(rec,indent=2)+'\n')
    assert not errors,rec
    print(json.dumps(rec),flush=True)

if __name__=='__main__':run(int(sys.argv[1]))
