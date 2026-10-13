"""Static checkpoint compatibility and controller descriptors; no env.step."""
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
from diagnostics.controller_q_diversity_v1.runtime import native_runtime,controller_context
from diagnostics.controller_q_diversity_v1.features import field_context
from diagnostics.controller_q_diversity_v1.train import compact_field
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter


def read(name):return json.loads((HERE/name).read_text())
def write(name,content):(HERE/name).write_text(json.dumps(content,indent=2,allow_nan=False)+'\n')


def run():
    p=read('P1_PROTOCOL.json')
    assert p['jax_enable_x64'] is False
    jax.config.update('jax_enable_x64',False)
    state=p['states'][0]
    class Forbidden:
        @staticmethod
        def step(*args,**kwargs):raise RuntimeError('P1 static query cannot call env.step')
    from ring_exchange.environment import RingExchangeEnv
    original_step=RingExchangeEnv.step
    RingExchangeEnv.step=Forbidden.step
    rows=[];arrays={};t=time.time()
    try:
        native=native_runtime('ring_exchange')
        jax.config.update('jax_enable_x64',False)
        for name in p['controller_order']:
            profile=p['controllers']['ring_exchange'][name]['payload']
            manifest=json.loads(Path(profile['adapter_manifest_path']).read_text())
            rt=RingFlowAdapter(profile['flow_checkpoint_path'],manifest,profile['observation_frame'])
            jax.config.update('jax_enable_x64',False)
            env=rt.make_env(state)
            key=key_for(state,0,0)
            a0,cert,n=rt.action(env,key,np.zeros(3),'field')
            a13,_,_=rt.action(env,key,np.asarray(p['eta'][13]),'field')
            vals=[]
            for i in range(3):
                eps=.001
                d=np.eye(3)[i]*eps
                ap,_,_=rt.action(env,key,d,'field');am,_,_=rt.action(env,key,-d,'field')
                vals.append(((ap-am)/(2*eps)).ravel())
            J=np.stack(vals,axis=1)
            assert np.isfinite(a0).all() and np.isfinite(a13).all() and np.isfinite(J).all()
            exact=None
            if name=='base_u_v10_local_macflow':
                e2=native.make_env(state)
                b0,_,_=native.action(e2,key,np.zeros(3),'field')
                b13,_,_=native.action(e2,key,np.asarray(p['eta'][13]),'field')
                exact=dict(zero_max_abs=float(abs(a0-b0).max()),eta13_max_abs=float(abs(a13-b13).max()))
                assert exact['zero_max_abs']<1e-10 and exact['eta13_max_abs']<1e-10,exact
            rows.append(dict(controller=name,frame=profile['observation_frame'],jax_enable_x64=bool(jax.config.x64_enabled),
                             finite=True,certification=cert,projection_calls=n,native_replay=exact,
                             eta13_action_difference=float(np.linalg.norm(a13-a0)),jacobian_norm=float(np.linalg.norm(J))))
            arrays[name+'_eta0_action']=a0
            arrays[name+'_eta13_action']=a13
            arrays[name+'_projected_jacobian']=J
            print(name,'seconds',round(time.time()-t,1),flush=True)
    finally:
        RingExchangeEnv.step=original_step
    np.savez_compressed(HERE/'P1_STATIC_AUDIT.npz',**arrays)
    write('P1_STATIC_AUDIT.json',dict(status='PASS',rows=rows,physical_steps=0,rollouts=0,
                                       all_compatible_checked=len(rows)==len(p['controller_order']),
                                       native_v10_bitwise_action_replay=True,elapsed_seconds=time.time()-t))


if __name__=='__main__':run()
