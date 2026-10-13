"""Source adapters/feature definitions reused; env.step forbidden for R3/R4."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from common import *
import numpy as np,time

def run(index):
    from blend_runtime import make
    from diagnostics.controller_q_transfer_round3_v1.p1_features import feature
    from stage0 import feature_path
    from ring_exchange.environment import RingExchangeEnv
    import jax
    folder=HERE/'stage_2';p=read(folder/'protocol.json');names=p['anchor_order']+p['controller_order'];name=names[index]
    assert read(HERE/'STAGE_1_GATE.json')['passed']
    assert sha(HERE/'blend_runtime.py')==p['runtime_sha256']
    jax.config.update('jax_enable_x64',False)
    d=folder/'static_features';d.mkdir(exist_ok=True);dest=d/f'{name}.npz'
    if dest.exists():return
    start=time.time();old=RingExchangeEnv.step
    def forbidden(*args,**kwargs):raise RuntimeError('static feature called env.step')
    RingExchangeEnv.step=forbidden;errors=[]
    try:
        rt=make(name,p,{})
        reused={}
        if name in p['anchor_order']:
            z=np.load(feature_path(name));reused={str(s):i for i,s in enumerate(z['state_uids'])}
        rows=[]
        for i,s in enumerate(p['states']):
            try:
                if s['state_uid'] in reused:
                    j=reused[s['state_uid']];a,b=z['R3'][j],z['R4_action'][j]
                else:a,b=feature(rt,s)
                assert a.shape==(256,) and b.shape==(64,) and np.isfinite(a).all() and np.isfinite(b).all()
            except Exception as exc:
                errors.append(dict(state_index=i,state_uid=s['state_uid'],type=type(exc).__name__,message=str(exc)[:500]));a=np.full(256,np.nan,np.float32);b=np.full(64,np.nan,np.float32)
            rows.append((a,b))
            if (i+1)%16==0:print(json.dumps(dict(controller=name,states=i+1,seconds=time.time()-start)),flush=True)
        tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,R3=np.array([r[0] for r in rows]),R4_action=np.array([r[1] for r in rows]),state_uids=np.array([s['state_uid'] for s in p['states']]));tmp.replace(dest)
    finally:RingExchangeEnv.step=old
    rec=dict(controller=name,states=len(p['states']),reused_static_state_count=len(set(reused)&{s['state_uid'] for s in p['states']}),physical_steps=0,outcome_reads=0,errors=errors,jax_enable_x64=False,seconds=time.time()-start,feature_code_sha256=sha(__file__),runtime_sha256=p['runtime_sha256'])
    write(d/f'{name}.receipt.json',rec);assert not errors,rec
if __name__=='__main__':run(int(sys.argv[1]))
