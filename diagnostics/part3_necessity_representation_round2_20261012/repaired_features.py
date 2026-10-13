"""No-outcome re-query of P1 static policy probes, retaining indexed response."""
from __future__ import annotations
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
P1=ROOT/'diagnostics/controller_q_transfer_round3_v1'

import numpy as np
import jax
from diagnostics.controller_q_diversity_v1.runtime import controller_context
from diagnostics.controller_q_diversity_v1.features import field_context
from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter

def descriptors(nominal,field):
    """Per-agent 64D R3, 238D indexed R5, and sampled raw behavior."""
    f=field.reshape(4,7,3,5,2)
    intercept=f[:,:,:,0,:].reshape(4,42)
    jac=np.stack((f[:,:,:,1,:]-f[:,:,:,2,:],f[:,:,:,3,:]-f[:,:,:,4,:]),axis=-1)
    r3=np.concatenate((nominal,intercept,jac.mean(axis=(1,2)).reshape(4,4),
                       jac.var(axis=(1,2)).reshape(4,4)),axis=-1)
    indexed=jac.reshape(4,84)
    moving=f[:,4:7].reshape(4,90)
    r5=np.concatenate((r3,indexed,moving),axis=-1)
    assert r3.shape==(4,64) and r5.shape==(4,238)
    return r3.astype(np.float32).ravel(),r5.astype(np.float32).ravel()

def main():
    jax.config.update('jax_enable_x64',False)
    p=json.loads((P1/'P1_PROTOCOL.json').read_text())
    from ring_exchange.environment import RingExchangeEnv
    old=RingExchangeEnv.step
    def forbidden(*args,**kwargs):raise RuntimeError('static probe attempted env.step')
    RingExchangeEnv.step=forbidden
    receipts=[];start=time.monotonic()
    try:
        for name in p['controller_order']:
            dest=HERE/'features'/(name+'.npz');dest.parent.mkdir(exist_ok=True)
            if dest.exists():
                with np.load(dest) as prior:assert prior['R5'].shape==(24,952)
                receipts.append(dict(controller=name,reused=True));continue
            profile=p['controllers']['ring_exchange'][name]['payload']
            manifest=json.loads(Path(profile['adapter_manifest_path']).read_text())
            rt=RingFlowAdapter(profile['flow_checkpoint_path'],manifest,profile['observation_frame'])
            r3s=[];r5s=[];raws=[];nominals=[];deltas=[]
            archived=np.load(P1/'p1_features'/(name+'.npz'))
            for idx,state in enumerate(p['states']):
                nominal=controller_context(rt,state)
                query=type('Query',(),{'__getattr__':lambda self,key:getattr(rt,key)})()
                field=field_context(query,state)
                r3,r5=descriptors(nominal,field)
                delta=float(np.max(np.abs(r3-archived['R3'][idx])))
                if delta>1e-4:raise RuntimeError(f'R3 replay mismatch {name} {idx}: {delta}')
                r3s.append(r3);r5s.append(r5);raws.append(field.ravel());nominals.append(nominal.ravel());deltas.append(delta)
            archived.close()
            np.savez_compressed(dest,R3=np.stack(r3s),R5=np.stack(r5s),raw_field=np.stack(raws),
                                nominal=np.stack(nominals),state_uids=np.array([s['state_uid'] for s in p['states']]))
            receipts.append(dict(controller=name,reused=False,max_R3_replay_error=max(deltas),
                                 seconds=time.monotonic()-start,states=24,simulator_steps=0))
            print(json.dumps(receipts[-1]),flush=True)
    finally:
        RingExchangeEnv.step=old
    (HERE/'feature_receipt.json').write_text(json.dumps(dict(rows=receipts,simulator_steps=0,
        outcome_reads=0,total_seconds=time.monotonic()-start),indent=2)+'\n')

if __name__=='__main__':main()
