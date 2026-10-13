"""Check P1 adapter validation R3/R4 versus exact native v10 static runtime."""
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent

import jax
import numpy as np
from diagnostics.controller_q_diversity_v1.runtime import native_runtime
from diagnostics.controller_q_transfer_round3_v1.p1_features import feature
from ring_exchange.environment import RingExchangeEnv

def main():
    jax.config.update('jax_enable_x64',False)
    name='base_u_v10_local_macflow';p=json.loads((ROOT/'diagnostics/controller_q_transfer_round3_v1/P1_EXECUTION_PROTOCOL.json').read_text())
    old=np.load(ROOT/'diagnostics/controller_q_transfer_round3_v1/p1_features'/f'{name}.npz')
    assert list(old['state_uids'])==[s['state_uid'] for s in p['states']]
    rt=native_runtime('ring_exchange');original=RingExchangeEnv.step
    def forbidden(*a,**kw):raise RuntimeError('no env.step in feature audit')
    RingExchangeEnv.step=forbidden
    r3=[];r4=[]
    try:
        for s in p['states']:
            a,b=feature(rt,s);r3.append(a);r4.append(b)
    finally:RingExchangeEnv.step=original
    r3=np.asarray(r3);r4=np.asarray(r4)
    diff3=float(np.max(np.abs(r3-old['R3'])));diff4=float(np.max(np.abs(r4-old['R4_action'])))
    dest=HERE/'static_features'/f'{name}_validation.npz'
    with np.load(dest) as prior:openloop=np.asarray(prior['R4_openloop'])
    np.savez_compressed(dest,R3=r3,R4_action=r4,R4_openloop=openloop,state_uids=old['state_uids'])
    result=dict(controller=name,states=24,old_feature_source='P1 adapter protocol',new_feature_source='P1 exact native execution runtime',
        max_abs_R3_difference=diff3,max_abs_R4_action_difference=diff4,replaced_this_study_validation_features=True,
        physical_steps=0,jax_enable_x64=False)
    (HERE/'NATIVE_VALIDATION_FEATURE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
