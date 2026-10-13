"""No-step bitwise regression of held placement and source FF/TT programs."""
import json
import os
import sys
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


def main():
    jax.config.update('jax_enable_x64',False)
    p=json.loads((HERE/'P2PRIME_PROTOCOL.json').read_text())
    rt=native_runtime('ring_exchange');jax.config.update('jax_enable_x64',False)
    from ring_exchange.environment import RingExchangeEnv
    original=RingExchangeEnv.step
    def forbidden(*args,**kwargs):raise RuntimeError('static audit called env.step')
    RingExchangeEnv.step=forbidden
    rows=[]
    try:
        s=p['states'][0];key=key_for(s,0,0)
        for name in p['source_controllers']+p['held_controllers']:
            program=make_program(rt,name);env=program.make_env(s);r=[]
            for j in (0,13):
                eta=np.asarray(p['eta'][j]);a,cert,calls=program.action(env,key,eta,'field')
                assert np.isfinite(a).all() and cert['max_violation']<=rt.cbf.feasibility_tol
                direct=None
                if name in ('FF','TT'):
                    b,_,_=rt.action(env,key,eta,'field' if name=='FF' else 'old')
                    direct=float(np.max(np.abs(a-b)));assert direct==0
                if name in ('TF','FT'):
                    from validation_stage2.factorial_controller import action
                    b,_,_,_=action(rt,env,key,eta,name)
                    direct=float(np.max(np.abs(a-b)));assert direct==0
                r.append(dict(eta_index=j,action_norm=float(np.linalg.norm(a)),projection_calls=calls,direct_error=direct))
            rows.append(dict(controller=name,rows=r,jax_enable_x64=bool(jax.config.x64_enabled)))
    finally:
        RingExchangeEnv.step=original
    (HERE/'P2PRIME_STATIC_AUDIT.json').write_text(json.dumps(dict(status='PASS',physical_steps=0,rows=rows),indent=2)+'\n')
    print('P2prime static audit PASS')


if __name__=='__main__':main()
