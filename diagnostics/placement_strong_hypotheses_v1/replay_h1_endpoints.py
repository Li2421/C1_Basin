"""Recover continuous margins from immutable Q16 endpoint action traces."""
import json
import os
import sqlite3
import sys
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
sys.path.insert(0,str(ROOT/'diagnostics/placement_boundary_mechanism_audit_v1'))
from closed_loop_probe import margin
from validation_stage2.run_rollouts import make_runtime
import numpy as np


def main():
    protocol=json.loads((HERE/'h1_line_protocol.json').read_text())
    states={s['uid']:s for s in json.loads((HARD/'confirmation_manifest.json').read_text())['states']}
    con=sqlite3.connect(HARD/'rollout.sqlite')
    rt=make_runtime('ring_exchange');out=[]
    for line in protocol['lines']:
        if not line['available']:continue
        uid=line['uid']
        for endpoint,index in enumerate(line['candidate_indices']):
            for chain in ('TT','FF'):
                cases=[]
                for (blob,) in con.execute('SELECT result_json FROM rollout WHERE state_uid=? AND chain=? AND eta_index=?',(uid,chain,index)):
                    r=json.loads(blob)
                    if r['seed']==0:cases.append(r)
                if len(cases)!=1:raise RuntimeError((uid,index,chain,len(cases)))
                record=cases[0];path=HARD/'results'/(record['id']+'.npz')
                with np.load(path) as z:
                    actions=z['actions'].copy();positions=z['positions'].copy()
                env=rt.make_env(states[uid]);gm=[];sm=[];term=None
                for a in actions:
                    _,_,done,info=env.step(a)
                    g,s=margin('ring_exchange',env,info);gm.append(g);sm.append(s)
                    term=info['termination']
                    if done:break
                if len(gm)!=len(actions) or np.max(np.abs(env.positions-positions[-1]))>1e-10:
                    raise RuntimeError('cached replay mismatch '+str(path))
                rho=min(max(gm),min(sm));success=bool(env.summary()['collision_free_success'] and term=='success')
                if success!=record['success']:raise RuntimeError('cached binary mismatch')
                out.append(dict(uid=uid,chain=chain,fraction=float(endpoint),candidate_index=index,
                                source_case_id=record['id'],rho=float(rho),success=success,termination=term,
                                goal_ever_margin=float(max(gm)),safety_always_margin=float(min(sm)),
                                goal_active_step=int(np.argmax(gm))+1,safety_active_step=int(np.argmin(sm))+1))
    con.close()
    (HERE/'h1_endpoint_replay.json').write_text(json.dumps(out,indent=2)+'\n')
    print(len(out),'cached endpoints replayed')


if __name__=='__main__':main()
