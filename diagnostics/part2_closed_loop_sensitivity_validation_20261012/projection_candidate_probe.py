"""Strictly active P/Q test at frozen actual candidate eta[4] TT targets."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ['JAX_PLATFORMS']='cpu';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
for p in (str(ROOT),str(FIELD),str(FIELD/'source'),str(FIELD/'source/scripts'),str(HERE)):sys.path.insert(0,p)
from validation_stage2.run_rollouts import make_runtime
from derivative_validation import projection_derivative,geometry_q,stats


def main():
    states={s['uid']:s for s in json.loads((HARD/'confirmation_manifest.json').read_text())['states']}
    eta=np.asarray(json.loads((HARD/'screening_manifest.json').read_text())['eta'][4])
    selected=['hard_confirm_toy_1_820001','hard_confirm_toy_1_820013','hard_confirm_ring_1_920004','hard_confirm_ring_1_920005']
    path=HERE/'jacobian_validation.csv'
    with path.open() as f:
        rd=csv.DictReader(f);rows=list(rd);fields=rd.fieldnames
    rows=[r for r in rows if r['layer'] not in ('P_candidate4','Q_candidate4')]
    new=[]
    for scene in ('toy_give_way','ring_exchange'):
        rt=make_runtime(scene)
        for uid in (s for s in selected if states[s]['scenario']==scene):
            env=rt.make_env(states[uid]);s=rt.safety(env)
            with np.load(HARD/'transfer_mechanism_v2/state_dependence_control'/f'{uid}.npz') as z:
                safe=z['safe_nominal'];B=z['basis']
            target=safe+(B@eta).reshape(safe.shape);output,_=s.project(target);P,d=projection_derivative(s,target,output)
            n=target.size
            for e in (1e-3,3e-4,1e-4):
                pf=[];x0=env.positions.copy();qf=[]
                for j in range(n):
                    v=np.zeros(n);v[j]=e
                    a=s.project((target.ravel()+v).reshape(target.shape))[0]
                    b=s.project((target.ravel()-v).reshape(target.shape))[0]
                    pf.append(((a-b)/(2*e)).ravel())
                Pfd=np.stack(pf,axis=1)
                Q=geometry_q(rt,env,target,s,output,e)
                for j in range(x0.size):
                    v=np.zeros_like(x0);v.ravel()[j]=e
                    env.positions=x0+v;a=rt.safety(env).project(target)[0]
                    env.positions=x0-v;b=rt.safety(env).project(target)[0]
                    env.positions=x0;qf.append(((a-b)/(2*e)).ravel())
                Qfd=np.stack(qf,axis=1)
                for layer,a,b in [('P_candidate4',P,Pfd),('Q_candidate4',Q,Qfd)]:
                    new.append(dict(scene=scene,uid=uid,placement='TT',layer=layer,epsilon=e,
                        active_linear=d['active_linear'],active_speed=d['active_speed'],weak=d['weak'],
                        kkt_residual=d['kkt_residual'],**stats(a,b)))
            print(uid,d,flush=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows+new)
    print('candidate rows',len(new),flush=True)


if __name__=='__main__':main()
