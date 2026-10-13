"""Four-chain fixed-key closed-loop K*S+G spot checks; no new env.step."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ['JAX_PLATFORMS']='cpu';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1';os.environ['XLA_FLAGS']='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
OLD=ROOT/'diagnostics/placement_boundary_mechanism_audit_v1/results_h80'
NEW=HERE/'trajectory_results'
for p in (str(ROOT),str(FIELD),str(FIELD/'source'),str(FIELD/'source/scripts'),str(HERE)):
    sys.path.insert(0,p)
from validation_stage2.run_rollouts import make_runtime
from derivative_validation import evaluate
from reuse_analysis import write_csv,write_json

UIDS=('hard_confirm_toy_1_820013','hard_confirm_ring_1_920013')
CHAINS=('TT','TF','FT','FF')
EPS=.003


def load(uid,ch,variant):
    if ch in ('TT','FF'):
        name=f'{uid}_{ch}_zero_{variant}'
        if variant=='center':name=f'{uid}_{ch}_zero_center'
        folder=OLD
    else:
        name=f'{uid}_{ch}_{variant}';folder=NEW
    path=folder/(name+'.npz')
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}


def main():
    states={s['uid']:s for s in json.loads((HARD/'confirmation_manifest.json').read_text())['states']}
    rows=[]
    for scene in ('toy_give_way','ring_exchange'):
        rt=make_runtime(scene)
        for uid in (u for u in UIDS if states[u]['scenario']==scene):
            for ch in CHAINS:
                base=load(uid,ch,'center');side={}
                for j in range(3):
                    for sign in (-1,1):
                        name=f'e{j}_{sign:+d}_{EPS}' if ch in ('TT','FF') else f'e{j}_{sign}_{EPS}'
                        side[j,sign]=load(uid,ch,name)
                for t in (0,10,40,60):
                    p=base['positions'][t];v=base['velocities'][t]
                    env=rt.make_env(states[uid]);env.positions=p.copy();env.velocities=v.copy();env.step_count=t
                    key=base['noise_keys'][t];x=np.r_[p.ravel(),v.ravel()];n=p.size;de=.0001
                    def setx(xx):env.positions=xx[:n].reshape(p.shape).copy();env.velocities=xx[n:].reshape(v.shape).copy()
                    nominal=evaluate(rt,env,key,np.zeros(3),ch)[0]
                    replay=float(np.linalg.norm(nominal-base['actions'][t].ravel()))
                    cols=[]
                    for j in range(len(x)):
                        w=np.zeros(len(x));w[j]=de
                        setx(x+w);ap=evaluate(rt,env,key,np.zeros(3),ch)[0]
                        setx(x-w);am=evaluate(rt,env,key,np.zeros(3),ch)[0]
                        cols.append((ap-am)/(2*de))
                    setx(x);K=np.stack(cols,axis=1)
                    g=[];s=[];action=[];matching=[]
                    for j in range(3):
                        e=np.zeros(3);e[j]=EPS
                        ap=evaluate(rt,env,key,e,ch)[0]
                        am=evaluate(rt,env,key,-e,ch)[0]
                        g.append((ap-am)/(2*EPS))
                        plus=side[j,1];minus=side[j,-1]
                        s.append(np.r_[((plus['positions'][t]-minus['positions'][t])/(2*EPS)).ravel(),
                                       ((plus['velocities'][t]-minus['velocities'][t])/(2*EPS)).ravel()])
                        action.append(((plus['actions'][t]-minus['actions'][t])/(2*EPS)).ravel())
                        matching.append(np.array_equal(plus['noise_keys'][t],minus['noise_keys'][t]) and
                                        np.array_equal(plus['noise_keys'][t],key))
                    G=np.stack(g,axis=1);S=np.stack(s,axis=1);observed=np.stack(action,axis=1)
                    predicted=K@S+G
                    residual=float(np.linalg.norm(predicted-observed));rel=residual/max(float(np.linalg.norm(observed)),1e-12)
                    obs_next=[]
                    for j in range(3):
                        plus=side[j,1];minus=side[j,-1]
                        obs_next.append(np.r_[((plus['positions'][t+1]-minus['positions'][t+1])/(2*EPS)).ravel(),
                                               ((plus['velocities'][t+1]-minus['velocities'][t+1])/(2*EPS)).ravel()])
                    obs_next=np.stack(obs_next,axis=1)
                    pred_next=np.r_[S[:n]+rt.config.dt*predicted,predicted]
                    next_error=float(np.linalg.norm(pred_next-obs_next))
                    rows.append(dict(scene=scene,uid=uid,placement=ch,step=t,eta_epsilon=EPS,state_epsilon=de,
                        action_replay_l2=replay,all_noise_keys_match=all(matching),K_norm=float(np.linalg.norm(K)),
                        G_norm=float(np.linalg.norm(G)),S_norm=float(np.linalg.norm(S)),
                        action_sensitivity_norm=float(np.linalg.norm(observed)),
                        action_absolute_error=residual,action_relative_error=rel,
                        next_state_absolute_error=next_error,
                        next_state_relative_error=next_error/max(float(np.linalg.norm(obs_next)),1e-12)))
                print(uid,ch,'done',flush=True)
    write_csv('recurrence_validation.csv',rows,list(rows[0]))
    write_json('recurrence_summary.json',dict(rows=len(rows),new_rollouts=0,
        max_replay_error=max(r['action_replay_l2'] for r in rows),all_noise_keys_match=all(r['all_noise_keys_match'] for r in rows)))


if __name__=='__main__':main()
