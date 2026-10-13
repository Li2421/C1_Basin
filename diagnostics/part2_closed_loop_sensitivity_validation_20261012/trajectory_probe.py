"""Preflighted, fixed-key H80 TF/FT continuation sensitivity validation."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from pathlib import Path

os.environ['JAX_PLATFORMS']='cpu'
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['XLA_FLAGS']='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
for path in (str(ROOT),str(FIELD),str(FIELD/'source'),str(FIELD/'source/scripts'),str(ROOT/'diagnostics/placement_boundary_mechanism_audit_v1')):
    sys.path.insert(0,path)

import numpy as np
from shared_rollout_db.src.rollout_db import eta_identity
from validation_stage2.run_rollouts import make_runtime
from closed_loop_probe import simulate
from reuse_analysis import completion_margin,TOL,HORIZON,write_csv,write_json


UIDS=('hard_confirm_toy_1_820013','hard_confirm_ring_1_920013')
CHAINS=('TF','FT')
EPS=(.01,.003)
D=np.ones(3)/np.sqrt(3)


def read(path):return json.loads(Path(path).read_text())


def points():
    z=[dict(name='center',eta=[0.,0.,0.],epsilon=0.,axis=None,sign=None)]
    for e in EPS:
        for j in range(3):
            for sign in (-1,1):
                v=np.zeros(3);v[j]=sign*e
                z.append(dict(name=f'e{j}_{sign}_{e}',eta=v.tolist(),epsilon=e,axis=j,sign=sign))
    z.append(dict(name='heldout_diagonal',eta=(.001*D).tolist(),epsilon=.001,axis='diagonal',sign=1))
    return z


def prepare():
    states={s['uid']:s for s in read(HARD/'confirmation_manifest.json')['states']}
    con=sqlite3.connect(HARD/'rollout.sqlite')
    jobs=[];req=[]
    for uid in UIDS:
        for chain in CHAINS:
            ctl=con.execute('select controller_uid from rollout where state_uid=? and chain=? limit 1',(uid,chain)).fetchone()
            if ctl is None:raise RuntimeError('missing frozen controller identity')
            for p in points():
                job=dict(uid=uid,scene=states[uid]['scenario'],chain=chain,state=states[uid],controller_uid=ctl[0],**p)
                job['eta_uid']=eta_identity(job['eta'])[0]
                jobs.append(job)
                req.append(dict(state_uid=uid,eta_uid=job['eta_uid'],controller_uid=ctl[0],
                    seed_keys=['{"future_root":2026100601,"seed":0}']))
    con.close()
    write_json('trajectory_protocol.json',dict(jobs=jobs,selection='old pre-registered 13th frozen confirmation state per scene',
        noise='future_root 2026100601, seed 0, SHA state token, step fold_in',eta_schedule='fixed episode, B recomputed each physical step',
        horizon=80,new_full_continuations=0,new_short_continuations=len(jobs),
        estimated_physical_steps=len(jobs)*80,resources='CPU only, one process, one thread',
        checkpoint_paths=dict(toy=str(FIELD/'validation_v2/assets/toy_ckpt.pkl'),
                              ring=str(FIELD/'poc/assets/diagnostics/ring_exchange_stage1/base_u_v10_local_macflow/best.pkl'))))
    write_json('planned_rollouts.json',dict(requests=req))
    print('planned',len(jobs),flush=True)


def run():
    pre=read(HERE/'cache_preflight.json')
    if pre['summary'].get('ambiguous',0):raise RuntimeError('AMBIGUOUS preflight')
    # These states/controllers use a historical registry; no exact compatible
    # H80 TF/FT eta-neighborhood evidence was found in old experiment folders.
    jobs=read(HERE/'trajectory_protocol.json')['jobs'];root=read(HARD/'screening_manifest.json')['future_root']
    folder=HERE/'trajectory_results';folder.mkdir(exist_ok=True)
    for scene in ('toy_give_way','ring_exchange'):
        rt=make_runtime(scene)
        for job in (j for j in jobs if j['scene']==scene):
            stem=f"{job['uid']}_{job['chain']}_{job['name']}";f=folder/(stem+'.json');a=folder/(stem+'.npz')
            if f.exists() and a.exists():continue
            arrays,record=simulate(rt,job['state'],job['chain'],job['eta'],0,root,80)
            record.update(protocol_name=job['name'],controller_uid=job['controller_uid'],eta_uid=job['eta_uid'])
            np.savez_compressed(a,**arrays);f.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
            print(stem,record['steps'],record['termination'],flush=True)


def analyze():
    jobs=read(HERE/'trajectory_protocol.json')['jobs'];folder=HERE/'trajectory_results';rows=[];margins=[]
    for uid in UIDS:
        scene=next(j['scene'] for j in jobs if j['uid']==uid)
        for chain in CHAINS:
            def load(name):
                stem=f'{uid}_{chain}_{name}'
                r=read(folder/(stem+'.json'))
                with np.load(folder/(stem+'.npz')) as z:pos=z['positions'].copy();vel=z['velocities'].copy()
                return r,pos,vel
            base,p0,v0=load('center');g=np.asarray(next(j['state']['physical']['goals'] for j in jobs if j['uid']==uid))
            bm=completion_margin(p0,g,TOL[scene],80)
            scale_by_eps={}
            for e in EPS:
                sp=[];sv=[];grad=[];switch=[]
                for axis in range(3):
                    minus,pm,vm=load(f'e{axis}_-1_{e}');plus,pp,vp=load(f'e{axis}_1_{e}')
                    n=min(len(pm),len(pp),len(p0));sp.append((pp[:n]-pm[:n])/(2*e));sv.append((vp[:n]-vm[:n])/(2*e))
                    mm=completion_margin(pm,g,TOL[scene],80);mp=completion_margin(pp,g,TOL[scene],80)
                    grad.append((mp['rho']-mm['rho'])/(2*e))
                    switch.append((mm['active_step'],mm['active_agent'])!=(bm['active_step'],bm['active_agent']) or
                                  (mp['active_step'],mp['active_agent'])!=(bm['active_step'],bm['active_agent']))
                n=min(len(x) for x in sp);s=np.stack([x[:n] for x in sp],axis=-1)
                scale_by_eps[e]=s
                t=bm['active_step'];i=bm['active_agent']
                chain_grad=None
                if t<n:
                    v=(p0[t,i]-g[i])/np.linalg.norm(p0[t,i]-g[i])
                    chain_grad=(-(v@s[t,i])).tolist()
                direct=np.asarray(grad)
                for sample in (1,10,40,80):
                    if sample>=n:continue
                    rows.append(dict(scene=scene,uid=uid,placement=chain,layer='S',epsilon=e,step=sample,
                        norm=float(np.linalg.norm(s[sample])),scale_comparison='',active_switch=any(switch),
                        absolute_error='',relative_error='',cosine='',baseline_termination=base['termination']))
                margins.append(dict(scene=scene,uid=uid,placement=chain,epsilon=e,baseline_rho=bm['rho'],
                    active_step=t,active_agent=i,gradient_direct=direct.tolist(),gradient_chain=chain_grad,
                    gradient_error=float(np.linalg.norm(direct-np.asarray(chain_grad))) if chain_grad is not None else None,
                    active_switch=any(switch)))
                if e==.003:
                    held,ph,vh=load('heldout_diagonal');nn=min(len(ph),len(p0),n)
                    for sample in (1,10,40,80):
                        if sample>=nn:continue
                        actual=(ph[sample]-p0[sample]).ravel();pred=(s[sample].reshape(-1,3)@(.001*D))
                        error=np.linalg.norm(actual-pred)
                        rows.append(dict(scene=scene,uid=uid,placement=chain,layer='S_heldout',epsilon=.001,step=sample,
                            norm=float(np.linalg.norm(actual)),scale_comparison='',active_switch=any(switch),
                            absolute_error=float(error),relative_error=float(error/max(np.linalg.norm(actual),1e-12)),
                            cosine=float(actual@pred/max(np.linalg.norm(actual)*np.linalg.norm(pred),1e-12)),
                            baseline_termination=base['termination']))
                    hm=completion_margin(ph,g,TOL[scene],80);pred_rho=.001*float(direct@D)
                    margins.append(dict(scene=scene,uid=uid,placement=chain,epsilon=.001,baseline_rho=bm['rho'],
                        heldout_rho=hm['rho'],actual_change=hm['rho']-bm['rho'],predicted_change=pred_rho,
                        heldout_abs_error=abs((hm['rho']-bm['rho'])-pred_rho),active_step=t,active_agent=i,
                        active_switch=any(switch)))
            a=scale_by_eps[.01];b=scale_by_eps[.003]
            for sample in (1,10,40,80):
                if sample>=min(len(a),len(b)):continue
                err=float(np.linalg.norm(a[sample]-b[sample]));den=float(np.linalg.norm(b[sample]))
                rows.append(dict(scene=scene,uid=uid,placement=chain,layer='S_scale',epsilon='.01_vs_.003',step=sample,
                    norm=den,scale_comparison='fixed central differences',active_switch='',absolute_error=err,
                    relative_error=err/max(den,1e-12),cosine=float(np.sum(a[sample]*b[sample])/max(np.linalg.norm(a[sample])*den,1e-12)),
                    baseline_termination=base['termination']))
    write_csv('trajectory_validation.csv',rows,list(rows[0]))
    margin_fields=[]
    for row in margins:
        for key in row:
            if key not in margin_fields:margin_fields.append(key)
    write_csv('trajectory_margin_validation.csv',margins,margin_fields)
    print('trajectory rows',len(rows),'margin rows',len(margins),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','analyze']);stage=ap.parse_args().stage
    {'prepare':prepare,'run':run,'analyze':analyze}[stage]()
