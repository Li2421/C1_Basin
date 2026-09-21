"""Score under saved predictions; execute a fixed branch panel with new noise."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from evaluate_c1_frozen_unseen import ROOT,manifest,prefix
from audit_c1_candidate_coverage import candidates,delta_at,START,K,SHORT,DT
from single_integrator.environment import bounded_nominal
from single_integrator.cbf import CBFConfig,barrier_constraints,project_velocity,CBFSolverError
from single_integrator.evaluate import load_policy
from c1_heldout_safety import check as safety_check

BASE=ROOT/'results/c1_frozen_unseen_64';OUT=ROOT/'results/c1_pretraining_audits/mismatch'
NAMES=['P','P+S','P+g','P+g+S'];SEEDS=[20261001,20261002,20261003,20261004]

def prepare():
    OUT.mkdir(exist_ok=True);(OUT/'rows').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    frozen=json.loads((BASE/'protocol.json').read_text());assert manifest()==frozen['hashes']
    ablation=json.loads((OUT.parent/'ablation.json').read_text());panels=[]
    for s in ablation['states']:
        rows=[json.loads(f.read_text()) for f in (BASE/'traces'/str(s['rid'])).glob('*.json')]
        ranked=sorted([r for r in rows if r['costs'] is not None],key=lambda r:(r['costs']['score'],r['cid']))
        ids={s['selected'][k]['cid'] for k in NAMES}|{'zero'}
        ids.update(ranked[min(rank,len(ranked))-1]['cid'] for rank in [2,10,40,80,120,161])
        panels.append(dict(rid=s['rid'],cids=sorted(ids),selected={k:s['selected'][k]['cid'] for k in NAMES},
            prediction_scores={r['cid']:r['costs']['score'] for r in ranked if r['cid'] in ids}))
    p=dict(seeds=SEEDS,panels=panels,initials=frozen['initials'],library=frozen['library'],hashes=frozen['hashes'],
        scope='All64 states, four independent execution-noise seeds; fixed panel includes four ablation winners, zero, frozen full-score ranks2/10/40/80/120/161.',
        selection='No rescoring or reselection under execution noise; same prefix to5s and original interventions, controller,42.5s endpoint.',
        coupling='Common execution random input across panel candidates within each state/seed; independent of prediction seed.',
        ranking='Conditional top1 and pairwise concordance on this predeclared panel only; no claim to exhaustive161-candidate execution coverage.',
        pilot_gate='Mismatch screen: full success >=90% and no more than5 percentage points below P; gradient audit still required.')
    dest=OUT/'protocol.json'
    if dest.exists():assert json.loads(dest.read_text())==p
    else:dest.write_text(json.dumps(p,indent=2)+'\n')
    print('Frozen mismatch branches',sum(len(s['cids']) for s in panels)*len(SEEDS),flush=True)

def run(policy,panel,seed,candidate,env0,buf0,cfg):
    dest=OUT/'rows'/f"{panel['rid']}_{seed}_{candidate['cid']}.json"
    if dest.exists():return
    env=copy.deepcopy(env0);buf={k:list(v) for k,v in buf0.items()};error=None
    key=jax.random.fold_in(jax.random.PRNGKey(seed),panel['rid'])
    for t in range(START,K):
        if buf['success'][-1] or buf['collision'][-1]:break
        x=env.positions.copy();raw=np.asarray(policy.sample_actions(jnp.asarray(env.observation()[None]),seed=jax.random.fold_in(key,t)))[0]
        A,b,_=barrier_constraints(env.snapshot(),cfg)
        try:
            safe=project_velocity(bounded_nominal(raw,.5),A,b,.5,cfg)[0].reshape(4)
            delta=delta_at(candidate,t);v=safe+delta
            u=project_velocity(v,A,b,.5,cfg)[0].reshape(4) if np.any(delta) else safe
        except CBFSolverError as e:error=str(e);break
        _,_,done,info=env.step(u.reshape(2,2))
        values=dict(positions_before=x,positions_after=env.positions.copy(),candidate=v,applied=u,
            success=info['task_success'],deadlock=info['deadlock'],candidate_deadlock=info['candidate_deadlock'],
            collision=info['wall_collision'] or info['agent_collision'])
        for k,value in values.items():buf[k].append(value)
        if done:break
    z={k:np.asarray(v) for k,v in buf.items()};o=dict(success=bool(np.any(z['success'])),
        deadlock=bool(np.any(z['deadlock'])),collision=bool(np.any(z['collision'])),seconds=len(z['success'])*DT,
        stagnation_seconds=float(z['candidate_deadlock'].sum()*DT),remaining=float(np.linalg.norm(env.positions-env.goals,axis=1).max()))
    row=dict(rid=panel['rid'],seed=seed,cid=candidate['cid'],outcome=o,controller_error=error,safety=safety_check(z,env,cfg))
    dest.write_text(json.dumps(row,indent=2)+'\n')

def worker(index,total):
    p=json.loads((OUT/'protocol.json').read_text());assert manifest()==p['hashes']
    policy,_=load_policy(ROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl');cfg=CBFConfig();started=time.time()
    for i in range(index,len(p['panels']),total):
        panel=p['panels'][i];env,buf=prefix(policy,np.asarray(p['initials'][i]),panel['rid'],cfg)
        for seed in SEEDS:
            for cid in panel['cids']:
                run(policy,panel,seed,next(c for c in p['library'] if c['cid']==cid),env,buf,cfg)
        print('STATE',panel['rid'],'seconds',round(time.time()-started,1),flush=True)
    assert manifest()==p['hashes']

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',type=int);parser.add_argument('--workers',type=int,default=8);a=parser.parse_args()
    if a.prepare:prepare()
    else:worker(a.worker,a.workers)
