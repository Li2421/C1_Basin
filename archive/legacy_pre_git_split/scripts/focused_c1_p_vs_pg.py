"""Frozen direct-G_phi P versus P+g multi-seed expansion.

Evaluation deliberately has no candidate library, scoring call, or action
argmin: it applies G_phi directly at every state then the authoritative CBF
projection.  Existing seed0--2 checkpoints are reused verbatim; seed3--5 use
the same fixed train states, optimizer and 80-update budget.
"""
import argparse, hashlib, json, pickle, time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax

from train_c1_four_objectives import ROOT, initialize, noise_for
from single_integrator.c1.joint_frozen_rollout import rollout
from single_integrator.environment import GiveWayEnv, bounded_nominal
from single_integrator.cbf import barrier_constraints, project_velocity, CBFSolverError
from c1_heldout_safety import check as safety_check

SOURCE=ROOT/'results/c1_four_objectives_multiseed'
OUT=ROOT/'results/c1_focused_p_vs_pg_6seeds'
ARMS=['P','Pg']; SEEDS=list(range(6)); EXISTING=[0,1,2]; NEW=[3,4,5]

def atomic(path, value, binary=False):
    temp=path.with_name(path.name+'.tmp')
    if binary: temp.write_bytes(pickle.dumps(value))
    else: temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def hashes():
    files=['single_integrator/c1/risk/joint_frozen.py','single_integrator/c1/joint_frozen_rollout.py',
           'single_integrator/c1/differentiable_rollout.py','single_integrator/c1/models/residual.py',
           'single_integrator/cbf.py','single_integrator/environment.py','scripts/focused_c1_p_vs_pg.py',
           'scripts/c1_heldout_safety.py','scripts/train_c1_four_objectives.py',
           'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl']
    return {f:digest(ROOT/f) for f in files}

def draw(seed,n):
    rng=np.random.default_rng(seed);x=rng.uniform(.55,1.05,n);y=rng.uniform(-.025,.025,(n,2))
    return np.stack([np.c_[-x,y[:,0]],np.c_[x,y[:,1]]],axis=1)

def prepare():
    OUT.mkdir(exist_ok=True)
    for d in ['runs','evaluation','logs']:(OUT/d).mkdir(exist_ok=True)
    old=json.loads((SOURCE/'protocol.json').read_text())
    assert all(digest(ROOT/f)==h for f,h in old['source_hashes'].items())
    test=draw(20261116,128)
    prior=np.asarray(old['train_initials']+old['test_initials']+
      json.loads((ROOT/'results/c1_jax_training_audit/pilot/protocol.json').read_text())['initials']+
      json.loads((ROOT/'results/c1_frozen_unseen_64/protocol.json').read_text())['initials'])
    assert np.min(np.max(np.abs(test[:,None]-prior[None]),axis=(2,3)))>0
    schedules={}
    for seed in NEW:
        rng=np.random.default_rng(seed); schedules[str(seed)]=np.concatenate([rng.permutation(32) for _ in range(3)])[:80].tolist()
    p=dict(arms=ARMS,training_seeds=SEEDS,reused_training_seeds=EXISTING,new_training_seeds=NEW,
      training_source=str(SOURCE),training_source_protocol_sha256=digest(SOURCE/'protocol.json'),
      training_initials=old['train_initials'],training_ids=old['train_ids'],new_schedules=schedules,
      scheduled_updates=80,optimizer=old['optimizer'],architecture=old['architecture'],
      objective={'P':'J_def + P','Pg':'J_def + P + mean(g)'},
      evaluation_ids=list(range(500000,500128)),evaluation_initials=test.tolist(),evaluation_data_seed=20261116,
      matched_execution_seed=20260916,independent_execution_seeds=[20261001,20261002],
      prediction_horizon_seconds=20,execution_end_seconds=42.5,
      direct_policy_evaluation='For each t>=5s: correction=G_phi(observation,safe); v=safe+correction; applied=project_velocity(v). No candidate list, risk score, candidate ranking, or argmin is evaluated at test time.',
      model_selection='Final 80-update checkpoint only; no validation checkpoint selection.',
      source_hashes=hashes())
    path=OUT/'protocol.json'
    if path.exists(): assert json.loads(path.read_text())==p
    else: atomic(path,p)
    print('Frozen P/Pg 6seed expansion:128 fresh test states; direct G_phi evaluation only.',flush=True)

def objective(arm, terms, deviation): return deviation+terms[0]+(terms[1] if arm=='Pg' else 0.)

def source_checkpoint(arm,seed): return SOURCE/'runs'/f'{arm}_seed{seed}'/'params.pkl'

def train(arm,seed):
    p=json.loads((OUT/'protocol.json').read_text());assert hashes()==p['source_hashes']
    dest=OUT/'runs'/f'{arm}_seed{seed}';dest.mkdir(exist_ok=True)
    if seed in EXISTING:
        src=source_checkpoint(arm,seed);assert src.exists()
        result=dict(arm=arm,seed=seed,reused=True,source_checkpoint=str(src),source_sha256=digest(src),
                    source_complete=json.loads((SOURCE/'runs'/f'{arm}_seed{seed}'/'complete.json').read_text()),
                    source_hashes_verified=True)
        atomic(dest/'complete.json',result);return
    params,field,plant,cbf=initialize(seed);optimizer=optax.adam(1e-5);state=optimizer.init(params);history=[]
    if (dest/'checkpoint.pkl').exists():
        saved=pickle.loads((dest/'checkpoint.pkl').read_bytes());params=jax.tree_util.tree_map(jnp.asarray,saved['params']);state=jax.tree_util.tree_map(jnp.asarray,saved['optimizer']);history=saved['history']
    initials=jnp.asarray(p['training_initials']);noises=jnp.stack([noise_for(rid) for rid in p['training_ids']])
    fn=jax.jit(lambda phi,x,n:rollout(phi,field,x,n,plant,cbf)[:2])
    def lossfn(phi,x,n):
        terms,dev=fn(phi,x,n);return objective(arm,terms,dev),(terms,dev)
    vg=jax.value_and_grad(lossfn,has_aux=True);started=time.time()
    for step in range(len(history),80):
        i=p['new_schedules'][str(seed)][step];row=dict(step=step,rid=p['training_ids'][i],accepted=False,base_error=None,proposal_errors=[])
        try:
            (value,(terms,dev)),grad=vg(params,initials[i],noises[i]);norm=float(optax.global_norm(grad))
            if not np.isfinite(float(value)) or not np.isfinite(norm):raise FloatingPointError('nonfinite loss/gradient')
            row.update(pre=float(value),terms=np.asarray(terms).tolist(),deviation=float(dev),gradient_norm=norm)
            update,newstate=optimizer.update(grad,state,params)
            for backtrack in range(9):
                proposal=optax.apply_updates(params,jax.tree_util.tree_map(lambda u:u*2.**(-backtrack),update))
                try:
                    after=float(lossfn(proposal,initials[i],noises[i])[0])
                    if np.isfinite(after) and after<=float(value)+1e-10:
                        params,state=proposal,newstate;row.update(accepted=True,post=after,backtracks=backtrack);break
                except Exception as error:row['proposal_errors'].append(dict(backtrack=backtrack,error=str(error)))
            if not row['accepted']:row.update(post=float(value),backtracks=8)
        except Exception as error:row['base_error']=str(error)
        history.append(row);atomic(dest/'history.json',history)
        atomic(dest/'checkpoint.pkl',dict(params=jax.tree_util.tree_map(np.asarray,params),optimizer=jax.tree_util.tree_map(np.asarray,state),history=history),True)
        print(arm,seed,step+1,row['accepted'],round(time.time()-started,1),flush=True)
    atomic(dest/'params.pkl',jax.tree_util.tree_map(np.asarray,params),True)
    atomic(dest/'complete.json',dict(arm=arm,seed=seed,reused=False,scheduled=80,accepted=sum(r['accepted'] for r in history),
        base_errors=sum(r['base_error'] is not None for r in history),source_hashes_verified=hashes()==p['source_hashes']))

def load_params(arm,seed):
    if seed in EXISTING: return jax.tree_util.tree_map(jnp.asarray,pickle.loads(source_checkpoint(arm,seed).read_bytes()))
    return jax.tree_util.tree_map(jnp.asarray,pickle.loads((OUT/'runs'/f'{arm}_seed{seed}'/'params.pkl').read_bytes()))

def evaluate(arm,seed):
    p=json.loads((OUT/'protocol.json').read_text());assert hashes()==p['source_hashes']
    assert (OUT/'runs'/f'{arm}_seed{seed}'/'complete.json').exists()
    params,field,plant,cfg=initialize(seed);params=load_params(arm,seed);dest=OUT/'evaluation'/f'{arm}_seed{seed}';dest.mkdir(exist_ok=True)
    for rid,initial in zip(p['evaluation_ids'],p['evaluation_initials']):
      for exseed in [p['matched_execution_seed']]+p['independent_execution_seeds']:
        path=dest/f'{rid}_{exseed}.json'
        if path.exists():continue
        env=GiveWayEnv(plant);env.reset(np.asarray(initial));buf={k:[] for k in ['positions_before','positions_after','candidate','applied','success','deadlock','candidate_deadlock','collision']};error=None
        for t in range(850):
            x=env.positions.copy()
            with jax.experimental.disable_x64():
                key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(20260915 if t<100 else exseed),rid),t)
                raw=np.asarray(field.baseline.sample_actions(jnp.asarray(env.observation()[None]),seed=key))[0]
            A,b,_=barrier_constraints(env.snapshot(),cfg)
            try:
                safe=project_velocity(bounded_nominal(raw,.5),A,b,.5,cfg)[0].reshape(4)
                correction=np.asarray(field.correction(params,jnp.asarray(env.observation()[None]),jnp.asarray(safe[None])))[0] if t>=100 else np.zeros(4)
                v=safe+correction;u=project_velocity(v,A,b,.5,cfg)[0].reshape(4) if np.any(correction) else safe
            except CBFSolverError as exc:error=str(exc);break
            _,_,done,info=env.step(u.reshape(2,2));values=dict(positions_before=x,positions_after=env.positions.copy(),candidate=v,applied=u,
               success=info['task_success'],deadlock=info['deadlock'],candidate_deadlock=info['candidate_deadlock'],collision=info['wall_collision'] or info['agent_collision'])
            for k,value in values.items():buf[k].append(value)
            if done:break
        z={k:np.asarray(v) for k,v in buf.items()};success=bool(z['success'].any());deadlock=bool(z['deadlock'].any());collision=bool(z['collision'].any())
        row=dict(arm=arm,training_seed=seed,rid=rid,execution_seed=exseed,condition='matched' if exseed==p['matched_execution_seed'] else 'independent',
            direct_G_phi=True,candidate_search=False,success=success and not collision,deadlock=not success and deadlock,any_deadlock=deadlock,
            timeout=not success and not deadlock and not collision and error is None,collision=collision,controller_error=error,
            stagnation=float(z['candidate_deadlock'].sum()*.05),seconds=len(z['success'])*.05,safety=safety_check(z,env,cfg))
        np.savez_compressed(path.with_suffix('.npz'),**z);atomic(path,row)
      print(arm,seed,'evaluated',rid,flush=True)
    atomic(dest/'complete.json',dict(episodes=384,direct_G_phi_only=True,candidate_search=False,source_hashes_verified=hashes()==p['source_hashes']))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--train',action='store_true');parser.add_argument('--evaluate',action='store_true');parser.add_argument('--arm',choices=ARMS);parser.add_argument('--seed',type=int);args=parser.parse_args()
    if args.prepare:prepare()
    elif args.train:train(args.arm,args.seed)
    elif args.evaluate:evaluate(args.arm,args.seed)
