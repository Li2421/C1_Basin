"""Prospectively frozen four-objective, three-seed training experiment."""
import argparse,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax
from audit_c1_joint_bptt import ROOT,setup,noise_for
from single_integrator.c1.joint_frozen_rollout import rollout
from single_integrator.c1.models import ResidualCorrection

OUT=ROOT/'results/c1_four_objectives_multiseed'
ARMS=['P','PS','Pg','full'];SEEDS=[0,1,2];UPDATES=80

def atomic(path,value,binary=False):
    temp=path.with_name(path.name+'.tmp')
    if binary:temp.write_bytes(pickle.dumps(value))
    else:temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)

def hashes():
    paths=['single_integrator/c1/risk/joint_frozen.py','single_integrator/c1/joint_frozen_rollout.py',
        'single_integrator/c1/differentiable_rollout.py','single_integrator/c1/models/residual.py','single_integrator/cbf.py','single_integrator/environment.py',
        'scripts/audit_c1_joint_witness_risk.py','scripts/audit_c1_candidate_coverage.py','scripts/audit_c1_joint_bptt.py',
        'scripts/train_c1_four_objectives.py','scripts/evaluate_c1_four_objectives.py','scripts/c1_heldout_safety.py',
        'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl']
    return {s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in paths}

def prepare():
    OUT.mkdir(exist_ok=True)
    for name in ['runs','logs','evaluation']:(OUT/name).mkdir(exist_ok=True)
    def draw(seed,n):
        rng=np.random.default_rng(seed);x=rng.uniform(.55,1.05,n);y=rng.uniform(-.025,.025,(n,2))
        return np.stack([np.c_[-x,y[:,0]],np.c_[x,y[:,1]]],axis=1)
    train=draw(20261029,32);test=draw(20261030,64)
    assert np.min(np.max(np.abs(train[:,None]-test[None]),axis=(2,3)))>0
    previous=json.loads((ROOT/'results/c1_frozen_unseen_64/protocol.json').read_text())['initials']
    oldtrain=json.loads((ROOT/'results/c1_jax_training_audit/pilot/protocol.json').read_text())['initials']
    prior=np.asarray(previous+oldtrain)
    assert np.min(np.max(np.abs(test[:,None]-prior[None]),axis=(2,3)))>0
    schedules={}
    for seed in SEEDS:
        rng=np.random.default_rng(seed);schedules[str(seed)]=np.concatenate([rng.permutation(32) for _ in range(3)])[:UPDATES].tolist()
    p=dict(arms=ARMS,training_seeds=SEEDS,scheduled_updates=UPDATES,train_ids=list(range(300000,300032)),test_ids=list(range(400000,400064)),
        train_initials=train.tolist(),test_initials=test.tolist(),training_data_seed=20261029,test_data_seed=20261030,schedules=schedules,
        formulas={'P':'P','PS':'P+mean(Stwo)','Pg':'P+mean(g)','full':'P+mean(g+Stwo-g*Stwo)'},
        loss='same-state J_def + risk; same fixed multiplier1 in every arm',
        optimizer=dict(name='Adam',learning_rate=1e-5,batch_size=1,max_backtracks=8,acceptance_tolerance=1e-10),
        architecture=[256,256,256],prefix_seconds=5,prediction_end_seconds=25,score_window=[5,15],execution_end_seconds=42.5,
        prefix_seed=20260915,matched_execution_seed=20260916,independent_execution_seeds=[20261001,20261002],
        matching='Same32 train states and frozen rollout noise in all arms/seeds; same initialization and sample order within each training seed.',
        failures='Invalid base rollout/nonfinite gradient consumes scheduled attempt, leaves params/Adam unchanged; no replacement data. Invalid proposals rejected using same8-halving rule. Explicit counts retained.',
        checkpoint='Last state after80 scheduled updates; no validation-based checkpoint selection or tuning.',
        evaluation='64 fresh train-disjoint states; matched plus two independent noise realizations; all arms paired within training seed.',
        source_hashes=hashes())
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==p
    else:atomic(OUT/'protocol.json',p)
    print('Frozen12 runs x80 scheduled updates;64 test states x3 execution seeds.',flush=True)

def initialize(seed):
    params,field,plant,cbf=setup()
    model=ResidualCorrection();params=model.init(jax.random.PRNGKey(seed),jnp.zeros((1,4)),jnp.zeros((1,1)),jnp.zeros((1,20)))
    return jax.tree_util.tree_map(lambda x:jnp.asarray(x,jnp.float64),params),field,plant,cbf

def risk(terms,arm):
    return {'P':terms[0],'PS':terms[0]+terms[2],'Pg':terms[0]+terms[1],'full':terms[3]}[arm]

def train(arm,seed):
    p=json.loads((OUT/'protocol.json').read_text());assert hashes()==p['source_hashes']
    dest=OUT/'runs'/f'{arm}_seed{seed}';dest.mkdir(exist_ok=True)
    params,field,plant,cbf=initialize(seed);optimizer=optax.adam(1e-5);state=optimizer.init(params);history=[]
    fingerprint=hashlib.sha256(b''.join(np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(params))).hexdigest()
    if (dest/'checkpoint.pkl').exists():
        saved=pickle.loads((dest/'checkpoint.pkl').read_bytes());params=jax.tree_util.tree_map(jnp.asarray,saved['params']);state=jax.tree_util.tree_map(jnp.asarray,saved['optimizer']);history=saved['history']
    initials=jnp.asarray(p['train_initials']);noises=jnp.stack([noise_for(rid) for rid in p['train_ids']])
    fn=jax.jit(lambda phi,x,n:rollout(phi,field,x,n,plant,cbf)[:2])
    def objective(phi,x,n):
        terms,dev=fn(phi,x,n);return risk(terms,arm)+dev,(terms,dev)
    vg=jax.value_and_grad(objective,has_aux=True);start=time.time()
    for step in range(len(history),UPDATES):
        i=p['schedules'][str(seed)][step];row=dict(step=step,rid=p['train_ids'][i],accepted=False,proposal_errors=[],base_error=None)
        try:
            (loss,(terms,dev)),grad=vg(params,initials[i],noises[i]);norm=float(optax.global_norm(grad))
            if not np.isfinite(norm) or not np.isfinite(float(loss)):raise FloatingPointError('nonfinite loss/gradient')
            row.update(pre=float(loss),terms=np.asarray(terms).tolist(),deviation=float(dev),gradient_norm=norm)
            update,newstate=optimizer.update(grad,state,params)
            for backtrack in range(9):
                candidate=optax.apply_updates(params,jax.tree_util.tree_map(lambda u:u*2.**(-backtrack),update))
                try:
                    value,_=objective(candidate,initials[i],noises[i]);value=float(value)
                    if not np.isfinite(value):raise FloatingPointError('nonfinite proposal')
                    if value<=float(loss)+1e-10:
                        params=candidate;state=newstate;row.update(accepted=True,post=value,backtracks=backtrack);break
                except Exception as e:row['proposal_errors'].append(dict(backtrack=backtrack,error=str(e)))
            if not row['accepted']:row.update(post=float(loss),backtracks=8)
        except Exception as e:row['base_error']=str(e)
        history.append(row)
        atomic(dest/'checkpoint.pkl',dict(params=jax.tree_util.tree_map(np.asarray,params),optimizer=jax.tree_util.tree_map(np.asarray,state),history=history),True)
        atomic(dest/'history.json',history)
        print(arm,seed,'step',step+1,'accepted',row['accepted'],'pre',row.get('pre'),'post',row.get('post'),'elapsed',round(time.time()-start,1),flush=True)
    atomic(dest/'params.pkl',jax.tree_util.tree_map(np.asarray,params),True)
    atomic(dest/'complete.json',dict(arm=arm,seed=seed,scheduled=len(history),accepted=sum(r['accepted'] for r in history),
        base_errors=sum(r['base_error'] is not None for r in history),initial_parameter_hash=fingerprint,
        final_parameter_hash=hashlib.sha256((dest/'params.pkl').read_bytes()).hexdigest(),source_hashes_verified=hashes()==p['source_hashes']))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--arm',choices=ARMS);parser.add_argument('--seed',type=int);a=parser.parse_args()
    if a.prepare:prepare()
    else:train(a.arm,a.seed)
