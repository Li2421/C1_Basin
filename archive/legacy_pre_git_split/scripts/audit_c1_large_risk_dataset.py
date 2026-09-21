"""Fresh wide-initial-state fixed-horizon R_risk association experiment.

Independent from the controller/training implementation.  It writes an audit
dataset, then scores only the frozen trajectories it generated.
"""
import argparse, hashlib, json, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from scipy.special import expit

from audit_c1_joint_witness_risk import ROOT, LocalRisk, PROTOCOL, save, clean, auc
from audit_c1_completed_waiting_risk import completion, KAPPA, MAXR
from single_integrator.cbf import CBFConfig, CBFSafetyFilter, barrier_constraints
from single_integrator.environment import Config, GiveWayEnv, bounded_nominal
from single_integrator.evaluate import load_policy

H=42.5; DT=.05; K=850

def starts(n,seed):
    rng=np.random.default_rng(seed)
    x=rng.uniform(.55,1.05,n);y=rng.uniform(-.025,.025,(n,2))
    return np.stack([np.c_[-x,y[:,0]],np.c_[x,y[:,1]]],axis=1)

def rollout(policy,initial,seed,rid,config,cbf):
    env=GiveWayEnv(config);env.reset(initial);key=jax.random.fold_in(jax.random.PRNGKey(seed),rid)
    keys=['positions_before','positions_after','candidate','applied','deadlock','success','goal_errors']
    buf={k:[] for k in keys}
    for t in range(K):
        obs=env.observation();before=env.positions.copy()
        raw=np.asarray(policy.sample_actions(jnp.asarray(obs[None]),seed=jax.random.fold_in(key,t)))[0]
        nom=bounded_nominal(raw,config.max_speed)
        safe=CBFSafetyFilter(cbf)(env.snapshot(),nom).velocity.reshape(4)
        _,_,done,info=env.step(safe.reshape(2,2))
        if done and t<K-1:
            raise RuntimeError(f'unexpected terminal at rid={rid} step={t}: {info["termination"]}')
        vals=dict(positions_before=before,positions_after=env.positions.copy(),candidate=safe,applied=safe,
                  deadlock=info['deadlock'],success=info['task_success'],goal_errors=info['goal_errors'])
        for k in keys:buf[k].append(vals[k])
    return {k:np.asarray(v) for k,v in buf.items()}

def score(z):
    # Exact local geometry is evaluated every second, held piecewise constant.
    ids=np.unique(np.r_[np.arange(0,K,20),K-1]);g=[];stops=[]
    cfg=Config(corridor_half_length=1.3);env=GiveWayEnv(cfg);cbf=CBFConfig()
    for t in ids:
        A,b,_=barrier_constraints(dict(positions=z['positions_before'][t],walls=env.walls,config=cfg.to_dict()),cbf)
        result=LocalRisk(A,b,np.full(2,cfg.max_speed)).score(z['candidate'][t])
        if not result['valid']:raise RuntimeError('zero candidate absent from frozen Safety baseline')
        g.append(result['risk']/MAXR);stops.append(KAPPA**2/(KAPPA**2+np.sum((z['candidate'][t]/cfg.max_speed)**2)))
    lookup=np.searchsorted(ids,np.arange(K),side='right')-1
    g=np.asarray(g)[lookup];stop=np.asarray(stops)[lookup]
    # Score completed task as absorbing at its first true success, even though
    # the fixed-horizon evaluator continues for recovery/event observation.
    first_success=np.flatnonzero(z['success']);finish=int(first_success[0]+1) if len(first_success) else K
    active=np.arange(K)<finish;g[~active]=0;stop[~active]=0
    pos=z['positions_after'];dist=np.linalg.norm(pos-np.array([[1.09,0.],[-1.09,0.]]),axis=-1)
    denom=np.linalg.norm(z['positions_before'][0]-np.array([[1.09,0.],[-1.09,0.]]),axis=-1).sum()+1e-5
    V=np.r_[1.,dist.sum(1)/denom];end=np.arange(1,K+1);start=np.maximum(end-80,0)
    rate=(V[start]-V[end])/((end-start)*DT)
    U=1-np.prod(1-expit((dist-.08)/.02),axis=1);U[~active]=0
    p=U*expit((.005-rate)/.00125);p[~active]=0
    hard=np.where(active,np.where(np.isfinite(g),g,1),0)
    smooth=completion(g,np.sum((z['candidate']/.5)**2,axis=1));smooth[~active]=0
    return dict(hard=float(.5*(hard.mean()+p.mean())),smooth=float(.5*(smooth.mean()+p.mean())),
                geometry=float(smooth.mean()),progress=float(p.mean()),stop=float(stop.mean()),
                first_success=finish*DT if finish<K else None,remaining=float(dist[finish-1].max()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=400);p.add_argument('--seed',type=int,default=20260915)
    p.add_argument('--out',type=Path,default=ROOT/'results/c1_large_risk_association_400');args=p.parse_args();out=args.out
    out.mkdir(parents=True,exist_ok=True);(out/'traces').mkdir(exist_ok=True)
    protocol=dict(n=args.n,seed=args.seed,horizon_seconds=H,starts='x_abs~Uniform(.55,1.05), y_i~Uniform(-.025,.025), signs opposing; no rejection.',
                  control='frozen seed0 Flow-BC + authoritative CBF; exact common-number generator per rollout.',
                  labels='deadlock_event: any environment detector event before first success; unresolved: no success by H; recovered_event: detector event and later success.',
                  score='fixed H, first-success absorbing task cost, joint risk every 1 sec, actual-rate progress each .05 sec.',
                  split='IDs whose id%5==0 heldout, fixed before outcomes; other development.',kappa=KAPPA)
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    else:save(out/'protocol.json',protocol)
    policy,_=load_policy(ROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
    config=replace(Config(corridor_half_length=1.3),terminate_on_success=False,terminate_on_deadlock=False)
    cbf=CBFConfig();initials=starts(args.n,args.seed);rows=[];started=time.perf_counter()
    for rid,initial in enumerate(initials):
        file=out/'traces'/f'{rid:04d}.npz'
        if file.exists():z=np.load(file);data={k:z[k] for k in z.files}
        else:
            data=rollout(policy,initial,args.seed,rid,config,cbf);np.savez_compressed(file,**data,initial=initial)
        s=score(data);fs=np.flatnonzero(data['success']);fd=np.flatnonzero(data['deadlock']);first_success=fs[0]+1 if len(fs) else K+1
        event=bool(len(fd) and fd[0]+1<first_success);recovered=bool(event and len(fs))
        rows.append(dict(rid=rid,split='heldout' if rid%5==0 else 'development',deadlock_event=event,recovered_event=recovered,
                         unresolved=not len(fs),**s))
        if rid%10==0:print(f'{rid}/{args.n}, elapsed={time.perf_counter()-started:.1f}s',flush=True)
        save(out/'rows.json',rows)
    metrics={}
    fields=['hard','smooth','geometry','progress','stop','remaining']
    for split in ['all','development','heldout']:
        rr=[r for r in rows if split=='all' or r['split']==split]
        metrics[split]=dict(n=len(rr),deadlocks=sum(r['deadlock_event'] for r in rr),recovered=sum(r['recovered_event'] for r in rr),unresolved=sum(r['unresolved'] for r in rr),
            deadlock_auc={f:auc([r['deadlock_event'] for r in rr],[r[f] for r in rr]) for f in fields},
            unresolved_auc={f:auc([r['unresolved'] for r in rr],[r[f] for r in rr]) for f in fields})
    rng=np.random.default_rng(args.seed);boot=[]
    for _ in range(1000):
        ii=rng.integers(0,len(rows),len(rows));y=np.array([r['deadlock_event'] for r in rows])[ii]
        if y.any() and (~y).any():boot.append([auc(y,np.array([r[f] for r in rows])[ii]) for f in fields])
    save(out/'metrics.json',dict(metrics=metrics,bootstrap=dict(fields=fields,replicates=len(boot),interval95=np.quantile(boot,[.025,.975],axis=0).T)))
    save(out/'manifest.json',dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=args.n))
    print(json.dumps(clean(metrics),indent=2),flush=True)

if __name__=='__main__':main()
