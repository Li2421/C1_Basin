"""Exact offline horizon/label audit on fixed, already-seen paired executions.

Scores every 0.05s; no controller runs, learning, label-based sampling or tuning.
Compare both extending P+g support and the historically scanned S endpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from single_integrator.c1.risk.joint_frozen import geometry
from single_integrator.c1.risk.risk_v3 import progress_steps
from single_integrator.c1.differentiable_rollout import barrier_constraints
from single_integrator.cbf import CBFConfig
from single_integrator.environment import Config,GiveWayEnv
from single_integrator.c1.training.persistence import atomic_save

MODELS=['baseline','Pg_seed0']
ENDS=[15.,20.,25.,30.,35.,42.5]
LEGACY_ENDS=[20.,25.,30.,35.,42.5]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--limit',type=int,help='Runtime smoke only; resume without limit for complete audit')
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--shard',type=int,default=0)
    args=parser.parse_args()
    if args.shards<1 or not 0<=args.shard<args.shards:parser.error('invalid shard')
    jax.config.update('jax_enable_x64',True)
    root=ROOT/'results/c1_four_objectives_multiseed/evaluation'
    paths=[p for name in MODELS for p in sorted((root/name).glob('4*.json'))]
    if len(paths)!=384:raise ValueError('expected64 states x3 noises x2 paired models')
    sources=[Path(__file__),ROOT/'single_integrator/c1/risk/joint_frozen.py',
        ROOT/'single_integrator/c1/risk/risk_v3.py',ROOT/'single_integrator/c1/differentiable_rollout.py',
        ROOT/'single_integrator/cbf.py',ROOT/'scripts/audit_c1_joint_witness_risk.py']
    protocol=dict(models=MODELS,initial_states=64,execution_noise_replicas=3,episodes=384,
        ends=ENDS,legacy_wait_endpoints=LEGACY_ENDS,dt=.05,score_start=5.,
        selection='All saved baseline and Pg_seed0 executions; no outcome-based selection',
        scope='Already-seen trajectories; descriptive association, not independent efficacy or probability calibration',
        labels='unresolved_deadlock vs others; any_deadlock; failure; deadlock vs other timeout separately',
        scoring='Exact every action; P+g prefixes and historical P+mean(g+S_two-g*S_two) with unchanged5-15s support',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
        inputs={str(p.relative_to(ROOT)):{'json':sha(p),'trace':sha(p.with_suffix('.npz'))} for p in paths})
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'scores').mkdir(exist_ok=True)
    protocol_path=args.out/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=protocol:raise ValueError('frozen audit protocol changed')
    else:atomic_save(protocol_path,protocol)
    plant=Config(corridor_half_length=1.3,terminate_on_deadlock=False)
    env=GiveWayEnv(plant);goals=jnp.asarray(env.goals);walls=jnp.asarray(env.walls);cbf=CBFConfig()
    @jax.jit
    def score(before,after,candidate,success):
        p,alive=progress_steps(before,after,goals,success)
        A,b,_=jax.vmap(lambda x:barrier_constraints(x,walls,plant.to_dict(),cbf))(before[100:])
        query=jnp.where(alive[100:,None],candidate[100:],.1)
        g=geometry(query,A,b)
        return p,jnp.where(alive[100:],g,0.),alive
    started=time.monotonic();rows=[]
    for index,path in enumerate(paths):
        if args.limit is not None and index>=args.limit:break
        if index % args.shards != args.shard:continue
        stem=path.parent.name+'_'+path.stem;dest=args.out/'scores'/(stem+'.json')
        if dest.exists():rows.append(json.loads(dest.read_text()));continue
        label=json.loads(path.read_text())
        if label['controller_error'] is not None or label['collision']:
            raise ValueError('source controller/safety failure: do not silently discard')
        with np.load(path.with_suffix('.npz')) as z:trace={k:z[k] for k in z.files}
        n=len(trace['success']);pad=850-n
        if pad<0 or (pad and not trace['success'][-1]):raise ValueError('unexpected censored source')
        before=np.concatenate([trace['positions_before'],np.repeat(trace['positions_after'][-1][None],pad,axis=0)])
        after=np.concatenate([trace['positions_after'],np.repeat(trace['positions_after'][-1][None],pad,axis=0)])
        candidate=np.concatenate([trace['candidate'],np.zeros((pad,4))])
        success=np.r_[trace['success'],np.ones(pad,bool)]
        p,g,alive=map(np.asarray,score(jnp.asarray(before),jnp.asarray(after),jnp.asarray(candidate),jnp.asarray(success)))
        valid=bool(np.isfinite(p).all() and np.isfinite(g).all())
        first=lambda flags: float((np.flatnonzero(flags)[0]+1)*.05) if np.any(flags) else None
        row=dict(model=path.parent.name,rid=label['rid'],execution_seed=label['execution_seed'],
            condition=label['condition'],success=label['success'],deadlock=label['deadlock'],
            any_deadlock=label['any_deadlock'],failure=not label['success'],timeout=label['timeout'],
            stagnation=label['stagnation'],first_deadlock=first(trace['deadlock']),
            first_success=first(trace['success']),valid=valid,undefined_g_frames=int((~np.isfinite(g)).sum()),scores={})
        # Unscorable geometry remains explicit; no imputation or low-risk fill.
        for end in ENDS:
            k=round(end/.05);P=float(p[100:k].mean());G=float(g[:k-100].mean())
            row['scores'].update({f'P_{end:g}':P,f'g_{end:g}':G if np.isfinite(G) else None,
                                 f'Pg_{end:g}':P+G if np.isfinite(G) else None})
        initial=np.linalg.norm(before[0]-env.goals,axis=-1).sum()+1e-5
        distances=np.linalg.norm(after-env.goals,axis=-1)
        d=np.linalg.norm(before[100:300]-env.goals,axis=-1)
        unfinished=1-np.prod(1-expit((d-.08)/.02),axis=-1)
        short=distances[299].sum()/initial
        cs=expit((.005-(d.sum(-1)/initial-short)/((300-np.arange(100,300))*.05))/.00125)
        raw=.05**2/(.05**2+np.sum((candidate[100:300]/.5)**2,axis=-1))
        for end in LEGACY_ENDS:
            last=round(end/.05)-1;late=distances[last].sum()/initial
            cl=expit((.005-(short-late)/(end-15))/.00125)
            S=np.where(alive[100:300],unfinished*raw*(cs*cl+(1-cs)*cl**2),0.)
            full=float(p[100:300].mean()+np.mean(g[:200]+S-g[:200]*S))
            row['scores'][f'legacy_full_L{end:g}']=full if np.isfinite(full) else None
        atomic_save(dest,row)
        np.savez_compressed(args.out/'scores'/(stem+'.npz'),P=p,g=g,alive=alive)
        rows.append(row)
        if (index+1)%8==0:print(dict(completed=index+1,total=len(paths),elapsed=round(time.monotonic()-started,1)),flush=True)
    all_rows=[json.loads(p.read_text()) for p in sorted((args.out/'scores').glob('*.json'))]
    if len(all_rows)==len(paths):
        atomic_save(args.out/'rows.json',all_rows)
        rows=all_rows
    else:
        atomic_save(args.out/f'partial_{args.shard}.json',rows)
    if len(rows)==len(paths):atomic_save(args.out/'complete.json',dict(episodes=len(rows),undefined_episodes=sum(not r['valid'] for r in rows)))
    print(dict(scored=len(rows),elapsed=time.monotonic()-started),flush=True)


if __name__=='__main__':main()
