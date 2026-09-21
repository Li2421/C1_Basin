"""Held-out actual execution, fixed matched and independent random inputs."""
import argparse,json,pickle
import numpy as np
import jax
import jax.numpy as jnp
from train_c1_four_objectives import ROOT,OUT,ARMS,initialize,hashes,atomic
from single_integrator.environment import GiveWayEnv,bounded_nominal
from single_integrator.cbf import barrier_constraints,project_velocity,CBFSolverError
from c1_heldout_safety import check

def main(arm,seed):
    p=json.loads((OUT/'protocol.json').read_text());assert hashes()==p['source_hashes']
    params,field,plant,cfg=initialize(seed);name='baseline' if arm=='baseline' else f'{arm}_seed{seed}'
    if arm!='baseline':params=jax.tree_util.tree_map(jnp.asarray,pickle.loads((OUT/'runs'/name/'params.pkl').read_bytes()))
    dest=OUT/'evaluation'/name;dest.mkdir(exist_ok=True);rows=[]
    for rid,initial in zip(p['test_ids'],p['test_initials']):
        for exseed in [p['matched_execution_seed']]+p['independent_execution_seeds']:
            path=dest/f'{rid}_{exseed}.json'
            if path.exists():rows.append(json.loads(path.read_text()));continue
            env=GiveWayEnv(plant);env.reset(np.asarray(initial));buf={k:[] for k in ['positions_before','positions_after','candidate','applied','success','deadlock','candidate_deadlock','collision']};error=None
            for t in range(850):
                x=env.positions.copy()
                with jax.experimental.disable_x64():
                    key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(p['prefix_seed'] if t<100 else exseed),rid),t)
                    raw=np.asarray(field.baseline.sample_actions(jnp.asarray(env.observation()[None]),seed=key))[0]
                A,b,_=barrier_constraints(env.snapshot(),cfg)
                try:
                    safe=project_velocity(bounded_nominal(raw,.5),A,b,.5,cfg)[0].reshape(4)
                    correction=np.asarray(field.correction(params,jnp.asarray(env.observation()[None]),jnp.asarray(safe[None])))[0] if t>=100 and arm!='baseline' else np.zeros(4)
                    v=safe+correction;u=project_velocity(v,A,b,.5,cfg)[0].reshape(4) if np.any(correction) else safe
                except CBFSolverError as e:error=str(e);break
                _,_,done,info=env.step(u.reshape(2,2))
                values=dict(positions_before=x,positions_after=env.positions.copy(),candidate=v,applied=u,success=info['task_success'],deadlock=info['deadlock'],
                    candidate_deadlock=info['candidate_deadlock'],collision=info['wall_collision'] or info['agent_collision'])
                for k,val in values.items():buf[k].append(val)
                if done:break
            z={k:np.asarray(v) for k,v in buf.items()};success=bool(z['success'].any());deadlock=bool(z['deadlock'].any());collision=bool(z['collision'].any())
            row=dict(arm=arm,training_seed=seed,rid=rid,execution_seed=exseed,condition='matched' if exseed==p['matched_execution_seed'] else 'independent',
                success=success and not collision,deadlock=not success and deadlock,any_deadlock=deadlock,
                timeout=not success and not deadlock and not collision and error is None,collision=collision,controller_error=error,
                stagnation=float(z['candidate_deadlock'].sum()*.05),seconds=len(z['success'])*.05,safety=check(z,env,cfg))
            np.savez_compressed(path.with_suffix('.npz'),**z);atomic(path,row);rows.append(row)
        print(name,'evaluated',rid,'episodes',len(rows),flush=True)
    atomic(dest/'complete.json',dict(episodes=len(rows),source_hashes_verified=hashes()==p['source_hashes']))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS+['baseline'],required=True);parser.add_argument('--seed',type=int,default=0);a=parser.parse_args();main(a.arm,a.seed)
