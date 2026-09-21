"""Actual42.5s matched execution of baseline and both learned residuals."""
import argparse,json,pickle
import numpy as np
import jax
import jax.numpy as jnp
from audit_c1_joint_bptt import ROOT,OUT,setup
from single_integrator.environment import GiveWayEnv,bounded_nominal
from single_integrator.cbf import barrier_constraints,project_velocity,CBFSolverError
from c1_heldout_safety import check

def main(arm):
    zero,field,plant,cfg=setup();dest=OUT/'pilot';p=json.loads((dest/'protocol.json').read_text())
    params=zero if arm=='baseline' else jax.tree_util.tree_map(jnp.asarray,pickle.loads((dest/f'{arm}_params.pkl').read_bytes()))
    original=json.loads((ROOT/'results/c1_frozen_unseen_64/protocol.json').read_text());rows=[]
    for rid in p['evaluation_ids']:
        for seed in p['evaluation_seeds']:
            env=GiveWayEnv(plant);env.reset(np.asarray(original['initials'][original['ids'].index(rid)]));buf={k:[] for k in ['positions_before','positions_after','candidate','applied','success','deadlock','candidate_deadlock','collision']};error=None
            for t in range(850):
                x=env.positions.copy()
                with jax.experimental.disable_x64():
                    key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(20260915 if t<100 else seed),rid),t)
                    raw=np.asarray(field.baseline.sample_actions(jnp.asarray(env.observation()[None]),seed=key))[0]
                A,b,_=barrier_constraints(env.snapshot(),cfg)
                try:
                    safe=project_velocity(bounded_nominal(raw,.5),A,b,.5,cfg)[0].reshape(4)
                    correction=np.asarray(field.correction(params,jnp.asarray(env.observation()[None]),jnp.asarray(safe[None])))[0] if t>=100 else np.zeros(4)
                    v=safe+correction;u=project_velocity(v,A,b,.5,cfg)[0].reshape(4) if np.any(correction) else safe
                except CBFSolverError as e:error=str(e);break
                _,_,done,info=env.step(u.reshape(2,2));values=dict(positions_before=x,positions_after=env.positions.copy(),candidate=v,applied=u,
                    success=info['task_success'],deadlock=info['deadlock'],candidate_deadlock=info['candidate_deadlock'],collision=info['wall_collision'] or info['agent_collision'])
                for k,value in values.items():buf[k].append(value)
                if done:break
            z={k:np.asarray(v) for k,v in buf.items()};success=bool(z['success'].any());deadlock=bool(z['deadlock'].any())
            rows.append(dict(rid=rid,seed=seed,success=success,deadlock=deadlock,timeout=not success and not deadlock and error is None,
                controller_error=error,stagnation=float(z['candidate_deadlock'].sum()*.05),seconds=len(z['success'])*.05,safety=check(z,env,cfg)))
            (dest/f'{arm}_execution.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(arm,rid,seed,success,rows[-1]['stagnation'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['baseline','P','full'],required=True);a=p.parse_args();main(a.arm)
