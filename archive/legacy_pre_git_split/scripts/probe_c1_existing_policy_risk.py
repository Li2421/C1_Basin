"""Development-only feasibility witness using the previously trained C1 policy."""
import json
from pathlib import Path
import pickle
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from single_integrator.c1.train_deadlock_union import setup,noise,digest
from single_integrator.c1.rollout_deadlock_union import rollout
from single_integrator.c1.risk.exact_margin import trajectory as exact
from single_integrator.c1.risk.ordered_guidance import trajectory as ordered
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.environment import GiveWayEnv


def main():
    current=ROOT/'results/c1_exact_margin_v1'
    if json.loads((current/'complete.json').read_text())['status']!='no_validation_feasible_checkpoint':return
    path=ROOT/'results/c1_deadlock_union/baseline0_seed0/best_feasible.pkl'
    saved=pickle.loads(path.read_bytes())
    _,field,plant,cbf,baseline=setup()
    if saved['config']['baseline_sha256']!=digest(baseline):raise ValueError('baseline mismatch')
    for name,sha in saved['config']['source_hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('frozen prior source mismatch: '+name)
    params=jax.tree_util.tree_map(jnp.asarray,saved['params'])
    goals=jnp.asarray(GiveWayEnv(plant).goals)
    kw=dict(dt=plant.dt,max_speed=plant.max_speed,goal_tolerance=plant.goal_tolerance,
        hold_seconds=plant.deadlock_hold_seconds,progress_window_seconds=plant.progress_window_seconds,
        progress_epsilon=plant.progress_epsilon,speed_epsilon_fraction=plant.speed_epsilon_fraction)
    @jax.jit
    def fn(x,n):
        v,t=rollout(params,field,x,n,plant,cbf)
        args=(t['before'],t['after'],t['applied'],goals,t['alive'])
        return dict(old=v['J_live'],exact=exact(*args,terminal_timeout=v['timeout'],**kw)['J_live'],
            ordered=ordered(*args,terminal_timeout=v['timeout'],**kw)['J_live'],
            deadlock=v['either_deadlock'],success=v['success'])
    data=json.loads((current/'sets.json').read_text())
    rows=[];started=time.monotonic()
    for item in data['validation']:
        for seed in data['validation_noise_seeds']:
            r=fn(jnp.asarray(item['initial']),noise(seed,item['rid']))
            rows.append(dict(rid=item['rid'],noise_seed=seed,**{k:float(v) for k,v in r.items()}))
    result=dict(scope='Prior trained policy, current development validation only; not a newly trained policy or independent test',
        checkpoint_sha256=digest(path),episodes=len(rows),
        means={k:float(np.mean([r[k] for r in rows])) for k in ('old','exact','ordered','deadlock','success')},
        rows=rows,seconds=time.monotonic()-started)
    atomic_save(current/'prior_policy_witness.json',result)
    print({k:v for k,v in result.items() if k!='rows'})


if __name__=='__main__':main()
