"""50 executions: signal feasibility of a paired event-probability difference.

No old surrogate, path selection, parameter optimization, or training.
The finite difference is not claimed to be an exact probability gradient.
"""
from pathlib import Path
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.train_deadlock_union import setup,noise,source_hashes
from single_integrator.c1.evaluate_deadlock_union import execute
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace
from single_integrator.environment import GiveWayEnv


def main():
    out=ROOT/'results/c1_event_probability_difference_v1'
    if out.exists():raise FileExistsError(out)
    params,field,plant,cbf,_=setup();goals=np.asarray(GiveWayEnv(plant).goals)
    rng=np.random.default_rng(2026091876)
    x=rng.uniform(.30,.55,(25,2));y=rng.uniform(-.025,.025,(25,2))
    starts=np.stack((np.c_[-x[:,0],y[:,0]],np.c_[x[:,1],y[:,1]]),axis=1)
    direction=np.random.default_rng(2026091877).normal(size=4);direction/=np.linalg.norm(direction)
    delta=jax.tree_util.tree_map(jnp.zeros_like,params)
    delta['params']['zero_initialized_output']['bias']=jnp.asarray(direction)
    h=.0025
    variants={sign:jax.tree_util.tree_map(lambda p,d:p+sign*h*d,params,delta) for sign in (-1,1)}
    sources=source_hashes()
    for name in ('scripts/probe_c1_event_probability_difference.py','single_integrator/c1/evaluate_deadlock_union.py'):
        sources[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    out.mkdir()
    atomic_save(out/'protocol.json',dict(scope='probability-difference signal feasibility only',
        candidate='population probability of original strict/stalled first-event union; exact event upper bound',
        starts=starts.tolist(),initial_seed=2026091876,direction_seed=2026091877,
        noise_seed=82930,rid_offset=67000,bias_direction=direction.tolist(),h=h,
        paired_cases=25,total_full_executions=50,controller='unchanged frozen Flow-BC + residual + two hard projections',
        perturbation='shared output bias, throughout original closed loop; neither optimized nor selected',
        restrictions='No increase of h or rerun on new seeds if weak signal; no training or gradient-usefulness claim.',
        gradient_claim='Only estimates central population finite difference; differentiability and h-bias not certified',
        sources=sources))
    for name in sources:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    rows=[];begun=time.monotonic()
    for rid,initial in enumerate(starts):
        draws=noise(82930,67000+rid)
        for sign in (-1,1):
            row,tr=execute(variants[sign],field,initial,draws,plant,cbf)
            first='safe_deadlock' if row['any_deadlock'] else 'success' if row['success'] else 'other_timeout'
            label,_=classify_timeout_trace(dict(max_speed=np.linalg.norm(tr['applied'].reshape(-1,2,2),axis=-1).max(axis=-1),
                goal_errors=np.linalg.norm(tr['positions_after']-goals,axis=-1)),first,plant.dt)
            row.update(rid=rid,sign=sign,outcome=label,either_deadlock=label in ('safe_deadlock','stalled_deadlock'))
            rows.append(row);np.savez_compressed(out/f'trace_{rid}_{sign}.npz',**tr)
        atomic_save(out/'records.json',rows)
    differences=np.array([int(rows[2*i+1]['either_deadlock'])-int(rows[2*i]['either_deadlock']) for i in range(25)])
    summary=dict(executions=len(rows),paired_cases=25,discordant_pairs=int(np.count_nonzero(differences)),
        signed_event_difference_sum=int(differences.sum()),central_difference=float(differences.mean()/(2*h)),
        sample_standard_error=float(differences.std(ddof=1)/np.sqrt(25)/(2*h)),
        standard_error_warning='Zero discordance gives empirical SE zero but does not establish zero population sensitivity.',
        aggregate={str(sign):{label:sum(r['sign']==sign and r['outcome']==label for r in rows)
            for label in ('safe_deadlock','stalled_deadlock','success','other_timeout')} for sign in (-1,1)},
        elapsed=time.monotonic()-begun,early_prediction_validated=False,gradient_usefulness_validated=False,training_launched=False)
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in sources.items())
    atomic_save(out/'complete.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
