"""Fresh CPU prefix-equivalence checks for fixed-horizon model forecasts."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.train_deadlock_union import setup,noise
from single_integrator.c1.rollout_event_history import rollout
from single_integrator.c1.evaluate_deadlock_union import execute
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace
from single_integrator.environment import GiveWayEnv


def main():
    if jax.default_backend()!='cpu':raise RuntimeError('CPU only')
    out=ROOT/'results/c1_event_history_prefix_v1'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    params,field,plant,cbf,_=setup();goals=np.asarray(GiveWayEnv(plant).goals)
    rng=np.random.default_rng(2026091866)
    x=rng.uniform(.55,1.05,(3,2));y=rng.uniform(-.025,.025,(3,2))
    starts=np.stack([np.c_[-x[:,0],y[:,0]],np.c_[x[:,1],y[:,1]]],1)
    paths=['single_integrator/c1/rollout_event_history.py','single_integrator/c1/risk/event_history.py',
        'single_integrator/c1/evaluate_deadlock_union.py','single_integrator/cbf.py','scripts/check_c1_event_history.py']
    sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    (out/'protocol.json').write_text(json.dumps(dict(scope='prefix parity only, not efficacy',
        starts=starts.tolist(),seed=82780,rid_offset=62000,atol=1e-5,sources=sources),indent=2)+'\n')
    for p in paths:
        dst=out/'source_snapshot'/p;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/p).read_bytes())
    fn=jax.jit(lambda x,z:rollout(params,field,x,z,plant,cbf))
    results=[]
    for rid,x in enumerate(starts):
        z=noise(82780,62000+rid)
        terms,tr=fn(jnp.asarray(x),z)
        row,actual=execute(params,field,x,z,plant,cbf)
        n=row['steps'];assert int(terms['steps'])==n
        for k,j in [('success','success'),('deadlock','any_deadlock'),('timeout','timeout')]:
            assert bool(terms[k])==row[j]
        errors={}
        for k,j in [('before','positions_before'),('after','positions_after'),('applied','applied'),('safe','safe')]:
            predicted=np.asarray(tr[k])[:n];errors[k]=float(np.max(np.abs(predicted-actual[j])))
            np.testing.assert_allclose(predicted,actual[j],rtol=0.,atol=1e-5)
        np.testing.assert_allclose(float(terms['J_def']),row['J_def'],rtol=0.,atol=1e-7)
        outcome='safe_deadlock' if row['any_deadlock'] else 'success' if row['success'] else 'other_timeout'
        label,_=classify_timeout_trace(dict(max_speed=np.linalg.norm(actual['applied'].reshape(-1,2,2),axis=-1).max(axis=-1),
            goal_errors=np.linalg.norm(actual['positions_after']-goals,axis=-1)),outcome,plant.dt)
        dead=label in ('safe_deadlock','stalled_deadlock');risk=float(terms['J_live'])
        assert (risk>1) if dead else (risk<=1)
        np.savez_compressed(out/f'forecast_{rid}.npz',**{k:np.asarray(v) for k,v in tr.items()})
        np.savez_compressed(out/f'actual_{rid}.npz',**actual)
        result=dict(rid=rid,actual_steps=n,model_steps=len(tr['applied']),outcome=label,
            J_live=risk,errors=errors,safety=row['safety'])
        results.append(result);(out/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(result),flush=True)
    (out/'complete.json').write_text(json.dumps(dict(passed=True,cases=3,efficacy_proven=False),indent=2)+'\n')


if __name__=='__main__':main()
