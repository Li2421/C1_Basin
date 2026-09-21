"""Pre-experiment parity checks for direct action interventions; CPU only."""
import json
from pathlib import Path
import pickle
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.train_deadlock_union import setup,noise,digest,source_hashes
from single_integrator.c1.evaluate_deadlock_union import execute
from single_integrator.c1.action_probe import execute_offsets
from single_integrator.c1.rollout_early_diagnostic import rollout
from single_integrator.c1.training.persistence import atomic_save


def main():
    if jax.default_backend()!='cpu': raise RuntimeError('CPU only')
    out=ROOT/'results/c1_action_probe_parity_v1'
    if out.exists(): raise FileExistsError(out)
    reference,field,plant,cbf,baseline=setup()
    checkpoint=ROOT/'results/c1_deadlock_union/baseline0_seed0/best_feasible.pkl'
    saved=pickle.loads(checkpoint.read_bytes())
    assert saved['config']['baseline_sha256']==digest(baseline)
    warm=jax.tree_util.tree_map(jnp.asarray,saved['params'])
    initial=json.loads((ROOT/'results/c1_deadlock_union/sets.json').read_text())['pools']['train'][0]['initial']
    draws=noise(8241801,0)
    perturbation=np.random.default_rng(2026091849).normal(size=(100,4))
    perturbation*=.0025/np.sqrt(np.mean(perturbation**2))
    out.mkdir()
    sources=source_hashes()
    for p in ['single_integrator/c1/action_probe.py','single_integrator/c1/rollout_early_diagnostic.py',
              'single_integrator/c1/evaluate_deadlock_union.py','scripts/check_c1_action_probe.py']:
        sources[p]=digest(ROOT/p)
    atomic_save(out/'protocol.json',dict(scope='interface parity, not efficacy',backend='cpu',
        initial=initial,seed=8241801,rid=0,warm_checkpoint_sha256=digest(checkpoint),sources=sources,
        zero_wrapper_tolerance=0.,forward_atol=1e-5,forward_rtol=0.,
        tolerance_reason='0.4% of planned 0.0025 m/s perturbation; accounts for documented numerical solver/observation differences, not exact arithmetic equivalence'))
    for p in sources:
        dest=out/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())
    forward=jax.jit(lambda phi,offsets:rollout(phi,field,jnp.asarray(initial),draws,plant,cbf,
                      intervention_steps=plant.max_steps,action_offsets=offsets))
    results=[];begun=time.monotonic()
    for name,params in [('zero',reference),('warm',warm)]:
        base,base_trace=execute(params,field,initial,draws,plant,cbf)
        for label,offsets in [('zero',np.zeros((100,4))),('random',perturbation)]:
            row,tr=execute_offsets(params,field,initial,draws,plant,cbf,offsets)
            np.savez_compressed(out/f'{name}_{label}_physical.npz',**tr)
            wrapper_equal=None
            if label=='zero':
                wrapper_equal=all(np.array_equal(tr[k],v) for k,v in base_trace.items())
                wrapper_equal &= base=={k:v for k,v in row.items() if k!='direct_action_rms'}
            padded=jnp.asarray(np.pad(offsets,((0,plant.max_steps-len(offsets)),(0,0))))
            terms,predicted=forward(params,padded)
            n=int(terms['steps']);same_count=n==row['steps']
            same_outcome=all(bool(terms[k])==row[j] for k,j in [('success','success'),('deadlock','any_deadlock'),('timeout','timeout')])
            errors={}
            for k,j in [('before','positions_before'),('after','positions_after'),('applied','applied'),('safe','safe')]:
                errors[k]=float(np.max(np.abs(np.asarray(predicted[k])[:n]-tr[j]))) if same_count else None
            passed=bool(same_count and same_outcome and all(e is not None and e<=1e-5 for e in errors.values()) and wrapper_equal is not False)
            result=dict(params=name,offset=label,passed=passed,zero_wrapper_exact=wrapper_equal,
                steps=[n,row['steps']],same_outcome=same_outcome,errors=errors,direct_action_rms=row['direct_action_rms'],safety=row['safety'])
            results.append(result);atomic_save(out/'results.json',results);print(json.dumps(result),flush=True)
    passed=all(r['passed'] for r in results)
    atomic_save(out/'complete.json',dict(passed=passed,elapsed=time.monotonic()-begun,
        efficacy_demonstrated=False,independent_test_opened=False))
    if not passed: raise RuntimeError('parity gate failed')


if __name__=='__main__':main()
