"""CPU parity of differentiable and physical prefix rollouts, not efficacy."""
import json
import argparse
from pathlib import Path
import pickle
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.train_deadlock_union import setup,noise,digest
from single_integrator.c1.rollout_early_diagnostic import rollout
from single_integrator.c1.early_intervention import execute_prefix


def main():
    if jax.default_backend()!='cpu':raise RuntimeError('CPU parity check only')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out
    if out.exists():raise FileExistsError('new output required')
    reference,field,plant,cbf,baseline=setup()
    checkpoint=ROOT/'results/c1_deadlock_union/baseline0_seed0/best_feasible.pkl'
    saved=pickle.loads(checkpoint.read_bytes())
    assert saved['config']['baseline_sha256']==digest(baseline)
    params=jax.tree_util.tree_map(jnp.asarray,saved['params'])
    initial=json.loads((ROOT/'results/c1_early_gradient_v2/protocol.json').read_text())['starts'][0]
    draws=noise(8241850,40000)
    out.mkdir()
    paths=['single_integrator/c1/rollout_early_diagnostic.py','single_integrator/c1/early_intervention.py',
           'single_integrator/c1/evaluate_deadlock_union.py','single_integrator/cbf.py','scripts/check_c1_early_forward.py']
    (out/'protocol.json').write_text(json.dumps(dict(initial=initial,noise_seed=8241850,noise_rid=40000,
        checkpoint_sha256=digest(checkpoint),backend='cpu',sources={p:digest(ROOT/p) for p in paths},
        scope='forward parity only',atol=1e-7,rtol=1e-7),indent=2)+'\n')
    checks=[]
    for steps in [0,100,850]:
        fn=jax.jit(lambda phi:rollout(phi,field,jnp.asarray(initial),draws,plant,cbf,intervention_steps=steps))
        terms,tr=fn(params)
        row,actual=execute_prefix(reference,params,field,initial,draws,plant,cbf,prefix_steps=steps)
        count=int(terms['steps']);assert count==row['steps']
        assert bool(terms['success'])==row['success']
        assert bool(terms['deadlock'])==row['any_deadlock']
        assert bool(terms['timeout'])==row['timeout']
        errors={}; strict_agreement=True
        for a,b in [('before','positions_before'),('after','positions_after'),('applied','applied'),('safe','safe')]:
            x=np.asarray(tr[a])[:count];y=actual[b]
            strict_agreement &= bool(np.allclose(x,y,atol=1e-7,rtol=1e-7))
            errors[a]=float(np.max(np.abs(x-y)))
        checks.append(dict(prefix_steps=steps,episode_steps=count,errors=errors,
                           strict_agreement=strict_agreement,outcome_agreement=True,safety=row['safety']))
        print(json.dumps(checks[-1]),flush=True)
    (out/'complete.json').write_text(json.dumps(dict(passed=all(c['strict_agreement'] for c in checks),
        checks=checks,efficacy_demonstrated=False,note='Measurement completed; original 1e-7 threshold unchanged'),indent=2)+'\n')


if __name__=='__main__':main()
