"""V3 full BPTT on the first numbered, already-seen baseline deadlock."""
import argparse
import json
from pathlib import Path
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from single_integrator.c1.train_v3 import setup,noise
from single_integrator.c1.rollout_v3 import rollout
from single_integrator.c1.training.step_control import guarded_step,finite
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.c1.evaluate_v3 import execute


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    folder=ROOT/'results/c1_four_objectives_multiseed/evaluation/baseline'
    case=next(p for p in sorted(folder.glob('4*.json')) if json.loads(p.read_text())['deadlock'])
    rid,seed=map(int,case.stem.split('_'))
    with np.load(case.with_suffix('.npz')) as z:initial=jnp.asarray(z['positions_before'][0])
    params,field,plant,cbf,_=setup()
    draws=jnp.concatenate([noise(20260915,rid)[:100],noise(seed,rid)[100:]])
    fn=jax.jit(lambda phi:rollout(phi,field,initial,draws,plant,cbf)[0])
    def objective(phi):
        r=fn(phi)
        return r['J_live']+r['J_def'],r
    started=time.monotonic()
    args.out.mkdir(parents=True)
    before,grad=jax.value_and_grad(objective,has_aux=True)(params)
    assert finite((before,grad))
    optimizer=optax.adam(1e-5)
    new,_,after,decision=guarded_step(params,optimizer.init(params),grad,optimizer,objective,before)
    result=dict(case=str(case),scope='Already-seen deadlock; local fixed-lambda diagnostic, not held-out validation',
        before={k:float(v) for k,v in before[1].items()},after={k:float(v) for k,v in after[1].items()},
        gradient_norm=float(optax.global_norm(grad)),step_control=decision)
    atomic_save(args.out/'gradient.json',result)
    print(result,flush=True)
    for name,phi in [('before',params),('after',new)]:
        row,trace=execute(phi,field,initial,draws,plant,cbf)
        result[name+'_execution']=row
        np.savez_compressed(args.out/(name+'.npz'),**trace)
    result['elapsed_seconds']=time.monotonic()-started
    atomic_save(args.out/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
