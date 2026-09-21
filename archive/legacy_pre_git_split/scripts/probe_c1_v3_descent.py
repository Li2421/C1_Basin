"""Compare Adam and Euclidean descent on the same complete C1 objective.

Fixed exploratory directions on an already-seen deadlock; no test tuning.
"""
import argparse,json,sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from single_integrator.c1.train_v3 import setup,noise
from single_integrator.c1.rollout_v3 import rollout
from single_integrator.c1.training.persistence import atomic_save


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    files=sorted((ROOT/'results/c1_four_objectives_multiseed/evaluation/baseline').glob('4*.json'))
    case=next(p for p in files if json.loads(p.read_text())['deadlock'])
    rid,seed=map(int,case.stem.split('_'))
    with np.load(case.with_suffix('.npz')) as z:x=jnp.asarray(z['positions_before'][0])
    params,field,plant,cbf,_=setup()
    draws=jnp.concatenate([noise(20260915,rid)[:100],noise(seed,rid)[100:]])
    fn=jax.jit(lambda phi:rollout(phi,field,x,draws,plant,cbf)[0])
    def terms(phi):
        r=fn(phi);return jnp.array([r['P'],r['g'],r['J_def']])
    value,pullback=jax.vjp(terms,params)
    rows=[]
    for name,cot in [('full',[1.,1.,1.]),('progress',[1.,0.,0.]),('geometry',[0.,1.,0.])]:
        grad=pullback(jnp.asarray(cot))[0];norm=float(optax.global_norm(grad))
        for scale in [1e-5,1e-4]:
            direction=jax.tree_util.tree_map(lambda g:-scale*g/max(norm,1e-30),grad)
            r=fn(optax.apply_updates(params,direction))
            rows.append(dict(direction=name,parameter_norm=scale,gradient_norm=norm,
                values={k:float(v) for k,v in r.items()},
                full_loss_change=float(r['J_live']+r['J_def']-value.sum())))
            print(rows[-1],flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    atomic_save(a.out,dict(case=str(case),base_terms=np.asarray(value).tolist(),rows=rows,
        caveat='Component directions are diagnostic only; any accepted update must reduce the full C1 loss.'))


if __name__=='__main__':main()
