"""Actual signed directional replay at an already-seen deadlock branch."""
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
    def loss(phi):
        r=fn(phi);return r['J_live']+r['J_def']
    value,grad=jax.value_and_grad(loss)(params)
    opt=optax.adam(1e-5);direction,_=opt.update(grad,opt.init(params),params)
    ad=float(sum(jnp.sum(g*d) for g,d in zip(jax.tree_util.tree_leaves(grad),jax.tree_util.tree_leaves(direction))))
    rows=[]
    for scale in [1.,.125,.00390625]:
        pair={}
        for sign in [-1,1]:
            phi=optax.apply_updates(params,jax.tree_util.tree_map(lambda d:sign*scale*d,direction))
            r=fn(phi);pair[str(sign)]={k:float(v) for k,v in r.items()}
        left=pair['-1']['J_live']+pair['-1']['J_def'];right=pair['1']['J_live']+pair['1']['J_def']
        row=dict(scale=scale,values=pair,central_fd=(right-left)/(2*scale),
            forward_slope=(right-float(value))/scale,reverse_slope=(left-float(value))/scale)
        rows.append(row);print(row,flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    atomic_save(a.out,dict(case=str(case),value=float(value),AD_along_adam=ad,rows=rows,
        scope='One nonsmooth parameter direction; not a global gradient correctness verdict'))


if __name__=='__main__':main()
