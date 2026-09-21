"""Full500-step BPTT and negative-gradient checks before any pilot."""
import argparse,json,sys,time
from pathlib import Path
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from single_integrator.c1.models import ResidualCorrection
from single_integrator.c1.differentiable_rollout import ResidualFlowField
from single_integrator.c1.joint_frozen_rollout import rollout
from single_integrator.cbf import CBFConfig
from single_integrator.environment import Config
from single_integrator.evaluate import load_policy

OUT=ROOT/'results/c1_jax_training_audit';RIDS=[100000,100003,100034,100058]

def setup():
    jax.config.update('jax_enable_x64',True)
    baseline,_=load_policy(ROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
    model=ResidualCorrection();params=model.init(jax.random.PRNGKey(0),jnp.zeros((1,4)),jnp.zeros((1,1)),jnp.zeros((1,20)))
    params=jax.tree_util.tree_map(lambda x:jnp.asarray(x,jnp.float64),params)
    field=ResidualFlowField(baseline,model);field.baseline_sample=jax.jit(field.baseline_sample);field.correction=jax.jit(field.correction)
    plant=replace(Config(corridor_half_length=1.3),terminate_on_deadlock=False)
    return params,field,plant,CBFConfig()

def noise_for(rid):
    with jax.experimental.disable_x64():
        return jnp.stack([jax.random.normal(jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(20260915 if t<100 else 20260916),rid),t),(1,4))[0] for t in range(500)])

def main(index):
    params,field,plant,cbf=setup();rid=RIDS[index];start=time.time()
    z=np.load(ROOT/f'results/c1_frozen_unseen_64/traces/{rid}/zero.npz');noise=noise_for(rid);initial=jnp.asarray(z['positions_before'][0])
    fun=jax.jit(lambda phi:rollout(phi,field,initial,noise,plant,cbf))
    base,dev,trace=fun(params);T=min(500,len(z['success']))
    parity=float(np.max(np.abs(np.asarray(trace['after'][:T])-z['positions_after'][:T])))
    result=dict(rid=rid,baseline_terms=np.asarray(base).tolist(),baseline_deviation=float(dev),max_baseline_position_error=parity,components=[])
    print('forward',rid,result,flush=True)
    for k,name in enumerate(['P','g','S','full']):
        def objective(phi):
            terms,deviation,_=fun(phi)
            return terms[k]+(deviation if name=='full' else 0.)
        value,grad=jax.value_and_grad(objective)(params);leaves=jax.tree_util.tree_leaves(grad)
        norm=float(jnp.sqrt(sum(jnp.sum(x*x) for x in leaves)));finite=all(np.isfinite(np.asarray(x)).all() for x in leaves)
        row=dict(name=name,value=float(value),gradient_norm=norm,finite=finite,negative_steps=[])
        if finite and norm>1e-14:
            direction=jax.tree_util.tree_map(lambda x:x/norm,grad)
            for step in [1e-5,1e-4,1e-3]:
                shifted=jax.tree_util.tree_map(lambda p,d:p-step*d,params,direction)
                try:
                    terms,deviation,_=fun(shifted);after=float(terms[k]+(deviation if name=='full' else 0.))
                    row['negative_steps'].append(dict(step=step,value=after,change=after-float(value),risk_value=float(terms[k])))
                except Exception as e:row['negative_steps'].append(dict(step=step,error=str(e)))
            # Directional finite differences of the complete rollout.
            values=[]
            for sign in [-1,1]:
                shifted=jax.tree_util.tree_map(lambda p,d:p+sign*1e-5*d,params,direction)
                values.append(float(objective(shifted)))
            row['directional_fd']=(values[1]-values[0])/2e-5;row['directional_ad']=norm
            row['directional_relative_error']=abs(row['directional_fd']-norm)/max(1.,abs(row['directional_fd']),norm)
        result['components'].append(row);(OUT/f'bptt_{rid}.json').write_text(json.dumps(result,indent=2)+'\n');print(name,row,'elapsed',round(time.time()-start,1),flush=True)
    result['elapsed_seconds']=time.time()-start
    (OUT/f'bptt_{rid}.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,default=0);a=p.parse_args();main(a.index)
