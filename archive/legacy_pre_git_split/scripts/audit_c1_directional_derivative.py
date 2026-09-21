"""CPU common-noise path derivative audit on saved development directions."""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.train_deadlock_union import setup,noise,digest
from single_integrator.c1.rollout_early_diagnostic import rollout
from single_integrator.c1.risk.ordered_guidance import trajectory
from single_integrator.environment import GiveWayEnv


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/c1_directional_derivative_cpu_v1')
    parser.add_argument('--scales',type=float,nargs='+',default=[.5,.125,.03125])
    args=parser.parse_args()
    if any(not np.isfinite(h) or h<=0 for h in args.scales):
        raise ValueError('finite positive scales required')
    if jax.default_backend()!='cpu':raise RuntimeError('CPU only')
    source=ROOT/'results/c1_guard_factorial_v1';out=args.output
    if out.exists():raise FileExistsError('new output required')
    protocol=json.loads((source/'protocol.json').read_text())
    diagnostics=json.loads((source/'diagnostics.json').read_text())
    selected=sorted([d for d in diagnostics if d['risk']=='ordered' and d['gradient_norm']>0],key=lambda d:d['rid'])[:3]
    _,field,plant,cbf,_=setup();goals=jnp.asarray(GiveWayEnv(plant).goals)
    kw=dict(dt=plant.dt,max_speed=plant.max_speed,goal_tolerance=plant.goal_tolerance,
            hold_seconds=plant.deadlock_hold_seconds,progress_window_seconds=plant.progress_window_seconds,
            progress_epsilon=plant.progress_epsilon,speed_epsilon_fraction=plant.speed_epsilon_fraction)
    out.mkdir();atomic=lambda name,data:(out/name).write_text(json.dumps(data,indent=2)+'\n')
    atomic('protocol.json',dict(scope='posthoc development path derivative audit, not efficacy or unbiased expectation-gradient proof',
        backend='cpu',selected_rids=[d['rid'] for d in selected],selection='first three nonzero saved ordered directions, no outcome filtering',
        scales=args.scales,cpu_noise='regenerated on CPU; all AD/FD comparisons use identical CPU draws',
        source_script_sha256=digest(Path(__file__)),saved_protocol_sha256=digest(source/'protocol.json')))
    results=[];begun=time.monotonic()
    for d in selected:
        rid=d['rid'];path=source/f'local_params_{rid}.pkl';saved=pickle.loads(path.read_bytes())
        reference=jax.tree_util.tree_map(jnp.asarray,saved['params']['reference'])
        delta=jax.tree_util.tree_map(lambda b,a:jnp.asarray(b)-jnp.asarray(a),saved['params']['ordered_fixed'],saved['params']['reference'])
        draws=[noise(s,50000+rid) for s in protocol['prediction_seeds']]
        def objective(scale):
            phi=jax.tree_util.tree_map(lambda p,v:p+scale*v,reference,delta)
            values=[];counts=[]
            for z in draws:
                terms,tr=rollout(phi,field,jnp.asarray(saved['initial']),z,plant,cbf,intervention_steps=100)
                values.append(trajectory(tr['before'],tr['after'],tr['applied'],goals,tr['alive'],
                    terminal_timeout=terms['timeout'],**kw)['J_live']);counts.append(terms['steps'])
            return jnp.mean(jnp.stack(values)),jnp.stack(counts)
        vg=jax.jit(jax.value_and_grad(objective,has_aux=True));fn=jax.jit(objective)
        (value,counts),derivative=vg(jnp.array(0.));derivative=float(derivative)
        comparisons=[]
        for h in args.scales:
            (plus,cp),(minus,cm)=fn(jnp.array(h)),fn(jnp.array(-h))
            fd=float((plus-minus)/(2*h))
            comparisons.append(dict(scale=h,finite_difference=fd,
                positive_risk=float(plus),negative_risk=float(minus),
                forward_difference=float((plus-value)/h),backward_difference=float((value-minus)/h),
                relative_error=abs(fd-derivative)/max(abs(fd),abs(derivative),1e-12),
                same_sign=bool(fd*derivative>0 or fd==derivative==0),
                positive_steps=np.asarray(cp).tolist(),negative_steps=np.asarray(cm).tolist()))
        result=dict(rid=rid,checkpoint_sha256=digest(path),risk=float(value),base_steps=np.asarray(counts).tolist(),
            autodiff=derivative,comparisons=comparisons)
        results.append(result);atomic('results.json',results)
        print(json.dumps(result),flush=True)
    atomic('complete.json',dict(cases=len(results),elapsed=time.monotonic()-begun,
        scope='measurement complete, no automatic derivative correctness claim'))


if __name__=='__main__':main()
