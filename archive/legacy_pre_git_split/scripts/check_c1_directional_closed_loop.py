"""Short actual-controller directional audit; no event risk or training."""
from pathlib import Path
import hashlib
import json
import sys
import traceback
import argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.train_deadlock_union import setup,noise,source_hashes
from single_integrator.c1.train import observation
from single_integrator.c1.differentiable_rollout import bounded_nominal,barrier_constraints
from single_integrator.c1.projection_directional import projection_direction
from single_integrator.cbf import project_velocity,barrier_constraints as physical_constraints
from single_integrator.environment import GiveWayEnv
from single_integrator.c1.training.persistence import atomic_save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--out',type=Path,default=ROOT/'results/c1_directional_closed_loop_v1')
    parser.add_argument('--difference-steps',type=float,nargs='+',default=(.01,.003,.001))
    args=parser.parse_args()
    if not 1<=args.steps<=100 or any(not np.isfinite(h) or h<=0 for h in args.difference_steps):
        parser.error('diagnostic limited to 1..100 physical steps and positive finite differences')
    out=args.out
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    params,field,plant,cbf,_=setup()
    env=GiveWayEnv(plant);goals=jnp.asarray(env.goals);walls=jnp.asarray(env.walls)
    tangent=jax.tree_util.tree_map(jnp.zeros_like,params)
    direction=np.random.default_rng(2026091875).normal(size=4);direction/=np.linalg.norm(direction)
    tangent['params']['zero_initialized_output']['bias']=jnp.asarray(direction)
    rng=np.random.default_rng(2026091874)
    xx=rng.uniform(.30,.55,(3,2));yy=rng.uniform(-.025,.025,(3,2))
    starts=np.stack((np.c_[-xx[:,0],yy[:,0]],np.c_[xx[:,1],yy[:,1]]),axis=1)
    steps=args.steps;hs=tuple(args.difference_steps)
    sources=source_hashes()
    for name in ('scripts/check_c1_directional_closed_loop.py','single_integrator/c1/projection_directional.py'):
        sources[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    for name in sources:
        dst=out/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/name).read_bytes())
    atomic_save(out/'protocol.json',dict(scope='short actual-policy directional derivative, not risk validation',
        starts=starts.tolist(),initial_seed=2026091874,direction_seed=2026091875,
        bias_direction=direction.tolist(),steps=steps,noise_seed=82920,rid_offset=66000,
        finite_difference_steps=hs,endpoint_absolute_tolerance=.02,
        endpoint_relative_tolerance=.1,relative_scale_floor=.01,
        main_controller_modified=False,source_sha256=sources))

    def pre(x,v,xi):
        obs=observation(x[None],v[None],goals)
        nominal=bounded_nominal(field.baseline_sample(obs,xi[None]),plant.max_speed)[0]
        A,b,_=barrier_constraints(x,walls,plant.to_dict(),cbf)
        return nominal,A,b,obs
    pre=jax.jit(pre)
    pre_jvp=jax.jit(lambda x,v,xi,dx,dv:jax.jvp(lambda a,c:pre(a,c,xi),(x,v),(dx,dv)))
    corr=jax.jit(lambda phi,obs,safe:field.correction(phi,obs,safe[None])[0])
    corr_jvp=jax.jit(lambda phi,obs,safe,dp,dobs,ds:jax.jvp(corr,(phi,obs,safe),(dp,dobs,ds)))
    solve=lambda y,A,b:project_velocity(np.asarray(y),np.asarray(A),np.asarray(b),plant.max_speed,cbf)[0].reshape(4)

    def propagate(initial,draws,sign):
        env=GiveWayEnv(plant);env.reset(initial);dx=jnp.zeros((2,2));dv=jnp.zeros((2,2))
        dp=jax.tree_util.tree_map(lambda p:sign*p,tangent)
        traces=[];certs=[]
        for t in range(steps):
            (nom,A,b,obs),(dn,dA,db,dobs)=pre_jvp(jnp.asarray(env.positions),jnp.asarray(env.velocities),draws[t],dx,dv)
            pa,pb,_=physical_constraints(env.snapshot(),cbf)
            np.testing.assert_allclose(A,pa,atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(b,pb,atol=1e-12,rtol=1e-12)
            safe=solve(nom,A,b)
            try:
                ds,c1=projection_direction(nom,A,b,safe,dn,dA=dA,db=db)
            except (ValueError,RuntimeError):
                np.savez_compressed(out/'first_projection_failure.npz',y=np.asarray(nom),A=np.asarray(A),b=np.asarray(b),
                    p=safe,direction=np.asarray(dn),dA=np.asarray(dA),db=np.asarray(db),
                    step=t,sign=sign,positions=env.positions,velocity=env.velocities)
                raise
            correction,dc=corr_jvp(params,obs,jnp.asarray(safe),dp,dobs,jnp.asarray(ds))
            candidate=safe+np.asarray(correction);applied=solve(candidate,A,b)
            du,c2=projection_direction(candidate,A,b,applied,ds+np.asarray(dc),dA=dA,db=db)
            dx=dx+plant.dt*du.reshape(2,2);dv=jnp.asarray(du.reshape(2,2))
            _,_,done,info=env.step(applied.reshape(2,2))
            if done:raise RuntimeError('unexpected first event in short diagnostic')
            traces.append((env.positions.copy(),np.asarray(dx),applied,du))
            certs.append(dict(first=c1,second=c2))
        return traces,certs

    def physical(initial,draws,phi):
        env=GiveWayEnv(plant);env.reset(initial);positions=[]
        for t in range(steps):
            A,b,_=physical_constraints(env.snapshot(),cbf)
            prepared=field.prepare(phi,jnp.asarray(env.observation()[None]),draws[t:t+1],A,b,plant.max_speed,
                lambda y,a,c,s:jnp.asarray(solve(y[0],a,c)[None]))
            applied=solve(prepared['candidate'][0],A,b)
            _,_,done,info=env.step(applied.reshape(2,2))
            if done:raise RuntimeError('unexpected first event in physical diagnostic')
            positions.append(env.positions.copy())
        return np.asarray(positions)

    rows=[]
    try:
        for rid,initial in enumerate(starts):
            draws=noise(82920,66000+rid)
            base=physical(initial,draws,params)
            for sign in (-1,1):
                tr,certs=propagate(initial,draws,sign)
                prediction=np.stack([r[1] for r in tr]);forward=np.stack([r[0] for r in tr])
                parity=float(np.max(np.abs(forward-base)))
                if parity>1e-6:raise RuntimeError(f'forward parity failed: {parity}')
                checks=[]
                for h in hs:
                    phi=jax.tree_util.tree_map(lambda p,d:p+sign*h*d,params,tangent)
                    actual=physical(initial,draws,phi);fd=(actual-base)/h
                    error=float(np.linalg.norm(fd[-1]-prediction[-1]))
                    scale=max(.01,float(np.linalg.norm(prediction[-1])))
                    checks.append(dict(h=h,endpoint_error=error,relative_error=error/scale))
                    np.savez_compressed(out/f'trace_{rid}_{sign}_{h}.npz',base=base,perturbed=actual,
                        directional_positions=prediction,finite_difference=fd)
                rows.append(dict(rid=rid,sign=sign,forward_parity=parity,checks=checks,certificates=certs))
                atomic_save(out/'records.json',rows)
        passed=all(r['checks'][-1]['endpoint_error']<.02 and r['checks'][-1]['relative_error']<.1 for r in rows)
        atomic_save(out/'complete.json',dict(passed=passed,cases=len(rows),physical_short_prefixes=3+6*len(hs),
            directional_reference_prefixes=6,steps_per_prefix=steps,training_launched=False,risk_validated=False))
        print(json.dumps(dict(passed=passed,checks=[r['checks'] for r in rows])),flush=True)
    except Exception:
        atomic_save(out/'failure.json',dict(traceback=traceback.format_exc(),completed_cases=len(rows)))
        raise


if __name__=='__main__':main()
