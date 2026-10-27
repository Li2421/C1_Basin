"""Nominal -> bound/arrival stop -> certified projection -> basis -> projection."""
from experiment import *
import numpy as np
import jax
import time
from scipy.optimize import nnls
from pilot_policies import PilotEnv as LiteratureEnv
from states import LiteratureConfig
from controllers import bound
from pilot_policies import PilotNominal as Nominal
from projection_memo import MemoSafetyFilter as LiteratureSafetyFilter
from shared_control.basis_families import get_basis_family
jax.config.update('jax_enable_x64',False)

def progress(p,g,d0,tolerance):
    dist=np.linalg.norm(g-p,axis=-1);out=np.empty_like(dist);moving=d0>0
    out[moving]=np.clip(1-dist[moving]/d0[moving],0,1);out[~moving]=(dist[~moving]<=tolerance).astype(float)
    return float(out.mean())

def correction_for(env,safe,eta,basis):
    if len(env.positions)==1:
        if np.any(eta):raise ValueError('N=1 capability supports safety-only; relative correction basis is undefined')
        return np.zeros_like(safe)
    return basis.compute(env.positions,env.goals,safe,env.config.max_speed).correction(eta)

def run(c):
    start=time.monotonic();env=LiteratureEnv(c['state'],LiteratureConfig(**c['scenario_payload']['config']));cfg=env.config;actor=Nominal(c['controller_payload']['nominal'])
    projector=LiteratureSafetyFilter();basis=get_basis_family('orthoflow3');root=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(2026102004),int(hashlib.sha256(c['state_uid'].encode()).hexdigest()[:8],16)),c['seed'])
    p=[env.positions.copy()];vel=[env.velocities.copy()];nom=[];safe=[];aa=[];residual=[];rewrite=[];active=[];clear=[];steps_seconds=[];switch=None;error=None;term='timeout'
    d0=np.linalg.norm(env.goals-env.positions,axis=-1)
    for t in range(cfg.max_steps):
        stepstart=time.monotonic()
        try:
            assert not jax.config.x64_enabled
            ref=bound(actor.action(env,jax.random.fold_in(root,t)),cfg.max_speed).astype(np.float64)
            ref[np.linalg.norm(env.goals-env.positions,axis=-1)<=cfg.goal_tolerance]=0
            snap=env.snapshot();first=projector(snap,ref);v=np.asarray(first.velocity,np.float64)
            if c['theta'] is not None and switch is None and (progress(env.positions,env.goals,d0,cfg.goal_tolerance)>=c['theta'] or t>=round(6*cfg.max_steps/7)):switch=t
            eta=np.asarray(c['eta_b'] if switch is not None else c['eta_a'])
            if c['method']=='nominal_only':a=ref
            else:
                correction=correction_for(env,v,eta,basis)
                a=np.asarray(projector(snap,v+correction).velocity,np.float64)
            # Active linear inequalities use their inward normals; the KKT
            # cone diagnostic solves ||u_nom+A_active.T*mu||, mu>=0.
            A,l,kinds=projector.linear_problem(snap);ids=np.flatnonzero(np.asarray(first.diagnostics['linear_residuals'])<=1e-7)
            if len(ids) and np.linalg.norm(ref)>1e-12:
                _,r=nnls(A[ids].T,-ref.ravel(),maxiter=10000);cone=float(r/np.linalg.norm(ref))
            else:cone=0. if np.linalg.norm(ref)<=1e-12 else 1.
            _,done,info=env.step(a,diagnose_stalls=False)
            p.append(env.positions.copy());vel.append(env.velocities.copy());aa.append(a.copy());nom.append(ref.copy());safe.append(v.copy());residual.append(cone);rewrite.append(np.linalg.norm(v-ref,axis=-1)>projector.config.intervention_tol);active.append([kinds[i] for i in ids]);clear.append(info['swept_clearance']);steps_seconds.append(time.monotonic()-stepstart);term=info['termination']
            if done:break
        except Exception as e:
            import traceback
            error=dict(step=t,type=type(e).__name__,message=str(e),traceback=traceback.format_exc());term='unknown';break
    result=dict(success=int(term=='success' and not env.collided),collision=int(env.collided),timeout=int(term=='timeout'),numerical_failure=int(error is not None),outcome=term,episode_length=len(aa),error=error,elapsed_seconds=time.monotonic()-start,switch_step=switch,minimum_swept_clearance=min(clear) if clear else None,final_goal_distances=np.linalg.norm(env.goals-env.positions,axis=-1),jax_enable_x64=bool(jax.config.x64_enabled),
                paper_position_convergence=int(np.all(np.linalg.norm(env.goals-env.positions,axis=-1)<.1)),paper_velocity_convergence=int(np.all(np.linalg.norm(env.velocities,axis=-1)<.1)))
    arrays=dict(positions=np.asarray(p),velocities=np.asarray(vel),actions=np.asarray(aa).reshape(-1,env.num_agents,2),nominal=np.asarray(nom).reshape(-1,env.num_agents,2),safe=np.asarray(safe).reshape(-1,env.num_agents,2),cone_residual=np.asarray(residual),projection_rewrite=np.asarray(rewrite).reshape(-1,env.num_agents),active_json=np.asarray(json.dumps(active)),goals=env.goals,swept_clearance=np.asarray(clear),step_seconds=np.asarray(steps_seconds))
    return result,arrays
