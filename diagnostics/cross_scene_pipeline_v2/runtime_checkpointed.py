"""Original SI order and RNG with new geometry, local range and specified mover."""
import bootstrap
from experiment import *
import time
import numpy as np
import jax
from scene_environment import NewEnv as SwapEnv
from legacy_states import LiteratureConfig
from swap_safety_user_fix_20261023 import SwapSafetyRevision as SwapSafety
from controllers import bound
from scene_environment import NewNominal as Nominal
from correction_revision_20261021 import distributed_fields,contact_radius
from central_reference import progress
from shared_control.hard_projection import HardProjectionConfig
jax.config.update('jax_enable_x64',False)
def run(c,checkpoint=None):
    start=time.monotonic();env=SwapEnv(c['state'],LiteratureConfig(**c['scenario_payload']['config']));cfg=env.config;actor=Nominal(c['controller_payload']['nominal']);hp=HardProjectionConfig()
    rng_uid=c.get('rng_state_uid',c['state_uid']);root=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(2026102004),int(hashlib.sha256(rng_uid.encode()).hexdigest()[:8],16)),c['seed'])
    positions=[env.positions.copy()];velocities=[env.velocities.copy()];nominal=[];safe=[];actions=[];rewrite=[];clear=[];geometry=[];certificates=[];durations=[];clear_types=[]
    switch=None;error=None;termination='timeout';d0=np.linalg.norm(env.goals-env.positions,axis=-1)
    elapsed_before=0.;next_step=0
    restored=checkpoint.load()if checkpoint is not None else None
    if restored is not None:
        env=restored['env']
        actor=restored['actor']
        positions=restored['positions']
        velocities=restored['velocities']
        nominal=restored['nominal']
        safe=restored['safe']
        actions=restored['actions']
        rewrite=restored['rewrite']
        clear=restored['clear']
        geometry=restored['geometry']
        certificates=restored['certificates']
        durations=restored['durations']
        clear_types=restored['clear_types']
        switch=restored['switch']
        error=restored['error']
        termination=restored['termination']
        d0=restored['d0']
        elapsed_before=restored['elapsed_before'];next_step=restored['next_step'];cfg=env.config
    for t in range(next_step,cfg.max_steps):
        before=time.monotonic()
        try:
            assert not jax.config.x64_enabled
            ref=bound(actor.action(env,jax.random.fold_in(root,t)),cfg.max_speed).astype(np.float64);ref[np.linalg.norm(env.goals-env.positions,axis=-1)<=cfg.goal_tolerance]=0
            if c['theta'] is not None and switch is None and (progress(env.positions,env.goals,d0,cfg.goal_tolerance)>=c['theta'] or t>=round(6*cfg.max_steps/7)):switch=t
            eta=np.asarray(c['eta_b'] if switch is not None else c['eta_a']);snap=env.snapshot();filters=[SwapSafety(snap,i) for i in range(env.num_agents)]
            first=[f.project(ref[i:i+1]) for i,f in enumerate(filters)];v=np.concatenate([r[0] for r in first]);target=v.copy()
            if env.num_agents==1:assert not np.any(eta)
            else:
                for i in range(env.num_agents):
                    fields=distributed_fields(env.positions,env.goals,v,i,cfg.max_speed,cfg.goal_tolerance,contact_radius(cfg));target[i]+=sum(float(x)*field for x,field in zip(eta,fields))
            second=[f.project(target[i:i+1]) for i,f in enumerate(filters)];a=np.concatenate([r[0] for r in second]);_,done,info=env.step(a)
            geometry.append([dict(A=f.A.tolist(),lower=f.lower.tolist(),effective_lower=d['effective_lower'],labels=f.kinds) for f,(_,d) in zip(filters,first)])
            certificates.append(dict(first=[d for _,d in first],second=[d for _,d in second]))
            positions.append(env.positions.copy());velocities.append(env.velocities.copy());nominal.append(ref.copy());safe.append(v.copy());actions.append(a.copy());rewrite.append(np.linalg.norm(v-ref,axis=-1)>hp.intervention_tol)
            clear.append(info['swept_clearance']);clear_types.append([info[k] for k in ('min_swept_agent_clearance','min_swept_wall_clearance','min_swept_moving_clearance')]);durations.append(time.monotonic()-before);termination=info['termination']
            if done:break
        except Exception as exc:
            import traceback
            error=dict(step=t,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());termination='unknown';break
        if checkpoint is not None:
            checkpoint.step(t+1,lambda:dict(env=env,actor=actor,positions=positions,velocities=velocities,nominal=nominal,safe=safe,actions=actions,rewrite=rewrite,clear=clear,geometry=geometry,certificates=certificates,durations=durations,clear_types=clear_types,switch=switch,error=error,termination=termination,d0=d0,elapsed_before=elapsed_before+time.monotonic()-start))
    violation=sum(any(d['maximum_violation']>1e-7 for ds in r.values() for d in ds) for r in certificates)
    excess=sum(any(d['speed_excess']>1e-7 for ds in r.values() for d in ds) for r in certificates)
    result=dict(success=int(termination=='success' and not env.collided),collision=int(env.collided),timeout=int(termination=='timeout'),numerical_failure=int(error is not None),outcome=termination,episode_length=len(actions),error=error,elapsed_seconds=elapsed_before+time.monotonic()-start,switch_step=switch,
        minimum_swept_clearance=min(clear) if clear else None,final_goal_distances=np.linalg.norm(env.goals-env.positions,axis=-1),jax_enable_x64=bool(jax.config.x64_enabled),safety_layer='distributed',responsibility=.5,
        collision_types={k:int(v) for k,v in env.collision_types.items()},constraint_violation_steps=violation,sampled_speed_violation_steps=excess,
        restoration_steps=sum(any(d['restoration_phase']!='original' for ds in r.values() for d in ds) for r in certificates),diagnostics='Saved first effective constraint rows; no controller rerun',reservation_no_goal_certificates=getattr(getattr(actor,'planner',None),'certificates',[]))
    arrays=dict(positions=np.asarray(positions),velocities=np.asarray(velocities),nominal=np.asarray(nominal).reshape(-1,env.num_agents,2),safe=np.asarray(safe).reshape(-1,env.num_agents,2),actions=np.asarray(actions).reshape(-1,env.num_agents,2),projection_rewrite=np.asarray(rewrite).reshape(-1,env.num_agents),goals=env.goals,swept_clearance=np.asarray(clear),clearance_types=np.asarray(clear_types).reshape(-1,3),first_geometry_json=np.asarray(json.dumps(geometry)),safety_certificates_json=np.asarray(json.dumps(certificates)),step_seconds=np.asarray(durations))
    if env.mover is not None:arrays.update(moving_positions=env.mover[:len(actions)+1],moving_pieces_json=np.asarray(json.dumps(env.mover_pieces[:len(actions)])))
    if checkpoint is not None:result['operational_resume']=checkpoint.provenance()
    return result,arrays
