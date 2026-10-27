"""v2 single-integrator execution; cone diagnostics are recomputed from traces."""
from experiment import *
import time
import numpy as np
import jax
from states import LiteratureConfig
from flow_scene import LiteratureEnv
from controllers import bound
from reservation_certificate_v2_pa_memo_query_disk_candidate_spatial_mask_20261022 import ReservationNominal as Nominal
from distributed_safety import LocalSafety,local_basis,responsibility
from projection_memo import MemoSafetyFilter
from projection_adapter import constraints
from shared_control.basis_families import get_basis_family
from central_reference import progress
from correction_revision_20261021 import distributed_fields,central_fields,contact_radius
jax.config.update('jax_enable_x64',False)

def run(c):
    started=time.monotonic()
    env=LiteratureEnv(c['state'],LiteratureConfig(**c['scenario_payload']['config']))
    cfg=env.config;actor=Nominal(c['controller_payload']['nominal'])
    layer=c.get('safety_layer',c['controller_payload'].get('safety_layer','distributed'))
    coefficient=c.get('responsibility',responsibility(cfg.scene))
    projector=MemoSafetyFilter();basis=get_basis_family('orthoflow3')
    rng_uid=c.get('rng_state_uid',c['state_uid'])
    root=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(2026102004),
        int(hashlib.sha256(rng_uid.encode()).hexdigest()[:8],16)),c['seed'])
    positions=[env.positions.copy()];velocities=[env.velocities.copy()]
    nominal=[];safe=[];actions=[];rewrite=[];active=[];local_active=[];clear=[];durations=[]
    switch=np.full(env.num_agents,-1,dtype=int);error=None;termination='timeout';d0=np.linalg.norm(env.goals-env.positions,axis=-1)
    for t in range(cfg.max_steps):
        before=time.monotonic()
        try:
            assert not jax.config.x64_enabled
            ref=bound(actor.action(env,jax.random.fold_in(root,t)),cfg.max_speed).astype(np.float64)
            ref[np.linalg.norm(env.goals-env.positions,axis=-1)<=cfg.goal_tolerance]=0
            snap=env.snapshot()
            assert layer == "distributed"
            dist=np.linalg.norm(env.goals-env.positions,axis=-1)
            ratio=np.empty_like(dist);moving=d0>0
            ratio[moving]=np.clip(1-dist[moving]/d0[moving],0,1)
            ratio[~moving]=(dist[~moving]<=cfg.goal_tolerance).astype(float)
            switch[(switch<0)&((ratio>=c["theta"])|(t>=round(6*cfg.max_steps/7)))]=t
            eta=np.where((switch>=0)[:,None],np.asarray(c["eta_b"]),np.asarray(c["eta_a"]))
            if c['method']=='nominal_only':
                v=ref.copy();a=ref.copy();global_labels=[];local_labels=[[] for _ in ref]
            elif layer=='central':
                first=projector(snap,ref);v=np.asarray(first.velocity,np.float64)
                correction=np.zeros_like(v) if len(v)==1 else central_fields(env.positions,env.goals,v,cfg.max_speed,cfg.goal_tolerance,contact_radius(cfg),basis).correction(eta)
                a=np.asarray(projector(snap,v+correction).velocity,np.float64)
                A,lower,kinds=projector.linear_problem(snap)
                ids=np.flatnonzero(A@v.ravel()-lower<=1e-7)
                global_labels=[kinds[k] for k in ids]
                local_labels=[[kind for kind in global_labels if
                    (kind[0]=='pair' and i in kind[1:]) or (kind[0]=='wall' and kind[1]==i)] for i in range(len(v))]
            else:
                problem=constraints(snap,projector.config)
                filters=[LocalSafety(snap,i,projector.config,coefficient,problem) for i in range(len(ref))]
                v=np.concatenate([f.project(ref[i:i+1])[0] for i,f in enumerate(filters)])
                target=v.copy()
                if len(v)==1:
                    assert not np.any(eta),'Capability episodes are safety-only'
                else:
                    for i in range(len(v)):
                        fields=distributed_fields(env.positions,env.goals,v,i,cfg.max_speed,cfg.goal_tolerance,contact_radius(cfg))
                        target[i]+=sum(float(value)*field for value,field in zip(eta[i],fields))
                a=np.concatenate([f.project(target[i:i+1])[0] for i,f in enumerate(filters)])
                local_labels=[[f.kinds[k] for k in np.flatnonzero(f.A@v[i]-f.lower<=1e-7)] for i,f in enumerate(filters)]
                A,lower,kinds=problem
                global_labels=[kinds[k] for k in np.flatnonzero(A@v.ravel()-lower<=1e-7)]
            _,done,info=env.step(a,diagnose_stalls=False)
            positions.append(env.positions.copy());velocities.append(env.velocities.copy())
            nominal.append(ref.copy());safe.append(v.copy());actions.append(a.copy())
            rewrite.append(np.linalg.norm(v-ref,axis=-1)>projector.config.intervention_tol)
            active.append(global_labels);local_active.append(local_labels)
            clear.append(info['swept_clearance']);durations.append(time.monotonic()-before)
            termination=info['termination']
            if done:break
        except Exception as exc:
            import traceback
            error=dict(step=t,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            termination='unknown';break
    result=dict(success=int(termination=='success' and not env.collided),collision=int(env.collided),
        timeout=int(termination=='timeout'),numerical_failure=int(error is not None),outcome=termination,
        episode_length=len(actions),error=error,elapsed_seconds=time.monotonic()-started,switch_step=switch,
        minimum_swept_clearance=min(clear) if clear else None,
        final_goal_distances=np.linalg.norm(env.goals-env.positions,axis=-1),jax_enable_x64=bool(jax.config.x64_enabled),
        paper_position_convergence=int(np.all(np.linalg.norm(env.goals-env.positions,axis=-1)<.1)),
        paper_velocity_convergence=int(np.all(np.linalg.norm(env.velocities,axis=-1)<.1)),
        safety_layer=layer,responsibility=coefficient,diagnostics='Cones recomputed from saved positions/nominal/first-safe without controller calls')
    result["reservation_no_goal_certificates"]=getattr(getattr(actor,"planner",None),"certificates",[])
    arrays=dict(positions=np.asarray(positions),velocities=np.asarray(velocities),
        actions=np.asarray(actions).reshape(-1,env.num_agents,2),nominal=np.asarray(nominal).reshape(-1,env.num_agents,2),
        safe=np.asarray(safe).reshape(-1,env.num_agents,2),projection_rewrite=np.asarray(rewrite).reshape(-1,env.num_agents),
        active_json=np.asarray(json.dumps(active)),local_active_json=np.asarray(json.dumps(local_active)),
        goals=env.goals,swept_clearance=np.asarray(clear),step_seconds=np.asarray(durations))
    return result,arrays
