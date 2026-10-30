"""Original two projections and plant, with declared route goal-input hooks."""
from core import *
from routes import guidance,route_geometry
from shared_control.basis_dictionary import NEW,ALL,add_target,RoleState
import inspect,time,traceback
from admission import AdmissionState,apply as apply_admission
import shared_control.basis_dictionary as BD
import correction_revision_20261021 as CR

def generated_route_evaluate():
    source=inspect.getsource(BD.evaluate)
    source=source.replace("names=NEW, t5_seed='tangent', clearance_side=1):","names=NEW, t5_seed='tangent', clearance_side=1, route_remaining=None, allow_reselection=False):")
    source=source.replace('start=time.perf_counter();','route_remaining=np.linalg.norm(np.asarray(g)-np.asarray(p),axis=1) if route_remaining is None else np.asarray(route_remaining)\n    start=time.perf_counter();')
    source=source.replace('np.linalg.norm(g[i]-p[i])','route_remaining[i]').replace('np.linalg.norm(g[j]-p[j])','route_remaining[j]')
    source=source.replace('remaining=np.linalg.norm(g-p,axis=1)','remaining=route_remaining')
    source=source.replace('    if role.pair is None and near_pairs:', '''    if allow_reselection and role.pair is not None and (np.linalg.norm(p[role.pair[0]]-p[role.pair[1]])>.5 or all(route_remaining[i]<=cfg.goal_tolerance for i in role.pair)):
        role.pair=None;role.yielder=None;role.passer=None;role.activation_step=None
    if role.pair is None and near_pairs:''')
    # The remaining-distance input changes only goal-distance tests and the
    # original shorter-distance role ordering. Every direction formula is exact.
    target=HERE/'generated/admission_route_dictionary_evaluate.py'
    if not target.exists():target.write_text(source)
    else:assert target.read_text()==source
    ns=dict(vars(BD));exec(compile(source,str(target),'exec'),ns);return ns['evaluate']

ROUTE_EVALUATE=generated_route_evaluate()

def fields_for(env,safe,guides,remaining):
    # Use the original local basis and original contact mask formula; the
    # distance input to the arrival test is the actual remaining route length.
    p=env.positions;cfg=env.config
    if env._route_enabled:
        arrived=remaining<=cfg.goal_tolerance
        close=np.linalg.norm(p[:,None]-p[None,:],axis=-1)<=RN.contact_radius(cfg)
        mask=np.any(close & ~np.eye(len(p),dtype=bool) & ~(arrived[:,None]&arrived[None,:]),axis=1)
        ff=np.stack([CR.local_basis(p,guides,safe,i,np.inf,cfg.max_speed)for i in range(len(p))])
        ff[~mask,2]=0.;return ff
    return np.stack([RN.distributed_fields(p,guides,safe,i,cfg.max_speed,cfg.goal_tolerance,RN.contact_radius(cfg))for i in range(len(p))])

def run(c):
    start=time.monotonic();env=RN.SwapEnv(c['state'],RN.LiteratureConfig(**c['scenario_payload']['config']));cfg=env.config
    actor=RN.Nominal(c['controller_payload']['nominal']);true_goals=env.goals.copy()
    env._route_enabled=bool(c['scenario_payload']['route_guidance']);geom=route_geometry(env.instance.geometry)
    rng_uid=c.get('rng_state_uid',c['state_uid'])
    root=RN.jax.random.fold_in(RN.jax.random.fold_in(RN.jax.random.PRNGKey(2026102004),int(hashlib.sha256(rng_uid.encode()).hexdigest()[:8],16)),c['seed'])
    positions=[env.positions.copy()];velocities=[env.velocities.copy()];nominal=[];safe=[];actions=[];rewrite=[];clear=[];geometry=[];certificates=[];durations=[];clear_types=[];guides_history=[];remaining_history=[]
    error=None;termination='timeout';role=RoleState();spec=c['spec'];eta=np.asarray(spec['eta']);co=spec.get('coefficients',{});history={}
    svd_seconds=0.;field_seconds=0.;unavailable={};role_history=[]
    passage_list=read(HERE/'PASSAGES.json')['by_scene'][c['scene']]['passages'] if spec.get('one_way')or spec.get('coordination') else []
    admission_state=AdmissionState(len(passage_list));admission_history=[]
    coordination=None
    if spec.get('coordination'):
        from followup_runtime import Coordinator
        coordination=Coordinator(spec['coordination'],len(passage_list))
    for t in range(cfg.max_steps):
        before=time.monotonic()
        try:
            guides,remaining,full_paths=guidance(geom,env.positions,true_goals,env._route_enabled)
            env.goals=guides
            # Priority space-time A* already supplies its own route and must see
            # the true terminal destination and its full reservation horizon.
            if actor.kind in ('prioritized_astar','conflict_based_search'):env.goals=true_goals
            ref=RN.bound(actor.action(env,RN.jax.random.fold_in(root,t)),cfg.max_speed).astype(np.float64)
            ref[np.linalg.norm(true_goals-env.positions,axis=-1)<=cfg.goal_tolerance]=0.
            env.goals=true_goals
            snap=env.snapshot();filters=[RN.SwapSafety(snap,i)for i in range(env.num_agents)]
            first=[f.project(ref[i:i+1])for i,f in enumerate(filters)];v=np.concatenate([r[0]for r in first]);target=v.copy()
            if env.num_agents==1:assert not np.any(eta)and not co
            else:
                ff=fields_for(env,v,guides,remaining)
                for i in range(env.num_agents):target[i]+=sum(float(x)*field for x,field in zip(eta,ff[i]))
                if co:
                    columns,meta=ROUTE_EVALUATE(env.positions,guides,v,ref,ff,filters,env.walls,cfg,role,t,names=co,route_remaining=remaining,allow_reselection=spec.get('role_reselection',False))
                    target=add_target(v,ff,columns,eta,co)
                    field_seconds+=meta['seconds'];svd_seconds+=meta['svd_seconds']
                    for k,val in meta['unavailable'].items():unavailable[k]=unavailable.get(k,0)+int(val)
                    role_history.append(meta['role'])
            if spec.get('reference'):
                history[t]=remaining.copy();history.pop(t-41,None)
                active=(history[t-40]-remaining<.05)&(remaining>cfg.goal_tolerance)if t>=40 else np.zeros(len(remaining),bool)
                if spec['reference']=='right_goal':
                    d=guides-env.positions;d=np.c_[d[:,1],-d[:,0]];d*=.35/np.maximum(np.linalg.norm(d,axis=1),1e-30)[:,None]
                    target[active]+=d[active]
                else:
                    from shared_control.correction_b5 import conflict_components
                    for ids in conflict_components(env.positions,RN.contact_radius(cfg)):
                        if not active[ids].any():continue
                        q=env.positions[ids]-env.positions[ids].mean(0);d=np.c_[q[:,1],-q[:,0]]
                        d*=.35/np.maximum(np.linalg.norm(d,axis=1),1e-30)[:,None];target[ids]+=d
            if spec.get('one_way'):
                target,admission_meta=apply_admission(target,env.positions,guides,remaining,full_paths,cfg,passage_list,admission_state,spec['one_way'])
                admission_history.append(admission_meta)
            elif coordination is not None:
                target,admission_meta=coordination.apply(target,env.positions,guides,remaining,full_paths,cfg,passage_list,filters,v,t,geom)
                admission_history.append(admission_meta)
            second=[f.project(target[i:i+1])for i,f in enumerate(filters)];a=np.concatenate([r[0]for r in second]);_,done,info=env.step(a)
            geometry.append([dict(A=f.A.tolist(),lower=f.lower.tolist(),effective_lower=d['effective_lower'],labels=f.kinds)for f,(_,d)in zip(filters,first)])
            certificates.append(dict(first=[d for _,d in first],second=[d for _,d in second]))
            positions.append(env.positions.copy());velocities.append(env.velocities.copy());nominal.append(ref.copy());safe.append(v.copy());actions.append(a.copy());rewrite.append(np.linalg.norm(v-ref,axis=-1)>RN.HardProjectionConfig().intervention_tol)
            clear.append(info['swept_clearance']);clear_types.append([info[k]for k in ('min_swept_agent_clearance','min_swept_wall_clearance','min_swept_moving_clearance')]);guides_history.append(guides);remaining_history.append(remaining);durations.append(time.monotonic()-before);termination=info['termination']
            if done:break
        except Exception as exc:
            env.goals=true_goals;error=dict(step=t,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());termination='unknown';break
    violation=sum(any(d['maximum_violation']>1e-7 for ds in r.values()for d in ds)for r in certificates)
    excess=sum(any(d['speed_excess']>1e-7 for ds in r.values()for d in ds)for r in certificates)
    final_route_error=None
    try:_,end_remaining,_=guidance(geom,env.positions,true_goals,env._route_enabled)
    except ValueError as exc:end_remaining=None;final_route_error=str(exc)
    success=int(termination=='success'and not env.collided)
    result=dict(success=success,collision=int(env.collided),timeout=int(termination=='timeout'),numerical_failure=int(error is not None),outcome=termination,episode_length=len(actions),error=error,elapsed_seconds=time.monotonic()-start,switch_step=None,
        minimum_swept_clearance=min(clear)if clear else None,final_goal_distances=np.linalg.norm(true_goals-env.positions,axis=-1),final_route_distances=end_remaining,final_route_error=final_route_error,
        jax_enable_x64=bool(RN.jax.config.x64_enabled),safety_layer='distributed',responsibility=.5,collision_types=env.collision_types,constraint_violation_steps=violation,sampled_speed_violation_steps=excess,
        restoration_steps=sum(any(d['restoration_phase']!='original'for ds in r.values()for d in ds)for r in certificates),
        completion_time=len(actions)*cfg.dt if success else None,truncated_time=len(actions)*cfg.dt if success else cfg.max_steps*cfg.dt,horizon_time=cfg.max_steps*cfg.dt,
        dictionary=dict(seconds=field_seconds,svd_seconds=svd_seconds,unavailable_steps=unavailable,role=role.__dict__),
        reservation_no_goal_certificates=getattr(getattr(actor,'planner',None),'certificates',[]))
    arrays=dict(positions=np.asarray(positions),velocities=np.asarray(velocities),nominal=np.asarray(nominal).reshape(-1,env.num_agents,2),safe=np.asarray(safe).reshape(-1,env.num_agents,2),actions=np.asarray(actions).reshape(-1,env.num_agents,2),projection_rewrite=np.asarray(rewrite).reshape(-1,env.num_agents),goals=true_goals,swept_clearance=np.asarray(clear),clearance_types=np.asarray(clear_types).reshape(-1,3),first_geometry_json=np.asarray(json.dumps(geometry)),safety_certificates_json=np.asarray(json.dumps(certificates)),step_seconds=np.asarray(durations),route_guides=np.asarray(guides_history),route_remaining=np.asarray(remaining_history),roles_json=np.asarray(json.dumps(role_history)),admission_json=np.asarray(json.dumps(clean(admission_history))))
    return clean(result),arrays
