"""Common physical plant, shared route information, per-agent error stopping."""
import sys
from pathlib import Path
# Immutable engine bundles sit two directories below the experiment root.
if Path(__file__).parent.name != 'scene_types_oracle_coverage_v3':
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from core import *
import os,sys,time,traceback
from routes import guidance,route_geometry
import importlib.util

def run(c):
    work=HERE/'paper_runtime'/c['rollout_uid']/str(os.getpid());work.mkdir(parents=True,exist_ok=True);os.chdir(work)
    (work/'savefig').mkdir(exist_ok=True)
    qt=work/'qt_runtime';qt.mkdir(exist_ok=True);os.chmod(qt,0o700)
    os.environ['XDG_RUNTIME_DIR']=str(qt);os.environ['MPLBACKEND']='Agg';os.environ['QT_QPA_PLATFORM']='offscreen'
    start=time.monotonic();env=RN.SwapEnv(c['state'],RN.LiteratureConfig(**c['scenario_payload']['config']));cfg=env.config
    true_goals=env.goals.copy();geom=route_geometry(env.instance.geometry);payload=c['controller_payload'];method=payload['paper_method']
    root_only=payload.get('paper_root_diagnostic_only',False)
    module_spec=importlib.util.spec_from_file_location('isolated_paper_method',payload['paper_method_adapter'])
    adapter=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(adapter)
    context=adapter.ErrorContext(env.num_agents,unified=not root_only)
    routes_on=bool(payload.get('paper_route_guidance',False))
    env._paper_guide_variant=bool(method=='chen_impc_ob_si_tracking'and routes_on)
    positions=[env.positions.copy()];velocities=[env.velocities.copy()];actions=[];clear=[];durations=[];guides_saved=[];stops=[]
    termination='timeout';setup_error=None;clipped=0;ctl=None;setup_seconds=None;error=None
    try:
        initial_guides,_,_=guidance(geom,env.positions,true_goals,routes_on);env.goals=initial_guides
        ctl=adapter.make(env,c['seed'],method,payload['paper_options'],context,mechanism=payload.get('paper_mechanism',True),root_only=root_only)
        env.goals=true_goals;setup_seconds=time.monotonic()-start
    except Exception as exc:
        if isinstance(exc,PermissionError):raise
        env.goals=true_goals;termination='unknown';setup_error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());error=setup_error
    if ctl is not None:
        for t in range(cfg.max_steps):
            before=time.monotonic();context.begin(t)
            try:
                guides,_,_=guidance(geom,env.positions,true_goals,routes_on);env.goals=guides
                try:
                    u=np.asarray(ctl.action(env,t),float)
                    if u.shape!=env.action_shape:raise ValueError('Malformed paper action '+str(u.shape))
                    invalid=np.flatnonzero(~np.isfinite(u).all(axis=1))
                    for i in invalid:context.fail(int(i),ValueError('Nonfinite paper action'));u[i]=0.
                except Exception as exc:
                    if root_only:raise
                    context.fail(None,exc);u=np.zeros(env.action_shape)
                env.goals=true_goals
                for i in context.failed:u[i]=0.
                norms=np.linalg.norm(u,axis=1);clipped+=int((norms>cfg.max_speed+1e-7).any())
                u*=np.minimum(1.,cfg.max_speed/np.maximum(norms,1e-30))[:,None]
                _,done,info=env.step(u);termination=info['termination']
                positions.append(env.positions.copy());velocities.append(env.velocities.copy());actions.append(u.copy());clear.append(info['swept_clearance']);guides_saved.append(guides.copy())
                stopped=np.zeros(env.num_agents,bool);stopped[list(context.failed)]=True;stops.append(stopped);durations.append(time.monotonic()-before)
                if done:break
            except Exception as exc:
                if isinstance(exc,PermissionError):raise
                env.goals=true_goals;termination='unknown';error=dict(step=t,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());break
    env.goals=true_goals;stats=ctl.stats()if ctl is not None else{}
    handled=[x for x in stats.get('solver',{}).get('failures',[])if x.get('paper_remark4_recovery')]
    for x in handled:
        if x['step']<len(stops):stops[x['step']][x['agent']]=True
    success=int(termination=='success'and not env.collided)
    raw_unknown=bool(setup_error or context.first or error)
    first_step=context.first['step']if context.first else(error or{}).get('step',0)
    raw=dict(success=0 if raw_unknown else success,collision=0 if raw_unknown else int(env.collided),timeout=0 if raw_unknown else int(termination=='timeout'),numerical_unknown=int(raw_unknown),outcome='unknown'if raw_unknown else termination,episode_length=first_step if raw_unknown else len(actions),completion_time=None if raw_unknown else(len(actions)*cfg.dt if success else None),first_error=context.first or setup_error or error)
    result=dict(success=success,collision=int(env.collided),timeout=int(termination=='timeout'),numerical_failure=int(termination=='unknown'),outcome=termination,error=error,setup_error=setup_error,episode_length=len(actions),elapsed_seconds=time.monotonic()-start,setup_seconds=setup_seconds,controller_stats=stats,solver_failure_count=len(stats.get('solver',{}).get('failures',[])),speed_clipped_steps=clipped,completion_time=len(actions)*cfg.dt if success else None,truncated_time=len(actions)*cfg.dt if success else cfg.max_steps*cfg.dt,horizon_time=cfg.max_steps*cfg.dt,final_goal_distances=np.linalg.norm(env.positions-true_goals,axis=1),minimum_swept_clearance=min(clear)if clear else None,collision_types=env.collision_types,paper_original=raw,paper_error_events=context.events,paper_error_types=context.errors,paper_stop_robot_steps=sum(len(x['agents'])for x in context.events),paper_stop_joint_steps=len({x['step']for x in context.events}),route_guidance=routes_on,original_constraint_metric_available=False)
    arrays=dict(positions=np.asarray(positions),velocities=np.asarray(velocities),actions=np.asarray(actions).reshape(-1,env.num_agents,2),goals=true_goals,swept_clearance=np.asarray(clear),step_seconds=np.asarray(durations),route_guides=np.asarray(guides_saved),error_stop_mask=np.asarray(stops).reshape(-1,env.num_agents),error_events_json=np.asarray(json.dumps(context.events)),error_types_json=np.asarray(json.dumps(context.errors)))
    result.update(paper_method=method,paper_options=payload['paper_options'],paper_mechanism=payload.get('paper_mechanism',True),paper_root_diagnostic_only=root_only,paper_handled_recovery_events=handled,paper_stop_robot_steps=int(np.sum(stops)),paper_stop_joint_steps=int(np.any(np.asarray(stops),axis=1).sum())if stops else 0)
    result['safety_layer']='paper_own_controller_safety'
    result.update(paper_profile=c.get('paper_profile'),paper_profile_labels=c.get('paper_profile_labels',[]))
    return clean(result),arrays

if __name__=='__main__':
    import runner,runtime
    c=read(sys.argv[1]);os.environ.pop('STB_COORDINATOR',None)
    from budget import reserve
    reserve(c)
    runtime.run=run;r=runner.work(c);print('paper',method if 'method'in globals()else r['controller'],r['type_id'],r['state_index'],r['outcome'],round(r['wall_seconds'],3),flush=True)
