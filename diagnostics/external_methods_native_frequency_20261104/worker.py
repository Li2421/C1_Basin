import os,sys,time,traceback,fcntl,argparse
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ.update(MPLBACKEND='Agg',QT_QPA_PLATFORM='offscreen',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1')
os.sched_setaffinity(0,{int(os.environ.get('BASELINE_CPU','23'))})
from native_io import *
from clock_plant import *
from adapters import adapter

def run(c):
    started=time.monotonic();pl=c['payload'];env=ClockPlant(c['scenario'],c['state'],pl['control_period_s'],pl.get('external_root'))
    grid,controls,scores,horizon=timeline(c['scenario'],pl['control_period_s'])
    if pl.get('native_static_geometry'):env._bfc_native_geometry_data=pl['native_static_geometry']
    p=[env.positions.copy()];v=[env.velocities.copy()];times=[0.];actions=[];raw=[];nom=[];clear=[];call_times=[];calls=[];dur=[];score_flags=[]
    controller=context=None;error=None;clipped=0;u=None;setup=None
    try:
        controller,context=adapter(env,pl);setup=time.monotonic()-started
        for index,t in enumerate(grid[:-1]):
            tick=time.monotonic()
            if t in controls:
                k=controls[t];env.set_control()
                if context:context.begin(k)
                if context:
                    try:
                        u=np.asarray(controller.action(env,k),float)
                        if u.shape!=env.action_shape:raise ValueError('Malformed paper action')
                        for i in np.flatnonzero(~np.isfinite(u).all(axis=1)):context.fail(int(i),ValueError('Nonfinite paper action'));u[i]=0.
                    except Exception as exc:context.fail(None,exc);u=np.zeros(env.action_shape)
                    for i in context.failed:u[i]=0.
                    reference=np.full(env.action_shape,np.nan)
                else:u,reference=controller.action(env,k);u=np.asarray(u,float)
                assert u.shape==env.action_shape and np.isfinite(u).all()
                raw.append(u.copy());norms=np.linalg.norm(u,axis=1);clipped+=int(np.any(norms>env.config.max_speed+1e-7))
                u*=np.minimum(1.,env.config.max_speed/np.maximum(norms,1e-30))[:,None]
                call_times.append(float(t));calls.append(u.copy());nom.append(np.asarray(reference).copy())
            end=grid[index+1];info=env.advance(u,end,end in scores)
            actions.append(u.copy());p.append(env.positions.copy());v.append(env.velocities.copy());times.append(float(end));clear.append(info['swept_clearance']);score_flags.append(end in scores);dur.append(time.monotonic()-tick)
            if index%50==0:dump(D/'progress'/(c['rollout_uid']+'.json'),dict(time=float(end),horizon=float(horizon),calls=len(calls),elapsed=time.monotonic()-started))
            if env.done:break
    except Exception:error=traceback.format_exc()
    outcome='unknown' if error else env.termination
    result=dict(outcome=outcome,success=int(outcome=='success'),collision=int(env.collided),timeout=int(outcome=='timeout'),numerical_failure=int(error is not None),error=error,
        episode_length=len(actions),control_calls=len(calls),simulated_seconds=float(env.time),completion_time=float(env.time) if outcome=='success' else None,
        horizon_time=float(horizon),minimum_swept_clearance=min(clear) if clear else None,final_goal_distances=np.linalg.norm(env.positions-env.goals,axis=1),collision_types=env.collision_types,
        elapsed_seconds=time.monotonic()-started,setup_seconds=setup,stats=controller.stats() if controller else {},speed_clipped_steps=clipped,
        control_period_s=pl['control_period_s'],score_period_s=c['scenario']['config']['dt'],physical_key=pl['physical_key'],cohort=pl['cohort'],
        paper_error_events=context.events if context else [],paper_error_types=context.errors if context else {},
        paper_original_outcome='unknown' if error or (context and context.first) else outcome,
        temporal_profile='native_callback_ZOH_original_score_clock_v1',safety='swept SI segments, collision precedence; no C1 safety projection',scientific_outcome_valid=error is None)
    n=env.num_agents
    arrays=dict(positions=np.asarray(p),velocities=np.asarray(v),times=np.asarray(times),actions=np.asarray(actions).reshape(-1,n,2),
        control_times=np.asarray(call_times),control_actions=np.asarray(calls).reshape(-1,n,2),raw_actions=np.asarray(raw).reshape(-1,n,2),nominal=np.asarray(nom).reshape(-1,n,2),
        swept_clearance=np.asarray(clear),goals=env.goals,score_flags=np.asarray(score_flags),step_seconds=np.asarray(dur))
    record(c,result,arrays);print(json.dumps(clean(result)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('case');parser.add_argument('--output-root',required=True);args=parser.parse_args()
    assert Path('/mnt/ssd').is_mount();out=Path(args.output_root).resolve();assert out==Path('/mnt/ssd/zhihan/Basin_C1/diagnostics')/D.name
    for name in ['raw','traces','logs','artifacts']:assert (D/name).resolve()==out/name
    c=read(args.case)
    profile=c['payload']['execution_profile'];assert sha(profile)==c['payload']['execution_profile_sha256']
    for path,h in read(profile).items():assert sha(path)==h,('immutable source changed',path)
    work=D/'artifacts'/c['rollout_uid'];work.mkdir(parents=True,exist_ok=True);os.chdir(work)
    os.environ['MPLCONFIGDIR']=str(work/'mplconfig');os.environ['XDG_CACHE_HOME']=str(work/'cache')
    with (D/'locks'/(c['rollout_uid']+'.lock')).open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        if not (D/'raw'/(c['rollout_uid']+'.json')).exists():run(c)
