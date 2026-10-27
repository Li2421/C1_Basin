"""Full frozen paper port on the literal original physical environment step."""
from core import *
import sys,os,time,traceback
sys.path.insert(0,str(HERE/'ports'))
from paper_adapters import METHODS
import runner,runtime

def run(c):
    start=time.monotonic();env=RN.SwapEnv(c['state'],RN.LiteratureConfig(**c['scenario_payload']['config']));cfg=env.config
    positions=[env.positions.copy()];velocities=[env.velocities.copy()];actions=[];clear=[];durations=[];error=None;termination='timeout';ctl=None;setup=None;clipped=0
    try:
        ctl=METHODS[c['controller_payload']['paper_method']](env,c['seed'],c['controller_payload'].get('paper_options'));setup=time.monotonic()-start
        for t in range(cfg.max_steps):
            before=time.monotonic();u=np.asarray(ctl.action(env,t),float)
            if u.shape!=env.action_shape or not np.isfinite(u).all():raise ValueError('Nonfinite or malformed paper action')
            norms=np.linalg.norm(u,axis=1);clipped+=int((norms>cfg.max_speed+1e-7).any());u*=np.minimum(1.,cfg.max_speed/np.maximum(norms,1e-30))[:,None]
            _,done,info=env.step(u);termination=info['termination'];positions.append(env.positions.copy());velocities.append(env.velocities.copy());actions.append(u);clear.append(info['swept_clearance']);durations.append(time.monotonic()-before)
            if done:break
    except Exception as exc:termination='unknown';error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    stats=ctl.stats()if ctl else{};success=int(termination=='success'and not env.collided)
    result=dict(success=success,collision=int(env.collided),timeout=int(termination=='timeout'),numerical_failure=int(error is not None),outcome=termination,error=error,episode_length=len(actions),elapsed_seconds=time.monotonic()-start,setup_seconds=setup,controller_stats=stats,solver_failure_count=len(stats.get('solver',{}).get('failures',[])),speed_clipped_steps=clipped,completion_time=len(actions)*cfg.dt if success else None,truncated_time=len(actions)*cfg.dt if success else cfg.max_steps*cfg.dt,horizon_time=cfg.max_steps*cfg.dt,final_goal_distances=np.linalg.norm(env.positions-env.goals,axis=1),minimum_swept_clearance=min(clear)if clear else None,collision_types=env.collision_types)
    arrays=dict(positions=np.asarray(positions),velocities=np.asarray(velocities),actions=np.asarray(actions).reshape(-1,env.num_agents,2),goals=env.goals,swept_clearance=np.asarray(clear),step_seconds=np.asarray(durations))
    return clean(result),arrays

if __name__=='__main__':
    c=read(sys.argv[1]);dump(HERE/'attempts'/f"{c['rollout_uid']}_{time.time_ns()}.json",dict(rollout_uid=c['rollout_uid'],job=os.environ.get('SLURM_JOB_ID'),time=time.time()))
    if len(list((HERE/'attempts').glob('*.json')))>45000:raise RuntimeError('HARD_STOP rollout budget')
    runtime.run=run;r=runner.work(c);print('paper',r['controller'],r['type_id'],r['state_index'],r['outcome'],round(r['wall_seconds'],3),flush=True)
