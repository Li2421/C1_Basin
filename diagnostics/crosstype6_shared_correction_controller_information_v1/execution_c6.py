"""New arm: Safety until stall, early/late schedule, isolated optional requery."""
from common import *
from mixed_controllers import make
from historical_controllers import Observed
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
from dataclasses import replace
from schedule_c6 import Schedule
import jax,time
jax.config.update('jax_enable_x64',False)

class TracedObserved(Observed):
    def nominal(self,env,key,traced=False):
        obs,action,path=self.base.nominal(env,key,traced)
        self.last_nominal=np.asarray(action).copy()
        return obs,action,path

def execute(c,p,cache,phase,probe=False):
    assert not os.environ.get('XLA_FLAGS')
    from new_execution import ensure
    ensure(c,p,cache);base=TracedObserved(make(c['base_controller_name'],p,cache));rt=base
    assert not c.get('rhr_rule')
    s=next(s for s in p['states'] if s['state_uid']==c['state_uid'])
    if probe:
        env=rt.env_cls(replace(rt.config,terminate_on_collision=False,terminate_on_success=False));v=s['physical']
        env.reset(np.asarray(v['positions']),velocities=np.asarray(v['velocities']),goals=np.asarray(v['goals']))
    else:env=rt.make_env(s)
    positions=[env.positions.copy()];velocities=[env.velocities.copy()];actions=[];nominal=[];active=[];swept=[]
    error=None;term='timeout';steps=0;requery=None;late_eta=None;query_physical=None;query_step=None
    rule=Schedule(positions[0],env.goals,c.get('progress_threshold',.968),c.get('trigger','none'))
    before=time.monotonic()
    for t in range(700):
        try:
            assert not jax.config.x64_enabled
            key=key_for(s,c['seed'],t)
            stage=rule.decide(t,env.positions,env.goals)
            if stage=='safety':eta=[0.,0.,0.]
            elif stage=='early':eta=c['eta_a']
            elif c.get('late_source','given')!='requery':eta=c['eta_b']
            else:
                if late_eta is None:
                    from requery_client import FrozenMean
                    if '_frozen_mean' not in cache:cache['_frozen_mean']=FrozenMean()
                    query_physical=dict(positions=env.positions.copy(),velocities=env.velocities.copy(),goals=env.goals.copy(),step=int(t))
                    requery=cache['_frozen_mean'].query(c['state_uid'],query_physical);late_eta=requery['eta_b'];query_step=int(t)
                eta=late_eta
            action,_,_=rt.action(env,key,np.asarray(eta,np.float64),'field' if c['placement']=='FF' else 'old');intervened=base.last.active
            steps+=1;_,_,done,info=env.step(action)
            positions.append(env.positions.copy());velocities.append(env.velocities.copy());actions.append(np.asarray(action).copy());nominal.append(base.last_nominal.copy());active.append(intervened)
            cfg=env.config
            swept.append(min(float(np.min(info['swept_obstacle_distance']))-cfg.collision_margin,float(np.min(info['swept_outer_distance']))-cfg.collision_margin,float(np.min(info['swept_agent_distances']))-cfg.agent_collision_margin))
            if not probe:
                term=info['termination']
                if done:break
        except Exception as exc:
            error=dict(step=t,type=type(exc).__name__,message=str(exc)[:500]);term='numerical_failure';break
    trace=HERE/phase/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(trace,positions=positions,velocities=velocities,nominal_actions=nominal,safe_actions=actions,actions=actions,projection_active=active,swept_clearance=swept,goals=env.goals,progress_history=rule.history,trigger_step=-1 if rule.triggered_at is None else rule.triggered_at,progress_switch_step=-1 if rule.switched_at is None else rule.switched_at,requery_eta_b=np.asarray([] if late_eta is None else late_eta,np.float64),requery_switch_step=-1 if query_step is None else query_step)
    result=dict(experiment_uid=EXP,case=c,**{k:c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key','rollout_uid')},trace_path=str(trace),trace_sha256=sha(trace),error=error,physical_steps=len(actions),execution_code_sha256=sha(__file__),schedule_code_sha256=sha(HERE/'schedule_c6.py'),jax_enable_x64=False,elapsed_seconds=time.monotonic()-before,trigger_step=rule.triggered_at,progress_switch_step=rule.switched_at,late_eta_requery=None if requery is None else dict(**requery,switch_step=query_step,physical=query_physical))
    if probe:
        result.update(schema='crosstype6_unlabeled_probe700_v1',complete=len(actions)==700,terminal_labels_read=False,expected_steps=700)
        assert not any(k in result for k in ('success','outcome','termination','summary'))
    else:
        summary=env.summary();collision=any(summary.get(k,False) for k in ('wall_collision','obstacle_collision','outer_collision','agent_collision'))
        result.update(cache_identity_schema='registered_exact_uid_v1',scenario='ring_exchange',scenario_uid=s['scenario_uid'],eta=c['eta_a'],future_index=c['seed'],future_root=2026100403,rng_namespace=s['rng_namespace'],original_uid=s['uid'],initial_positions=s['physical']['positions'],initial_velocities=s['physical']['velocities'],goals=s['physical']['goals'],flow_checkpoint_sha256=p['profiles'][c['base_controller_name']]['payload'].get('flow_checkpoint_sha256'),success=int(summary['collision_free_success'] and term=='success' and error is None),collision=int(collision),timeout=int(term=='timeout'),deadlock=0,numerical_failure=int(error is not None),outcome=term,episode_length=steps,summary=summary,first_action=actions[0] if actions else None)
    return clean(result)
