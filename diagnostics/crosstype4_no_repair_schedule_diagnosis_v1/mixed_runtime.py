"""New-arm execution and 300-step probes around the read-only Ring runtime."""
from common import *
from mixed_controllers import make
from historical_controllers import Observed
from right_hand import RightHand
from progress import ProgressSwitch
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
from dataclasses import replace
import jax
import time
jax.config.update('jax_enable_x64',False)


class TracedObserved(Observed):
    def nominal(self,env,key,traced=False):
        obs,action,path=self.base.nominal(env,key,traced)
        self.last_nominal=np.asarray(action).copy()
        return obs,action,path


def execute(c,p,cache,phase,probe=False):
    base=TracedObserved(make(c['base_controller_name'],p,cache))
    rt=RightHand(base,c['rhr_rule']) if c.get('rhr_rule') else base
    s=next(s for s in p['states'] if s['state_uid']==c['state_uid'])
    if probe:
        env=rt.env_cls(replace(rt.config,terminate_on_collision=False,terminate_on_success=False))
        physical=s['physical']
        env.reset(np.asarray(physical['positions']),velocities=np.asarray(physical['velocities']),goals=np.asarray(physical['goals']))
    else:
        env=rt.make_env(s)
    positions=[env.positions.copy()];velocities=[env.velocities.copy()]
    actions=[];nominal=[];active=[];swept=[];error=None;term='timeout';steps=0
    prefix=None;start_step=0
    if probe:
        from probe_prefix import load_prefix
        prefix=load_prefix(c,p,base,env,s)
        if prefix is not None:
            positions=prefix['positions'];velocities=prefix['velocities'];actions=prefix['actions']
            nominal=prefix['nominal'];active=prefix['active'];swept=prefix['swept']
            error=prefix['error'];start_step=prefix['start_step'];steps=start_step
    rule=ProgressSwitch(positions[0],env.goals,700,.9681620884830543) if c['arm']=='Sprog' else None
    before=time.monotonic()
    stop_step=700
    if probe:
        stop_step=300 if prefix is None or prefix['can_continue'] else start_step
    for t in range(start_step,stop_step):
        try:
            assert not jax.config.x64_enabled
            key=key_for(s,c['seed'],t)
            if c['arm']=='nominal_only':
                _,action,_=base.nominal(env,key);intervened=False
            else:
                eta=p['eta'][rule.eta_index(t,env.positions,env.goals)] if rule else (
                     c['eta_a'] if t<c['switch_step'] else c['eta_b'])
                action,_,_=rt.action(env,key,np.asarray(eta,np.float64),'field' if c['placement']=='FF' else 'old')
                intervened=base.last.active
            steps+=1
            _,_,done,info=env.step(action)
            positions.append(env.positions.copy());velocities.append(env.velocities.copy())
            actions.append(np.asarray(action).copy());nominal.append(base.last_nominal.copy());active.append(intervened)
            cfg=env.config
            swept.append(min(float(np.min(info['swept_obstacle_distance']))-cfg.collision_margin,
                             float(np.min(info['swept_outer_distance']))-cfg.collision_margin,
                             float(np.min(info['swept_agent_distances']))-cfg.agent_collision_margin))
            if not probe:
                term=info['termination']
                if done:break
        except Exception as exc:
            error=dict(step=t,type=type(exc).__name__,message=str(exc)[:500])
            if not probe:term='numerical_failure'
            break
    trace=HERE/phase/'traces'/f"{c['rollout_uid']}.npz"
    trace.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(trace,positions=positions,velocities=velocities,nominal_actions=nominal,
                        safe_actions=actions,actions=actions,projection_active=active,swept_clearance=swept,goals=env.goals)
    result=dict(experiment_uid=EXP,case=c,**{k:c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key','rollout_uid')},
                trace_path=str(trace),trace_sha256=sha(trace),error=error,physical_steps=len(actions),
                execution_code_sha256=sha(__file__),jax_enable_x64=False,elapsed_seconds=time.monotonic()-before)
    if probe:
        result.update(schema='crosstype1_mechanism_probe_v1',complete=len(actions)==300,
                      terminal_labels_read=False,expected_steps=300,
                      reused_prefix_steps=start_step,new_physical_steps=len(actions)-start_step,
                      reused_prefix_source=prefix['source'] if prefix is not None else None)
        assert not any(k in result for k in ('success','outcome','termination'))
    else:
        summary=env.summary();collision=any(summary.get(k,False) for k in ('wall_collision','obstacle_collision','outer_collision','agent_collision'))
        result.update(cache_identity_schema='registered_exact_uid_v1',scenario='ring_exchange',scenario_uid=s['scenario_uid'],
             eta=c['eta_a'],future_index=c['seed'],future_root=2026100403,rng_namespace=s['rng_namespace'],
             original_uid=s['uid'],initial_positions=s['physical']['positions'],initial_velocities=s['physical']['velocities'],goals=s['physical']['goals'],
             flow_checkpoint_sha256=p['profiles'][c['base_controller_name']]['payload'].get('flow_checkpoint_sha256'),
             success=int(summary['collision_free_success'] and term=='success' and error is None),
             collision=int(collision),timeout=int(term=='timeout'),deadlock=0,numerical_failure=int(error is not None),
             outcome=term,episode_length=steps,summary=summary,first_action=actions[0] if actions else None,
             progress_switch_step=rule.switched_at if rule else None)
    return clean(result)
