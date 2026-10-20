"""Original-position safety action followed by a triggered clockwise tangent."""
from common import *
from trigger import Trigger
from poc.field_controller import Runtime
from factor_runtime import make as factor_make
from runtime import execute as factor_execute
from controllers import make as novel_make
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
import jax,time
jax.config.update('jax_enable_x64',False)

class RightHand:
    def __init__(self,base,rule):self.base=base;self.rule=rule;self.trigger=Trigger(rule);self.last_safety=None;self.active_log=[]
    def __getattr__(self,k):return getattr(self.base,k)
    def safety(self,env):self.last_safety=self.base.safety(env);return self.last_safety
    def prepare(self,*args,**kw):return Runtime.prepare(self,*args,**kw)
    def action(self,env,key,eta,chain='old'):
        assert np.array_equal(np.asarray(eta),np.zeros(3));active=self.trigger.update(env);baseline,certificate,calls=Runtime.action(self,env,key,np.zeros(3),chain)
        if not active.any():self.active_log.append(active);return baseline,certificate,calls
        target=baseline.copy()
        for i in range(len(env.positions)):
            if active[i]:q=env.positions[i];target[i]+=self.rule['speed']*np.array([q[1],-q[0]])/max(np.linalg.norm(q),1e-30)
        action,_=self.last_safety.project(target);certificate=self.last_safety.assert_safe(action);self.active_log.append(active);return action,certificate,self.last_safety.calls

def novel_episode(c,p,proxy,phase):
    s=next(s for s in p['states'] if s['state_uid']==c['state_uid']);env=proxy.make_env(s);pos=[env.positions.copy()];vel=[env.velocities.copy()];actions=[];swept=[];term='timeout';error=None;before=time.monotonic()
    for step in range(proxy.config.max_steps):
        try:
            assert not jax.config.x64_enabled;key=key_for(s,c['seed'],step);action,_,_=proxy.action(env,key,np.zeros(3),'old');_,_,done,info=env.step(action);term=info['termination'];pos.append(env.positions.copy());vel.append(env.velocities.copy());actions.append(np.asarray(action).copy());cfg=proxy.config;swept.append(min(float(np.min(info['swept_obstacle_distance']))-cfg.collision_margin,float(np.min(info['swept_outer_distance']))-cfg.collision_margin,float(np.min(info['swept_agent_distances']))-cfg.agent_collision_margin))
            if done:break
        except Exception as ex:error=dict(step=step,type=type(ex).__name__,message=str(ex)[:600]);term='numerical_failure';break
    summary=env.summary();collision=any(summary.get(k,False) for k in ['wall_collision','obstacle_collision','outer_collision','agent_collision']);path=HERE/phase/'traces'/f"{c['rollout_uid']}.npz";path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,positions=np.asarray(pos),velocities=np.asarray(vel),actions=np.asarray(actions),goals=env.goals,swept_clearance=np.asarray(swept),rhr_active=np.asarray(proxy.active_log[:len(actions)],bool).reshape(-1,4));v=p['profiles'][c['base_controller_name']]['payload'];physical=s['physical']
    return clean(dict(cache_identity_schema='registered_exact_uid_v1',experiment_uid=EXP,scenario='ring_exchange',scenario_uid=s['scenario_uid'],case=c,**{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key','rollout_uid']},eta=[0.,0.,0.],future_index=c['seed'],future_root=2026100403,rng_namespace=s['rng_namespace'],original_uid=s['uid'],initial_positions=physical['positions'],initial_velocities=physical['velocities'],goals=physical['goals'],flow_checkpoint_sha256=v.get('flow_checkpoint_sha256'),success=int(summary['collision_free_success'] and term=='success' and error is None),collision=int(collision),timeout=int(term=='timeout'),deadlock=0,numerical_failure=int(error is not None),outcome=term,episode_length=len(actions),physical_steps=len(pos)-1,new_physical_steps=len(actions),first_action=actions[0] if actions else None,summary=summary,error=error,trace_path=str(path),trace_sha256=sha(path),execution_code_sha256=sha(__file__),jax_enable_x64=bool(jax.config.x64_enabled),elapsed_seconds=time.monotonic()-before))

def execute(c,p,cache,phase):
    if c['kind']=='Flow':
        base=factor_make(c['base_controller_name'],c['H'],c['r'],cache);proxy=RightHand(base,c['rhr_rule']);injected=dict(cache);injected[(c['base_controller_name'],c['H'],c['r'])]=proxy;r=factor_execute(c,p,injected,phase)
        path=Path(r['trace_path'])
        with np.load(path) as z:values={k:z[k] for k in z.files}
        values['rhr_active']=np.asarray(proxy.active_log[:r['physical_steps']],bool).reshape(-1,4);np.savez_compressed(path,**values);r['trace_sha256']=sha(path);r['execution_code_sha256']=sha(__file__);r['case']=c;return r
    base=novel_make(c['base_controller_name'],p,cache);return novel_episode(c,p,RightHand(base,c['rhr_rule']),phase)
