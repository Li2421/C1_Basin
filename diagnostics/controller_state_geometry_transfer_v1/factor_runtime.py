"""N-generic geometric navigator with existing affine FF, OrthoFlow3 and safety.
No fixed-N Flow checkpoint, no multi-agent planner, no outcome-informed routes.
"""
import sys,dataclasses,hashlib,time
from types import SimpleNamespace
import numpy as np
from bottleneck_family.scenario import Config,Geometry
from bottleneck_family.environment import BottleneckEnv
from diagnostics.controller_response_families_v1.runtime import EndpointMatchedField
from diagnostics.controller_q_diversity_v1.runtime import FIELD,bounded
sys.path.insert(0,str(FIELD))
from poc.field_controller import FixedSafetySet
from shared_control.basis_families import get_basis_family
from shared_control.hard_projection import HardProjectionConfig


def config(width,n,topology):
 openings=(((0.,float(width)),),) if topology=='single' else (((-.6,float(width)),(.6,float(width))),)
 return Config(num_agents=n,half_height=2.,barrier_x=(0.,),openings=openings,max_steps=850,seed=0,split='dev')

def state(i,n):
 rng=np.random.default_rng(np.random.SeedSequence([2026101113,i]));p=[]
 for k in range(4):
  while True:
   v=np.array([(-1 if k%2==0 else 1)*rng.uniform(1.4,2.4),rng.uniform(-.5,.5)])
   if all(np.linalg.norm(v-z)>.45 for z in p):p.append(v);break
 p=np.array(p[:n]);g=p.copy();g[:,0]*=-1
 return {'positions':p.tolist(),'goals':g.tolist(),'velocities':np.zeros_like(p).tolist(),'timestep':0}

class Env(BottleneckEnv):
 def __init__(self,cfg,physical):
  super().__init__(cfg);self.reset(np.array(physical['positions']),np.array(physical['goals']));self.minwall=self.minpair=float('inf');self.first_success=None;self.anywall=self.anypair=False
 def step(self,u):
  obs,done,info=super().step(u,diagnose_stalls=False)
  self.minwall=min(self.minwall,info['min_swept_wall_clearance']);self.minpair=min(self.minpair,info['min_swept_agent_clearance']);self.anywall|=info['min_swept_wall_clearance']<=0;self.anypair|=info['min_swept_agent_clearance']<=0
  if info['termination']=='success':self.first_success=self.step_count
  return obs,0.,done,info
 def summary(self):
  return {'collision_free_success':self.termination=='success' and not self.collided,'wall_collision':bool(self.anywall),'agent_collision':bool(self.anypair),'first_success_step':self.first_success,'min_swept_wall_clearance':self.minwall,'min_swept_agent_clearance':self.minpair,'episode_steps':self.step_count,'stalled_groups_are_not_deadlocks':True}

class Base:
 def __init__(self,cfg):
  self.config=cfg;self.scenario='bottleneck_factor';self.steps=10;self.agent=SimpleNamespace(config={'joint_act_dim':2*cfg.num_agents});self.scale=np.full((cfg.num_agents,2),.15);self.mean=np.zeros_like(self.scale);self.basis=get_basis_family('orthoflow3');self.cbf=HardProjectionConfig()
 def make_env(self,s):return Env(self.config,s['physical'])
 def observation(self,e):return e.observation()
 def to_world(self,a,e):return np.asarray(a)
 def to_local(self,a,e):return np.asarray(a)
 def safety(self,e):return FixedSafetySet(e.snapshot(),self.cbf)

class Field(EndpointMatchedField):
 def target(self,env):
  targets=[]
  for p,g in zip(env.positions,env.goals):
   path=env.instance.geometry.path(p,g)
   if path is None:raise ValueError('no individual path witness from current safe state')
   d=path[1]-p;dist=np.linalg.norm(d);targets.append(.85*self.config.max_speed*np.tanh(dist/.5)*d/max(dist,1e-12))
  p=env.positions;dp=p[:,None]-p[None];dd=np.linalg.norm(dp,axis=-1);proximity=np.exp(-(dd/.65)**2)*(1-np.eye(len(p)));radial=dp/np.maximum(dd[...,None],1e-12)
  target=np.asarray(targets)+self.config.max_speed*.25*np.sum(proximity[...,None]*radial,axis=1)-.2*env.velocities
  return bounded(target,self.config.max_speed)

def rollout(rt,s,eta,seed,trace_path=None):
 import jax
 from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
 env=rt.make_env(s);started=time.monotonic();error=None;term='timeout';steps=calls=0;first=None;worst=0.
 for t in range(rt.config.max_steps):
  try:
   a,cert,n=rt.action(env,key_for(s,seed,t),np.asarray(eta),'field');calls+=n;worst=max(worst,cert['max_violation']);steps+=1
   if first is None:first=a.tolist()
   _,_,done,info=env.step(a);term=info['termination']
   if done:break
  except Exception as ex:error={'step':t,'type':type(ex).__name__,'message':str(ex)[:500]};term='numerical_failure';break
 summ=env.summary()
 if trace_path is not None:np.savez_compressed(trace_path,positions=np.asarray(env.history),goals=env.goals,walls=env.walls)
 return dict(success=int(summ['collision_free_success'] and term=='success' and error is None),collision=int(env.collided),deadlock=0,timeout=int(term=='timeout'),numerical_failure=int(error is not None),episode_length=steps,outcome=term,elapsed_seconds=time.monotonic()-started,first_action=first,projection_calls=calls,error=error,summary=summ,max_projection_violation=worst,trace_sha256=hashlib.sha256(np.asarray(env.history).tobytes()).hexdigest())
