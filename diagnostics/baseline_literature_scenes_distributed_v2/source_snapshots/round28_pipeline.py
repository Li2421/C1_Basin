"""Per-robot TT filters. No joint safety optimization or projected-action exchange."""
from core import *
import jax
from poc.field_controller import FixedSafetySet
from ring_exchange.safety import central_obstacle_polygon,outer_boundary_polygon
from shared_control.hard_projection import HardProjectionConfig
from shared_control.basis_families import OrthoFlow3BasisFamily,ORTHOFLOW3_SCALE
from shared_control.diagnostic_corrector import bounded_rows
jax.config.update('jax_enable_x64',False)
CFG=HardProjectionConfig()
def neighbors(positions,i,radius):
 return [j for j in range(len(positions)) if j!=i and np.linalg.norm(positions[i]-positions[j])<=radius]
def wall_rows(position,cfg):
 walls=np.concatenate((central_obstacle_polygon(cfg.obstacle_radius,48),outer_boundary_polygon(cfg.outer_radius,48)))
 starts,ends=walls[:,0],walls[:,1];segment=ends-starts;length2=np.sum(segment*segment,axis=-1);delta=position[None]-starts;t=np.clip(np.sum(delta*segment,axis=-1)/length2,0,1);nearest=starts+t[:,None]*segment;d=position-nearest;distance=np.linalg.norm(d,axis=-1)
 if np.any(distance<=1e-14):raise ValueError('undefined local wall gradient')
 gradient=d/distance[:,None];h=distance-cfg.agent_radius-cfg.collision_margin-CFG.separation_buffer
 return gradient,-CFG.gamma_wall*h
class LocalSafety(FixedSafetySet):
 def __init__(self,positions,i,cfg,radius):
  self.config=CFG;self.max_speed=cfg.max_speed;self.calls=0;self.neighbor_indices=neighbors(positions,i,radius);rows=[];lower=[];safe=2*cfg.agent_radius+cfg.agent_collision_margin+CFG.separation_buffer
  for j in self.neighbor_indices:
   relative=positions[i]-positions[j];rows.append(2*relative);lower.append(-.5*CFG.gamma*(relative@relative-safe**2))
  A,b=wall_rows(positions[i],cfg);rows.extend(A);lower.extend(b);self.A=np.asarray(rows,np.float64);self.lower=np.asarray(lower,np.float64)
  if max(CFG.gamma,CFG.gamma_wall)*cfg.dt>1:raise ValueError('held-action safety condition violated')
def local_basis(positions,goals,safe,i,radius,speed):
 goal=bounded_rows((goals[i]-positions[i])[None],speed)[0];perp=ORTHOFLOW3_SCALE*OrthoFlow3BasisFamily.raw_ortho_flow(goal[None],safe[i:i+1],speed)[0];jj=neighbors(positions,i,radius);rel=bounded_rows(positions[i]-positions[jj],speed).mean(0) if jj else np.zeros(2)
 return np.stack((goal,perp,rel))
def decentralized_action(rt,env,key,eta,kind,slow_counts):
 radius=1.5 if kind=='local' else np.inf
 # These are four actual network evaluations on shared observations/public key.
 nominal=np.stack([rt.nominal(env,key)[1][i] for i in range(len(env.positions))]);filters=[LocalSafety(env.positions,i,env.config,radius) for i in range(len(nominal))];safe=np.stack([f.project(nominal[i:i+1])[0][0] for i,f in enumerate(filters)]);target=safe.copy();rhr_active=np.zeros(len(nominal),bool)
 for i in range(len(nominal)):
  if kind!='rhr':target[i]+=np.asarray(eta)@local_basis(env.positions,env.goals,safe,i,radius,env.config.max_speed)
  if kind in ['rhr','rhr_f10'] and slow_counts[i]>=40 and np.linalg.norm(env.goals[i]-env.positions[i])>env.config.goal_tolerance:
   q=env.positions[i];length=np.linalg.norm(q);target[i]+=.15*np.array([q[1],-q[0]])/max(length,1e-30);rhr_active[i]=True
 executed=np.stack([f.project(target[i:i+1])[0][0] for i,f in enumerate(filters)]);residual=[f.certificate(executed[i:i+1])['min_linear_residual'] for i,f in enumerate(filters)]
 return executed,dict(projection_calls=sum(f.calls for f in filters),minimum_local_linear_residual=min(residual),neighbor_counts=[len(f.neighbor_indices) for f in filters],rhr_active=rhr_active)
def make_runtime(name,p,cache):
 if name not in cache:
  from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
  from diagnostics.controller_q_diversity_v1.runtime import native_runtime
  jax.config.update('jax_enable_x64',False);v=p['base_profiles'][name]['payload'];cache[name]=native_runtime('ring_exchange') if name=='base_u_v10_local_macflow' else RingFlowAdapter(v['flow_checkpoint_path'],read(v['adapter_manifest_path']),v['observation_frame']);assert sha(cache[name].checkpoint)==v['flow_checkpoint_sha256']
 jax.config.update('jax_enable_x64',False);assert not jax.config.x64_enabled
 return cache[name]
