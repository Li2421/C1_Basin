"""Strict loader of immutable rollout identities plus outcome-free entity features."""
from pathlib import Path
import dataclasses,json,hashlib,collections
import numpy as np
HERE=Path(__file__).resolve().parent
C=['flow_isotropic','flow_anisotropic'];E=[0,2,4,5,8,10,12,15]

def build(source,broad=False):
 from single_integrator.environment import Config,GiveWayEnv
 from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
 p=json.loads((source/'protocol.json').read_text());cases=json.loads((source/'cases.json').read_text());W=sorted(map(float,p['widths']))
 if not broad:cases+=json.loads((HERE/'legacy_supplement/cases.json').read_text())
 if broad:
  from .broad_states import make_state
  states=[make_state(i) for i in range(16,44)]
 else:
  a=json.loads((HERE.parent/'controller_ff_factorial_repair_v1/protocol.json').read_text());by={s['state_uid']:s for s in a['states']};states=[by[u] for u in p['train_states']+p['new_states']]
 su={s['state_uid']:i for i,s in enumerate(states)};Y=np.full((len(W),len(states),2,8,4),np.nan);T=np.full_like(Y,np.nan);M=np.zeros_like(Y,bool);U=np.zeros_like(Y,bool);index=[];counts=collections.Counter();seconds=0
 for c in cases:
  path=source/'raw'/(c['rollout_uid']+'.json')
  if not path.exists() and not broad:path=HERE/'legacy_supplement/raw'/(c['rollout_uid']+'.json')
  assert path.exists(),('incomplete',path)
  raw=path.read_bytes();r=json.loads(raw);assert r['case']==c
  assert r['protocol_sha256']==hashlib.sha256((path.parents[1]/'protocol.json').read_bytes()).hexdigest()
  ix=(W.index(c['width']),su[c['parent_state_uid']],C.index(c['controller_name']),E.index(c['eta_index']),c['seed']);assert not M[ix];M[ix]=True
  U[ix]=bool(r['numerical_failure']);Y[ix]=np.nan if U[ix] else r['success'];T[ix]=r['summary'].get('first_success_step') or np.inf
  counts[r['outcome']]+=1;seconds+=r['elapsed_seconds'];index.append({'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'rollout_uid':c['rollout_uid']})
 X=[];rawcols=[];mech=[];flat=[]
 for w in W:
  cfg=Config(**p['geometries'][f'{w:.2f}']['meta']['config']);env=GiveWayEnv(cfg)
  obs=[dict(a=x[0].tolist(),b=x[1].tolist(),radius=cfg.wall_radius,curvature=0.,interior_sign=0.) for x in env.walls]
  for s in states:
   ph=s['physical'];pos=np.asarray(ph['positions']);goals=np.asarray(ph['goals']);env.reset(pos);walls,pairs=env.distances();dist=np.linalg.norm(goals-pos,axis=1);n=2
   scene=dict(positions=pos,goals=goals,velocities=ph['velocities'],flow=np.zeros((n,2)),radius=[cfg.agent_radius]*n,goal_tolerance=[cfg.goal_tolerance]*n,obstacles=obs,policy_origin=[0.,0.],policy_axis=[1.,0.],policy_axis_required=True,flow_committed=False,remaining_fraction=1.,remaining_seconds=cfg.max_steps*cfg.dt,dt=cfg.dt,max_speed=cfg.max_speed,max_seconds=cfg.max_steps*cfg.dt,wall_margin=cfg.wall_collision_margin,agent_margin=cfg.agent_collision_margin,monitor_active=cfg.terminate_on_deadlock,progress_window_seconds=cfg.progress_window_seconds,deadlock_hold_seconds=cfg.deadlock_hold_seconds,progress_epsilon=cfg.progress_epsilon,speed_epsilon_fraction=cfg.speed_epsilon_fraction,monitor_elapsed=0.,monitor_history_empty=True)
   X.append(rep.entities(scene));r=[w,cfg.bay_top,cfg.max_steps*cfg.dt,2.];rawcols.append(r)
   # Explicitly proxies/bounds; no future outcome enters any coordinate.
   ratio=(w-2*(cfg.wall_radius+cfg.wall_collision_margin))/(2*cfg.agent_radius)
   mech.append([ratio,cfg.max_steps*cfg.dt-dist.max()/cfg.max_speed,abs(abs(pos[0,0])-abs(pos[1,0]))/cfg.max_speed,float(walls.min()-cfg.wall_collision_margin),float(pairs),cfg.bay_top-w/2-2*(cfg.agent_radius+cfg.wall_radius+cfg.wall_collision_margin),1.,0.])
   flat.append(r+np.concatenate([np.ravel(ph[k]) for k in ['positions','goals','velocities']]).tolist())
 return dict(Y=Y,T=T,planned=M,unknown=U,X=rep.batch(X),raw=np.asarray(rawcols),mech=np.asarray(mech),flat=np.asarray(flat),eta=np.array([p['eta'][str(e)] for e in E]),W=W,states=states,index=index,counts=dict(counts),cpu_seconds=seconds)

def save_data(D,name):
 np.savez_compressed(HERE/'data'/(name+'.npz'),**{k:v for k,v in D.items() if isinstance(v,np.ndarray)},**{'entity_'+k:v for k,v in D['X'].items()})
 (HERE/'data'/(name+'_index.json')).write_text(json.dumps(D['index'],indent=2)+'\n')
 (HERE/'data'/(name+'_metadata.json')).write_text(json.dumps({k:D[k] for k in ['W','states','counts','cpu_seconds']},indent=2)+'\n')
