from pathlib import Path
import json,hashlib,collections
import numpy as np
from .factor_runtime import config,Env
HERE=Path(__file__).resolve().parent

def load():
 from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
 src=HERE/'factors';p=json.loads((src/'protocol.json').read_text());cases=json.loads((src/'cases.json').read_text());geos=p['geometries'];physical={s['state_uid']:s for s in p['states']};parent_hash=hashlib.sha256((src/'protocol.json').read_bytes()).hexdigest()
 Y=np.full((24,16,2,4,2),np.nan);U=np.zeros_like(Y,bool);T=np.full((12,16,2,4,2),np.nan);counts=collections.Counter();index=[];seconds=0
 for c in cases:
  f=src/'raw'/(c['rollout_uid']+'.json');assert f.exists(),('missing',f);raw=f.read_bytes();r=json.loads(raw);assert r['case']==c and r['protocol_sha256']==parent_hash
  g,s,cc,e,sd=c['geometry_index'],c['state_index'],c['controller_index'],p['eta_indices'].index(c['eta_index']),c['seed'];tt=r['summary']['first_success_step'] or np.inf;T[g,s,cc,e,sd]=tt
  for h,H in enumerate([400,850]):
   ix=(2*g+h,s,cc,e,sd);U[ix]=bool(r['numerical_failure']);Y[ix]=np.nan if U[ix] else int(r['success'] and tt<=H)
  counts[r['outcome']]+=1;seconds+=r['elapsed_seconds'];index.append({'path':str(f),'sha256':hashlib.sha256(raw).hexdigest(),'trace':r['trace_path']})
 X=[];rawcols=[];mech=[];contexts=[]
 for g in geos:
  cfg=config(g['width'],g['N'],g['topology']);n=g['N'];opening=1 if g['topology']=='single' else 2
  for h,H in enumerate([400,850]):
   contexts.append({**g,'target_horizon':H,'physical_geometry_index':g['geometry_index']})
   for si in range(16):
    s=next(s for s in p['states'] if s['geometry_index']==g['geometry_index'] and s['state_index']==si);ph=s['physical'];env=Env(cfg,ph);pos=env.positions;goals=env.goals;dist=np.linalg.norm(goals-pos,axis=-1);paths=[env.instance.geometry.path(a,b) for a,b in zip(pos,goals)];detour=np.mean([np.linalg.norm(np.diff(path,axis=0),axis=1).sum()/d for path,d in zip(paths,dist)])
    obstacles=[dict(a=w[0],b=w[1],radius=cfg.wall_radius,curvature=0.,interior_sign=0.) for w in env.walls]
    scene=dict(positions=pos,goals=goals,velocities=ph['velocities'],flow=np.zeros((n,2)),radius=[cfg.agent_radius]*n,goal_tolerance=[cfg.goal_tolerance]*n,obstacles=obstacles,policy_origin=[0.,0.],policy_axis=[1.,0.],policy_axis_required=True,flow_committed=False,remaining_fraction=1.,remaining_seconds=H*cfg.dt,dt=cfg.dt,max_speed=cfg.max_speed,max_seconds=H*cfg.dt,wall_margin=cfg.wall_collision_margin,agent_margin=cfg.agent_collision_margin,monitor_active=False,progress_window_seconds=0.,deadlock_hold_seconds=0.,progress_epsilon=0.,speed_epsilon_fraction=0.,monitor_elapsed=0.,monitor_history_empty=True)
    X.append(rep.entities(scene));rawcols.append([g['width'],n,H*cfg.dt,opening,opening-1])
    from single_integrator.environment import point_segment_distance
    clear=point_segment_distance(pos[:,None],env.walls[None,:,0],env.walls[None,:,1]).min()-env.instance.geometry.margin
    mech.append([(g['width']-2*(cfg.wall_radius+cfg.wall_collision_margin))/(2*cfg.agent_radius),H*cfg.dt-dist.max()/cfg.max_speed,detour,opening-1,opening,n/opening,np.std(abs(pos[:,0])/cfg.max_speed),clear])
 return dict(Y=Y,unknown=U,T=T,X=rep.batch(X),raw=np.asarray(rawcols),mech=np.asarray(mech),flat=np.asarray(mech),eta=np.array([p['eta'][str(e)] for e in p['eta_indices']]),contexts=contexts,counts=dict(counts),index=index,cpu_seconds=seconds)
