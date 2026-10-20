from common import *

def physical12(scene,state):
 cf=read(HERE/'HISTORICAL_INVENTORY.json')['environments'][scene]['config']['config'];p=np.asarray(state['physical']['positions'],float);v=np.asarray(state['physical']['velocities'],float);g=np.asarray(state['physical']['goals'],float);r=cf['agent_radius'];speed=cf['max_speed'];L=max(2*np.linalg.norm(g-g.mean(0),axis=-1).max(),2*r);d=np.linalg.norm(g-p,axis=-1)/L;ii,jj=np.triu_indices(len(p),1);pair=np.linalg.norm(p[ii]-p[jj],axis=-1)/(2*r)
 if scene=='ring_exchange':
  rr=np.linalg.norm(p,axis=-1);wall=np.minimum(rr-cf['obstacle_radius']-r,cf['outer_radius']-r-rr)/r
 else:
  if scene=='toy_give_way':
   from single_integrator.environment import Config,GiveWayEnv
   env=GiveWayEnv(Config(**cf))
  elif scene=='double_bottleneck':
   from double_bottleneck.environment import Config,DoubleBottleneckEnv
   env=DoubleBottleneckEnv(Config(**cf))
  else:
   from four_way_intersection.environment import Config,FourWayIntersectionEnv
   env=FourWayIntersectionEnv(Config(**cf))
  walls=np.asarray(env.walls);a=walls[:,0];b=walls[:,1];ab=b-a;delta=p[:,None]-a[None];t=np.clip(np.sum(delta*ab[None],-1)/np.maximum(np.sum(ab*ab,-1),1e-30),0,1);clear=np.linalg.norm(delta-t[...,None]*ab[None],axis=-1)-r-cf.get('wall_radius',0);wall=clear.min(1)/r
 vel=np.linalg.norm(v,axis=-1)/speed;return np.asarray([len(p),d.mean(),d.std(),d.min(),d.max(),pair.min(),pair.mean(),wall.min(),wall.mean(),vel.mean(),vel.max(),L/r],np.float32)
def action7(arm):
 inv=read(HERE/'HISTORICAL_INVENTORY.json');eta=inv['eta'];a=int(arm[1:3]) if arm.startswith('S') else int(arm[1:]);b=10 if arm.startswith('S') else a;f=(3/7 if arm.endswith('t1') else 5/7) if arm.startswith('S') else 1.;return np.asarray(eta[str(a)]['values']+eta[str(b)]['values']+[f],np.float32)
def fit_transform(X):
 with np.errstate(invalid='ignore'):mean=np.nanmean(X.astype(np.float64),axis=0)
 mean=np.where(np.isfinite(mean),mean,0);filled=np.where(np.isfinite(X),X,mean);scale=filled.std(0);scale=np.where(scale<1e-6,1,scale);return dict(mean=mean.astype(np.float32),scale=scale.astype(np.float32))
def apply_transform(X,tr):return ((np.where(np.isfinite(X),X,tr['mean'])-tr['mean'])/tr['scale']).astype(np.float32)
