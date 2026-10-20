"""Vectorized independent goal fields; scalar field arithmetic preserved per row."""
import numpy as np
from single_integrator.environment import segment_distance
from navigation_field import SPACING,TRACE_STEP
from observation import observe as observe35
from types import SimpleNamespace

class IndependentFields:
 def __init__(self,fields):
  assert fields
  self.grid=fields[0].grid;self.geometry=self.grid.geometry
  assert all(np.array_equal(f.grid.geometry.walls,self.geometry.walls) for f in fields)
  self.cost=np.stack([f.cost for f in fields]);self.goals=np.array([f.goal for f in fields]);self.fields=fields
 def cell(self,p):
  xy=(p-np.array([self.grid.x[0],self.grid.y[0]]))/SPACING;ij=np.floor(xy).astype(int);ij[:,0]=np.clip(ij[:,0],0,len(self.grid.x)-2);ij[:,1]=np.clip(ij[:,1],0,len(self.grid.y)-2)
  return ij,np.clip(xy-ij,0,1)
 def gradient(self,p,rows):
  ij,t=self.cell(p);x,y=ij.T;tx,ty=t.T
  a=self.cost[rows,y,x];b=self.cost[rows,y,x+1];c=self.cost[rows,y+1,x];d=self.cost[rows,y+1,x+1]
  four=np.stack([a,b,c,d],-1);good=np.isfinite(four).all(-1);values=np.where(np.isfinite(four),four,0);a,b,c,d=values.T
  g=np.stack([((1-ty)*(b-a)+ty*(d-c))/SPACING,((1-tx)*(c-a)+tx*(d-b))/SPACING],-1)
  return g,good
 def fallback(self,p,rows):
  ij=np.rint((p-np.array([self.grid.x[0],self.grid.y[0]]))/SPACING).astype(int);offsets=np.array([(dx,dy) for dy in [-1,0,1] for dx in [-1,0,1] if dx or dy])
  q=ij[:,None]+offsets;q[:,:,0]=np.clip(q[:,:,0],0,len(self.grid.x)-1);q[:,:,1]=np.clip(q[:,:,1],0,len(self.grid.y)-1)
  targets=self.grid.points[q[:,:,1],q[:,:,0]];cost=self.cost[rows[:,None],q[:,:,1],q[:,:,0]];dist=np.linalg.norm(targets-p[:,None],axis=-1)
  starts=np.broadcast_to(p[:,None,:],targets.shape);safe=segment_distance(starts[:,:,None,:],targets[:,:,None,:],self.geometry.walls[None,None,:,0],self.geometry.walls[None,None,:,1]).min(-1)>self.geometry.margin
  local=np.min(np.where(safe,cost+dist,np.inf),axis=1);slope=np.where(safe&(dist>1e-9),(local[:,None]-cost)/np.maximum(dist,1e-9),-np.inf);k=np.argmax(slope,axis=1);target=targets[np.arange(len(p)),k];v=target-p
  return v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-30)
 def waypoint(self,positions,s,rows=None):
  p=np.atleast_2d(np.asarray(positions,float)).copy();rows=np.arange(len(p)) if rows is None else np.asarray(rows);goals=self.goals[rows];remaining=np.full(len(p),float(s))
  for _ in range(int(np.ceil(s/TRACE_STEP))+1):
   distance=np.linalg.norm(goals-p,axis=1);active=remaining>1e-9;near=active&(distance<=np.minimum(remaining,2*SPACING))
   if near.any():
    ids=np.flatnonzero(near);ok=np.array([self.geometry.visible(p[i],goals[i]) for i in ids]);ids=ids[ok];p[ids]=goals[ids];remaining[ids]=0
   active=remaining>1e-9
   if not active.any():break
   ids=np.flatnonzero(active);a=p[ids];grad,valid=self.gradient(a,rows[ids]);norm=np.linalg.norm(grad,axis=1,keepdims=True);direction=-grad/np.maximum(norm,1e-30);step=np.minimum(TRACE_STEP,remaining[ids]);b=a+step[:,None]*direction
   safe=self.geometry.valid_points(b)&(segment_distance(a[:,None],b[:,None],self.geometry.walls[None,:,0],self.geometry.walls[None,:,1]).min(-1)>self.geometry.margin);fallback=~valid|~safe|(norm[:,0]<1e-9)
   if fallback.any():direction[fallback]=self.fallback(a[fallback],rows[ids][fallback]);b=a+step[:,None]*direction
   p[ids]=b;remaining[ids]-=step
  return p
 def observed(self,envs,rows):
  positions=np.concatenate([e.positions for e in envs]);velocities=np.concatenate([e.velocities for e in envs]);goals=np.concatenate([e.goals for e in envs])
  # Each env is truly single-agent. Proxy only batches independent ray arithmetic.
  base=observe35(SimpleNamespace(positions=positions,velocities=velocities,goals=goals,walls=self.geometry.walls));base[:,20:]=0
  near=self.waypoint(positions,.5,rows)-positions;far=self.waypoint(positions,1.5,rows)-positions
  return np.concatenate([base,near,far],axis=1).astype(np.float32)
 def expert(self,envs,rows):
  p=np.concatenate([e.positions for e in envs]);goals=np.concatenate([e.goals for e in envs]);d=np.linalg.norm(goals-p,axis=1);target=self.waypoint(p,.3,rows);delta=target-p
  u=delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-30)*envs[0].config.max_speed*np.minimum(1,d/.3)[:,None]
  u[d<=envs[0].config.goal_tolerance]=0
  return u
