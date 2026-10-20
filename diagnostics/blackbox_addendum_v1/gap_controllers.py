"""New N-independent P2 potential: fixed gains, native static wall geometry."""
import numpy as np
P2_GAINS=(1.0,.015,.01)
def bound(a,speed):
 a=np.asarray(a,np.float64);return a*np.minimum(1.,speed/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-30))
def potential(env):
 attraction,robot_gain,wall_gain=P2_GAINS;p=np.asarray(env.positions,np.float64);walls=np.asarray(env.walls,np.float64);delta=p[:,None]-p[None,:];distance=np.maximum(np.linalg.norm(delta,axis=-1),.03);rep=delta/distance[...,None]*(np.maximum(1/distance-1/.9,0)/distance**2)[...,None];rep[np.arange(len(p)),np.arange(len(p))]=0
 start=walls[:,0];v=walls[:,1]-start;relative=p[:,None]-start[None];t=np.clip(np.einsum('nwi,wi->nw',relative,v)/np.maximum(np.sum(v*v,axis=-1),1e-30)[None],0,1);away=p[:,None]-(start[None]+t[...,None]*v[None]);norm=np.maximum(np.linalg.norm(away,axis=-1),.03);clearance=np.maximum(norm-env.instance.geometry.margin,.03);obstacle=away/norm[...,None]*(np.maximum(1/clearance-1/.6,0)/clearance**2)[...,None]
 return bound(attraction*(env.goals-p)+robot_gain*rep.sum(1)+wall_gain*obstacle.sum(1),env.config.max_speed)
def nominal_reference(action,env):
 action=np.asarray(action,np.float64).copy();action[np.linalg.norm(env.goals-env.positions,axis=1)<=env.config.goal_tolerance]=0;return action
def corrected_action(env,reference,eta,projector,basis):
 snapshot=env.snapshot();first=projector(snapshot,reference);safe=np.asarray(first.velocity,np.float64);correction=basis.compute(env.positions,env.goals,safe,env.config.max_speed).correction(np.asarray(eta,np.float64));second=projector(snapshot,safe+correction);return np.asarray(second.velocity,np.float64),safe,correction
