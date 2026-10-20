"""Per-agent world-coordinate observation, with exact ray/segment intersections."""
import numpy as np

def observe(env):
    p=env.positions;v=env.velocities;n=len(p)
    angles=np.arange(16)*2*np.pi/16
    directions=np.column_stack([np.cos(angles),np.sin(angles)])
    a=env.walls[:,0];b=env.walls[:,1];s=b-a
    cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
    denominator=cross(directions[:,None,:],s[None,:,:])
    q=a[None,None,:,:]-p[:,None,None,:]
    safe=np.where(np.abs(denominator)>1e-12,denominator,1.)
    distance=cross(q,s[None,None,:,:])/safe[None,:,:]
    parameter=cross(q,directions[None,:,None,:])/safe[None,:,:]
    valid=(np.abs(denominator)[None,:,:]>1e-12)&(distance>=0)&(parameter>=0)&(parameter<=1)
    rays=np.minimum(3.,np.where(valid,distance,np.inf).min(-1))
    neighbors=np.zeros((n,3,5))
    for i in range(n):
        order=np.argsort(np.linalg.norm(p-p[i],axis=1),kind='stable')
        order=order[order!=i][:3]
        for k,j in enumerate(order):neighbors[i,k]=[*list(p[j]-p[i]),*list(v[j]-v[i]),1.]
    return np.concatenate([v,env.goals-p,rays,neighbors.reshape(n,15)],axis=1).astype(np.float32)

def navigation_action(env, paths):
    actions=[]
    for i,path in enumerate(paths):
        p=env.positions[i];goal=env.goals[i];d=np.linalg.norm(goal-p)
        if d<=env.config.goal_tolerance:actions.append(np.zeros(2));continue
        # Closest point on the immutable path, then 0.3 m arc-length lookahead.
        aa=path[:-1];bb=path[1:];vv=bb-aa;ss=np.maximum(np.sum(vv*vv,axis=1),1e-30)
        tt=np.clip(np.sum((p-aa)*vv,axis=1)/ss,0,1);pp=aa+tt[:,None]*vv
        j=int(np.argmin(np.linalg.norm(pp-p,axis=1)));target=pp[j];remaining=.3
        for k in range(j,len(path)-1):
            start=target if k==j else path[k]; delta=path[k+1]-start;length=np.linalg.norm(delta)
            if length>=remaining:target=start+delta*remaining/max(length,1e-30);break
            remaining-=length;target=path[k+1]
        if not env.instance.geometry.visible(p,target):target=path[j+1]
        delta=target-p;norm=np.linalg.norm(delta)
        speed=env.config.max_speed*min(1.,d/.3)
        actions.append(delta*min(speed/max(norm,1e-30),1/env.config.dt))
    return np.asarray(actions,np.float64)
