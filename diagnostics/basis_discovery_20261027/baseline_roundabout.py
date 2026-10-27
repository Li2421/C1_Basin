"""Lee et al. 2025 roundabout mechanism in the frozen SI plant.
A paper-mechanism adaptation, not an execution of the authors' unicycle stack.
Reference parameters are pinned in literature/mgr/params.py (downloaded, not imported).
"""
import numpy as np


def wall_distance(p,w):
    p=np.asarray(p).reshape(-1,2);w=np.asarray(w).reshape(-1,2,2)
    if not len(w):return np.full(len(p),np.inf)
    a=w[:,0];ab=w[:,1]-a;d=p[:,None]-a
    t=np.clip(np.sum(d*ab,axis=-1)/np.sum(ab*ab,axis=-1),0,1)
    return np.linalg.norm(d-t[...,None]*ab,axis=-1).min(axis=-1)


class Roundabout:
    def __init__(self,env):
        self.radius=max(.3,2*1.1*env.config.agent_radius)
        self.robot_radius=env.config.agent_radius;self.walls=np.asarray(env.walls).reshape(-1,2,2)
        self.circles=[];self.next_id=0;self.invalid=0;self.created=0;self.exits=0
        # Grid resolution: author's 300 pixels / 4m default; geometry only.
        pts=np.concatenate([env.positions,env.goals,self.walls.reshape(-1,2)])
        self.lo=pts.min(0)-self.radius;self.hi=pts.max(0)+self.radius
        self.grid=None;self.clear=None
    def center(self,c,n):
        required=self.radius+3*self.robot_radius+.1*max(n-1,0)
        if wall_distance(c[None],self.walls)[0]>=required:return c
        if self.grid is None:
            xs=np.arange(self.lo[0],self.hi[0]+1/75,1/75);ys=np.arange(self.lo[1],self.hi[1]+1/75,1/75)
            self.grid=np.stack(np.meshgrid(xs,ys),axis=-1).reshape(-1,2)
            self.clear=np.concatenate([wall_distance(x,self.walls) for x in np.array_split(self.grid,max(1,len(self.grid)//1000))])
        d=np.linalg.norm(self.grid-c,axis=-1);ok=(self.clear>=required)&(d<=10*self.radius)
        if not ok.any():self.invalid+=1;return None
        idx=np.flatnonzero(ok);return self.grid[idx[np.argmin(d[idx])]].copy()
    def update(self,env,arrived):
        q=env.positions;g=env.goals;v=env.velocities;n=len(q);speed=env.config.max_speed
        for circle in self.circles:circle['members']={i for i in circle['members'] if not arrived[i]}
        self.circles=[c for c in self.circles if len(c['members'])>=2]
        membership={i:c for c in self.circles for i in c['members']}
        # Local one-second prediction, author code's proximity threshold.
        used=set()
        for i in range(n):
            if arrived[i] or i in used:continue
            for j in range(i+1,n):
                if arrived[j] or j in used:continue
                rel=q[j]-q[i];dist=np.linalg.norm(rel)
                if dist>1.:continue
                if i in membership and j in membership and membership[i]['id']==membership[j]['id']:continue
                dv=v[j]-v[i];den=dv@dv;tt=np.clip(-(rel@dv)/den,0,1.) if den>=1e-8 else 0.
                if not (dist<2.5*self.robot_radius or np.linalg.norm(rel+tt*dv)<1.1*self.robot_radius):continue
                midpoint=(q[i]+q[j])/2
                candidates=[c for c in self.circles if np.linalg.norm(c['center']-midpoint)<6.5*self.radius and any(np.linalg.norm(q[k]-q[i])<=1. or np.linalg.norm(q[k]-q[j])<=1. for k in c['members'])]
                circle=candidates[0] if candidates else None
                members=({i,j}|circle['members']) if circle else {i,j}
                ctr=self.center(circle['center'] if circle else midpoint,len(members))
                if ctr is None:continue
                if circle is None:
                    circle=dict(id=self.next_id,center=ctr,members=set());self.next_id+=1;self.created+=1;self.circles.append(circle)
                circle['center']=ctr
                for k in [i,j]:
                    if k in membership:membership[k]['members'].discard(k)
                    circle['members'].add(k);membership[k]=circle;used.add(k)
        target={};sizes={}
        for circle in self.circles:
            ids=sorted(circle['members']);cnt=len(ids)
            if cnt<2:continue
            ctr=circle['center'];sizes[cnt]=sizes.get(cnt,0)+1
            for i in ids:
                d=q[i]-ctr;rad=np.linalg.norm(d);gd=g[i]-q[i];gn=np.linalg.norm(gd)
                if rad<1e-9:continue
                er=d/rad
                # Paper orthogonality + clear outer sector; tolerances from author code.
                orth=gn>0 and abs(er@gd/gn)<=np.sin(np.deg2rad(10))
                rel=q-ctr;rr=np.linalg.norm(rel,axis=-1)
                dots=np.sum(rel*er,axis=-1)/np.maximum(rr,1e-30)
                blocked=any(j!=i and rr[j]>=rad-self.robot_radius/2 and rr[j]<=rad+1. and dots[j]>=np.cos(np.deg2rad(15)) for j in range(n))
                # Sector clearance is conservatively evaluated along 31 rays, with true wall distances.
                ang=np.arctan2(er[1],er[0])+np.linspace(-np.pi/12,np.pi/12,31)
                rays=np.stack([np.cos(ang),np.sin(ang)],axis=-1)
                points=ctr+np.linspace(rad,rad+1.,40)[:,None,None]*rays[None]
                if orth and not blocked:blocked=np.any(wall_distance(points.reshape(-1,2),self.walls)<self.robot_radius)
                if orth and not blocked:
                    circle['members'].discard(i);self.exits+=1;continue
                # Paper Eq.11: CCW tangent and radial stabilization, k_p from author defaults.
                tangent=np.array([-er[1],er[0]])*speed
                radial=-(.05/cnt)*(rad-self.radius)*er*speed
                z=tangent+radial;target[i]=speed*z/max(np.linalg.norm(z),1e-30)
        return target,sizes
