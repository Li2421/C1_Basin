"""Copied feature/descriptor code (critic_pool_supplement_v1 scenes.py features, descriptors_v2.descriptor, correction_v1 literature_opening)."""
import math,numpy as np
import sys;sys.path.insert(0,'/home/zhihan/research/Basin_C1')
from single_integrator.environment import segment_distance
def literature_opening(name,walls,radius):
    if not walls:return 10.
    if name=='lit_circle_five':
        # The closed128-gon has inward support radius11m, hence width22m.
        return 22./(2*radius)
    w=np.asarray(walls,float);components=list(range(len(w)))
    def root(i):
        while components[i]!=i:i=components[i]
        return i
    for i in range(len(w)):
        for j in range(i):
            if np.min(np.linalg.norm(w[i,:,None,:]-w[j,None,:,:],axis=-1))<1e-12:
                components[root(i)]=root(j)
    gaps=[float(segment_distance(w[i,0],w[i,1],w[j,0],w[j,1]))for i in range(len(w))for j in range(i)if root(i)!=root(j)]
    assert gaps,('No defined opening descriptor',name)
    return min(gaps)/(2*radius)


def features(scene):
    c = scene['config']; n = c['num_agents']; r = c['agent_radius']; speed = c['max_speed']; walls = scene['walls']
    state_features = []; longest = []
    for s in scene['states']:
        p = np.asarray(s['positions']); g = np.asarray(s['goals']); d = g-p
        longest.append(float(np.linalg.norm(d, axis=-1).max()/speed))
        i, j = np.triu_indices(n, 1); rel = p[i]-p[j]; delta = d[i]-d[j]
        t = np.clip(-np.sum(rel*delta, axis=-1)/np.maximum(np.sum(delta*delta, axis=-1), 1e-30), 0., 1.)
        near = np.linalg.norm(rel+t[:, None]*delta, axis=-1)
        conflict = near < 3*r
        # Segment intersection includes collinear overlap, computed independently of shared arrival time.
        def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
        ab = cross(d[i], d[j]); offset = p[j]-p[i]
        ta = cross(offset, d[j])/np.where(abs(ab)>1e-14, ab, 1.)
        tb = cross(offset, d[i])/np.where(abs(ab)>1e-14, ab, 1.)
        intersect = (abs(ab)>1e-14)&(ta>=0)&(ta<=1)&(tb>=0)&(tb<=1)
        collinear = (abs(ab)<=1e-14)&(abs(cross(offset,d[i]))<=1e-14)
        along1 = np.sum(offset*d[i],axis=-1)/np.maximum(np.sum(d[i]*d[i],axis=-1),1e-30)
        along2 = np.sum((g[j]-p[i])*d[i],axis=-1)/np.maximum(np.sum(d[i]*d[i],axis=-1),1e-30)
        intersect |= collinear & (np.minimum(along1,along2)<=1)&(np.maximum(along1,along2)>=0)
        cosine = np.sum(d[i]*d[j],axis=-1)/np.maximum(np.linalg.norm(d[i],axis=-1)*np.linalg.norm(d[j],axis=-1),1e-30)
        head_on = np.mean(cosine[conflict]<-.5) if conflict.any() else 0.
        mid = (p+g)/2; mid_distance = np.linalg.norm(mid[i]-mid[j],axis=-1)
        # Midpoint crowding is per robot mean count within 3r; distances above are divided by r.
        state_features.append([near.min()/r, near.mean()/r, 2*conflict.sum()/n,
            2*intersect.sum()/n, head_on, 2*np.sum(mid_distance<3*r)/n])
    free_area = scene['free_area']
    density = 0. if free_area is None else n*math.pi*r*r/free_area
    return [float(n), r, speed*c['dt']/r, c['max_steps']*c['dt']/max(longest), density,
        float(len(walls)), scene['min_opening_over_diameter'], float(scene['serial_walls']),
        float(scene.get('moving',False)), .25/speed if scene.get('moving') else 0.,
        .3/r if scene.get('moving') else 0., float(scene['diametric'])] + np.mean(state_features,axis=0).tolist()



def wall_components(walls):
    w=np.asarray(walls,float).reshape(-1,2,2);parent=list(range(len(w)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in range(len(w)):
        for j in range(i):
            if np.min(np.linalg.norm(w[i,:,None,:]-w[j,None,:,:],axis=-1))<1e-9:parent[root(i)]=root(j)
    return len({root(i)for i in range(len(w))})

def descriptor(scene):
    f=np.asarray(scene['features'],float)
    travel=np.mean([np.linalg.norm(np.asarray(s['goals'])-s['positions'],axis=-1).mean()for s in scene['states']])
    assert travel>0 and np.all(f[[0,1,2,3,6]]>0)
    return np.r_[np.log(f[0]),np.log(f[1]/travel),np.log(f[2]),np.log(f[3]),f[4],wall_components(scene['walls']),np.log(f[6]),f[7:12]]


def state_h(scene):
    """per-state 6 features (same formulas as features())"""
    import copy
    out=[]
    for s in scene['states']:
        one=dict(scene,states=[s]);out.append(features(one)[12:])
    return np.asarray(out)
