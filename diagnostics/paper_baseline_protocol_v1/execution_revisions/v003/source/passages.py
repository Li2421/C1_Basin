"""Pure wall geometry, matching the supplied reference before merging."""
import numpy as np

def detect(walls,cfg):
    r=cfg['agent_radius'];off=r+cfg['wall_radius']+cfg['wall_collision_margin']
    cap=2*r+cfg['agent_collision_margin'];W=np.asarray(walls,float);raw=[]
    for i in range(len(W)):
        for j in range(i+1,len(W)):
            a1,b1=W[i];a2,b2=W[j];u=b1-a1;v=b2-a2
            l1=np.linalg.norm(u);l2=np.linalg.norm(v)
            if min(l1,l2)<1e-9:continue
            u=u/l1;v=v/l2
            if abs(u@v)<np.cos(np.radians(15)):continue
            n=np.array([-u[1],u[0]]);signed=(a2-a1)@n;w=abs(signed);wc=w-2*off
            if not 0<=wc<cap:continue
            ends=sorted([(a2-a1)@u,(b2-a1)@u]);lo=max(0,ends[0]);hi=min(l1,ends[1])
            if hi-lo<.02:continue
            center=a1+u*(lo+hi)/2+n*signed/2
            # Match reference duplicate suppression; use unrounded geometry
            # for control and record the rounded reference representation.
            if any(np.linalg.norm(np.round(center,3)-np.round(q['centre'],3))<.03 and abs(round(w,3)-round(q['width'],3))<1e-6 for q in raw):continue
            if tuple(u)<tuple(-u):u=-u
            raw.append(dict(centre=center.tolist(),axis=u.tolist(),half_length=float((hi-lo)/2),width=float(w),free_centre_width=float(wc),wall_pairs=[[i,j]]))
    merged=[]
    for p in raw:
        p=dict(p)
        changed=True
        while changed:
            changed=False
            for j,q in enumerate(merged):
                u=np.asarray(p['axis']);v=np.asarray(q['axis']);delta=np.asarray(q['centre'])-p['centre']
                if abs(u@v)<1-1e-10 or abs(np.cross(u,delta))>1e-8 or abs(p['width']-q['width'])>1e-8:continue
                a,b=-p['half_length'],p['half_length'];c,d=float(delta@u)-q['half_length'],float(delta@u)+q['half_length']
                if max(a,c)>min(b,d)+1e-8:continue
                lo,hi=min(a,c),max(b,d)
                p.update(centre=(np.asarray(p['centre'])+u*(lo+hi)/2).tolist(),half_length=(hi-lo)/2,
                         wall_pairs=p['wall_pairs']+q['wall_pairs'])
                merged.pop(j);changed=True;break
        merged.append(p)
    return raw,merged

def entry(path,passage,radius,lookahead):
    """First forward polyline/rectangle intersection; no planner or labels."""
    path=np.asarray(path,float);u=np.asarray(passage['axis']);n=np.array([-u[1],u[0]])
    local=np.stack([(path-passage['centre'])@u,(path-passage['centre'])@n],axis=-1)
    half=np.array([passage['half_length']+radius,passage['width']/2]);arc=0.
    for j in range(len(path)-1):
        delta=local[j+1]-local[j];length=np.linalg.norm(path[j+1]-path[j])
        if length<1e-14:continue
        low,high=0.,1.
        for axis in range(2):
            if abs(delta[axis])<1e-14:
                if abs(local[j,axis])>half[axis]:high=-1.;break
            else:
                a,b=sorted(((-half[axis]-local[j,axis])/delta[axis],(half[axis]-local[j,axis])/delta[axis]))
                low=max(low,a);high=min(high,b)
        if low<=high and arc+low*length<=lookahead+1e-12:
            return float(arc+low*length),int(np.sign(delta[0]))
        arc+=length
        if arc>lookahead:break
    return None
