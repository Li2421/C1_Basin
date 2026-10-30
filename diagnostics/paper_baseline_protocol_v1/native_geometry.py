"""Exact segment pair/AABB distances for recorded native trajectories."""
import guard
from guard import HERE
import sys,json
base=HERE/'inputs/diagnostics/paper_baselines_20261028';sys.path[:0]=[str(HERE/'native_deps')]+[str(p)for p in base.glob('deps*')if p.is_dir()]
import numpy as np

def segment_box_distance(a,b,lo,hi):
    # Broadcast leading dimensions. Minimize convex piecewise-quadratic squared distance.
    v=b-a;times=[np.zeros(a.shape[:-1]),np.ones(a.shape[:-1])]
    for axis in (0,1):
        for bound in (lo[...,axis],hi[...,axis]):
            times.append(np.clip(np.divide(bound-a[...,axis],v[...,axis],out=np.zeros_like(a[...,axis]),where=v[...,axis]!=0),0,1))
    ts=np.sort(np.stack(times,axis=-1),axis=-1);mid=(ts[...,:-1]+ts[...,1:])/2
    pm=a[...,None,:]+mid[...,None]*v[...,None,:]
    edges=np.where(pm<lo[...,None,:],lo[...,None,:],np.where(pm>hi[...,None,:],hi[...,None,:],pm))
    active=(pm<lo[...,None,:])|(pm>hi[...,None,:])
    aa=np.where(active,a[...,None,:]-edges,0);vv=np.where(active,v[...,None,:],0)
    numerator=-np.sum(aa*vv,axis=-1);denom=np.sum(vv*vv,axis=-1)
    star=np.divide(numerator,denom,out=mid.copy(),where=denom>0);star=np.maximum(ts[...,:-1],np.minimum(ts[...,1:],star))
    candidates=np.concatenate([ts,star],axis=-1);points=a[...,None,:]+candidates[...,None]*v[...,None,:]
    dd=np.maximum(np.maximum(lo[...,None,:]-points,points-hi[...,None,:]),0)
    return np.sqrt(np.min(np.sum(dd*dd,axis=-1),axis=-1))

def check(positions,radius,boxes):
    pos=np.asarray(positions);n=pos.shape[1];ii,jj=np.triu_indices(n,1);pair=np.inf;wall=np.inf
    for start in range(0,len(pos)-1,64):
        before=pos[start:min(start+64,len(pos)-1)];after=pos[start+1:min(start+65,len(pos))]
        if len(ii):
            a=before[:,ii]-before[:,jj];v=after[:,ii]-after[:,jj]-a
            t=np.clip(np.divide(-np.sum(a*v,axis=-1),np.sum(v*v,axis=-1),out=np.zeros(a.shape[:-1]),where=np.sum(v*v,axis=-1)>0),0,1)
            pair=min(pair,float(np.min(np.linalg.norm(a+t[...,None]*v,axis=-1)-2*radius)))
        if len(boxes):
            bb=np.asarray(boxes);shape=(len(before),n,len(bb),2);a=np.broadcast_to(before[:,:,None,:],shape);b=np.broadcast_to(after[:,:,None,:],shape)
            wall=min(wall,float(np.min(segment_box_distance(a,b,bb[None,None,:,:2],bb[None,None,:,2:])-radius)))
    return {'pair_segment_clearance':pair if np.isfinite(pair)else None,'wall_segment_clearance':wall if np.isfinite(wall)else None,'interpretation':'Finite stored trajectory, linearly interpolated motion. Negative residuals are reported; no mathematical certificate about continuous controller dynamics.'}

def main():
    rows={}
    d=np.load(HERE/'native_reproduction/dac/native_trace.npz');rows['dac_author_si']=check(d['positions'].transpose(1,0,2),.2,[[-2,.2,2,.4],[-2,-.4,2,-.2]])
    m=np.load(HERE/'native_reproduction/mgr/native_trace.npz')
    import yaml
    instance=yaml.safe_load((HERE/'native_reproduction/mgr/author/instances/rect_maps/RectEnv15/agents60/RectEnv_15_60_0.yaml').read_text())
    boxes=[]
    for o in instance['obstacles']:
        c=np.asarray(o['center'])*.4;wh=np.array([o['width'],o['height']])*.4;boxes.append([*(c-wh/2),*(c+wh/2)])
    rows['mgr_author_unicycle_tracking']=check(m['positions'],.2,boxes)
    d=np.load(HERE/'native_reproduction/chen/native_trace.npz');boxes=[[p[:,0].min(),p[:,1].min(),p[:,0].max(),p[:,1].max()]for p in d['obstacles']]
    rows['chen_impc_ob_si_tracking']=check(d['positions'],float(d['robot_radius']),boxes)
    (HERE/'NATIVE_GEOMETRY_CHECK.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n');print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
