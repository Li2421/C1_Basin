"""Prespecified scale-independent scene descriptors and raw-range guards."""
from common import *
import numpy as np

EXPECTED={
'lit_narrow_passage':[1.099,-1.087,.288,3.651,0,0,2.303,0],
'lit_crowded_random':[2.079,-2.058,0,2.960,0,0,2.303,0],
'lit_circle_five':[1.609,-1.374,-4.673,.866,.165,1,1.705,0],
'lit_swap8_free':[2.079,-3.687,-.693,1.320,0,0,2.303,0],
'lit_swap8_obstacles':[2.079,-4.025,-.693,1.353,0,4,2.313,0],
'lit_swap8_moving':[2.079,-3.687,-.693,1.320,0,0,2.303,0],
'xsc_swap_n4':[1.386,-2.983,-.693,1.392,0,0,2.303,0],
'xsc_swap_n6':[1.792,-3.398,-.693,1.386,0,0,2.303,0],
'xsc_swap_n10':[2.303,-3.913,-.693,1.387,0,0,2.303,0],
'xsc_swap_n12':[2.485,-4.093,-.693,1.388,0,0,2.303,0],
'xsc_swap_n8_dense':[2.079,-3.332,-.693,1.389,0,0,2.303,0],
'xsc_gap_n2':[.693,-4.356,-1.856,1.144,.001,3,.661,1],
'xsc_gap_n4':[1.386,-4.302,-1.856,1.614,.003,3,.661,1],
'xsc_gap_n6':[1.792,-4.286,-1.856,2.020,.004,3,.661,1],
'xsc_wide_n4':[1.386,-4.241,-1.856,1.646,.003,3,1.322,1],
'xsc_double_n4':[1.386,-4.395,-1.856,1.641,.003,5,.446,2],
'xsc_offset_n4':[1.386,-4.395,-1.856,1.641,.003,5,.446,2],
'xsc_three_doors_n4':[1.386,-4.241,-1.856,1.646,.003,5,.446,1]}

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

def guard(values,lo,hi):
    values=np.asarray(values,float);lo=np.asarray(lo,float);hi=np.asarray(hi,float)
    # The hi==lo case is covered exactly, with no numerical tolerance.
    dims=(values<lo-.25*(hi-lo))|(values>hi+.25*(hi-lo))
    return np.any(dims,axis=-1),dims

def ranges(inputs,sources,field='c_prime'):
    c=np.asarray([inputs[n][field]for n in sources]);h=np.vstack([inputs[n]['h']for n in sources])
    return dict(c_lo=c.min(0).tolist(),c_hi=c.max(0).tolist(),h_lo=h.min(0).tolist(),h_hi=h.max(0).tolist())

def target_guard(inputs,name,bounds,field='c_prime'):
    d=inputs[name];out,dim=guard(d[field],bounds['c_lo'],bounds['c_hi']);st,sd=guard(d['h'],bounds['h_lo'],bounds['h_hi'])
    return dict(scene=name,descriptor=d[field],**bounds,scene_out_of_range=bool(out),scene_trigger_dimensions=(np.flatnonzero(dim)+1).tolist(),
        states_out_of_range=st.tolist(),state_trigger_dimensions=[(np.flatnonzero(v)+1).tolist()for v in sd],state_indices=d['state_indices'])
