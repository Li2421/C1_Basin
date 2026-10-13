import numpy as np
from study import OUT,read,write
r=read(OUT/'C0_FF_TT_FOCUS.json');y=np.load(OUT/'controller_panels'/(r['panel']+'.npz'))['Y'];aa,bb=np.triu_indices(16,1);rows=[]
for flip in [0,1]:
 a=slice(0,8) if not flip else slice(8,16);b=slice(8,16) if not flip else slice(0,8);L=(y==1).astype(float);U=(y<0).astype(float);q=L[:,:,:,a].mean(-1);qu=U[:,:,:,a].mean(-1);v=L[:,:,:,b].mean(-1);vu=U[:,:,:,b].mean(-1)
 def sign(lo,un,gap):return np.where(lo[:,:,aa]-(lo+un)[:,:,bb]>=gap,1,np.where(lo[:,:,bb]-(lo+un)[:,:,aa]>=gap,-1,0))
 for gap in [.125,.25]:
  d=sign(q,qu,gap);e=sign(v,vu,.125);qual=(d[0]*d[1]==-1);confirmed=qual&(e[0]==d[0])&(e[1]==d[1]);rows.append(dict(half=flip,discovery_gap=gap,discovery_state_pairs=int(qual.sum()),discovery_states=int(qual.any(1).sum()),confirmed_state_pairs=int(confirmed.sum()),confirmed_states=int(confirmed.any(1).sum()),interpretation='Strict interval-separated controller-dependent preference reversal, evaluated on disjoint seed half. Pair counts are dependent; physical states are evidence units.'))
write(OUT/'C0_RANK_REVERSALS.json',rows);print(rows)
