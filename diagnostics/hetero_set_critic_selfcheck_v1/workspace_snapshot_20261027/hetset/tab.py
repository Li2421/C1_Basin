import json,numpy as np
KEYS=['safety']+[f'eta{i}' for i in range(16)]+[f'sE{i:02d}' for i in range(32)]
def load(files,nstates=24):
 T={}
 for f in files:
  for l in open(f):
   d=json.loads(l);t=T.setdefault(d['scene'],np.full((3,nstates,49),np.nan));t[d['controller'],d['state'],KEYS.index(d['action'])]=d['success']
 return T
def rate(t,states,a):return np.nanmean(t[:,states,a])
