#!/usr/bin/env python3
"""Freeze state-independent global and local eta candidate coordinates."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import qmc,norm
H=Path(__file__).parent
AFF=np.array([.875,0.,.375]);SCALE=np.array([.75,1.,.75]);LOW=np.array([.5,-.5,0.]);HIGH=np.array([1.25,.5,.75])
def dump(n,x):(H/n).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def inside(z,eq):return bool(np.all(eq[:,:3]@z+eq[:,3]<=2e-10))
def clearance(z,eq):return float(np.min(-(eq[:,:3]@z+eq[:,3])/np.linalg.norm(eq[:,:3],axis=1)))
def main():
 verts=[np.zeros(3)]+[np.array([a,b,c]) for a in (LOW[0],HIGH[0]) for b in (LOW[1],HIGH[1]) for c in (LOW[2],HIGH[2])];eq=ConvexHull((np.asarray(verts)-AFF)/SCALE).equations
 # One continuous unscrambled global Sobol rejection sequence.  The first 64
 # accepted points are stage 1 and the next 32 are the only stage-2 extension.
 eng=qmc.Sobol(3,scramble=False);glob=[];src=0
 while len(glob)<96:
  for u in eng.random(256):
   eta=np.array([0.,-.5,0.])+np.array([1.25,1.,.75])*u;z=(eta-AFF)/SCALE
   if inside(z,eq):glob.append((src,eta,z))
   src+=1
   if len(glob)>=96:break
 modes=np.asarray(json.load(open(H/'frozen_assets.json'))['transformed_eta'],float);targets=[3]*8+[2]*4;local=[]
 sob=qmc.Sobol(4,scramble=False).random_base2(12)[1:]
 for m,nneed in enumerate(targets):
  got=0
  # The same frozen offset stream is available around every anchor.  Each
  # anchor accepts its first feasible offsets; no outcome enters this choice.
  for j,u in enumerate(sob):
   d=norm.ppf(np.clip(u[:3],1e-8,1-1e-8));dn=np.linalg.norm(d)
   if dn<1e-12 or u[3]<=0:continue
   d/=dn;off=.15*(u[3]**(1/3))*d;eta=modes[m]+off*SCALE;z=(eta-AFF)/SCALE
   if inside(z,eq) and np.linalg.norm(off)>1e-8:
    local.append((m,j,eta,z,float(np.linalg.norm(off))));got+=1
    if got==nneed:break
  assert got==nneed,(m,got)
 rows=[]
 for i,(srcidx,e,z) in enumerate(glob):rows.append(dict(candidate_id=f'G{i:03d}',stage=1 if i<64 else 2,kind='global_sobol',frozen_order=i,source_index=srcidx,anchor_mode='',eta1=e[0],eta2=e[1],eta3=e[2],z1=z[0],z2=z[1],z3=z[2],nearest_mode_distance=float(np.min(np.linalg.norm((modes-e)/SCALE,axis=1))),domain_clearance=clearance(z,eq)))
 for i,(m,srcidx,e,z,r) in enumerate(local):rows.append(dict(candidate_id=f'L{i:03d}',stage=1,kind='local_sobol_offset',frozen_order=64+i,source_index=srcidx,anchor_mode=m,eta1=e[0],eta2=e[1],eta3=e[2],z1=z[0],z2=z[1],z3=z[2],nearest_mode_distance=float(np.min(np.linalg.norm((modes-e)/SCALE,axis=1))),domain_clearance=clearance(z,eq)))
 with open(H/'continuous_candidate_cloud.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 payload={'global_sequence':'scipy.stats.qmc.Sobol(d=3,scramble=False), E_bridge rejection, first 96 accepted','stage1':'G000..G063 plus L000..L031, each Q8','stage2':'G064..G095, each Q8, only if no stage1 B63','local_sequence':'Sobol(d=4,scramble=False); Gaussian direction and radius 0.15*u4^(1/3); deterministic per-mode subsequences; reject outside E_bridge','local_target_counts':[3]*8+[2]*4,'promotion':'at most four per stage among screen >=7/8; rank success count, nearest <=5/8 screened failure distance, domain clearance, frozen order','Q64':'screening seeds 0..7 plus promotion seeds 8..63','unresolved':'no B63 after both stages => CONTINUOUS_ORACLE_UNRESOLVED','candidate_cloud_sha256':hashlib.sha256((H/'continuous_candidate_cloud.csv').read_bytes()).hexdigest(),'normalization':{'center':AFF.tolist(),'scale':SCALE.tolist()},'frozen_before_discrete_outcomes_inspected':True}
 dump('continuous_protocol.json',payload)
 print(json.dumps({'global':len(glob),'local':len(local),'stage1':sum(r['stage']==1 for r in rows),'stage2':sum(r['stage']==2 for r in rows),'sha256':payload['candidate_cloud_sha256']},indent=2))
if __name__=='__main__':main()
