#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

H=Path(__file__).parent
D=H.parent
ROOT=D.parent
SH=D/'orthoflow3_shared_eta_codebook_v1'
GEN=D/'orthoflow3_general_basin_geometry_v1'
DEF=D/'orthoflow3_deformable_shared_modes_v1'
POOL=D/'double_bottleneck_initial_state_coverage/data'
AFF=np.array([.875,0.,.375],float)
SCALE=np.array([.75,1.,.75],float)

def read(p): return list(csv.DictReader(open(p)))
def write(name,rows,fields=None):
 p=H/name; fields=fields or (list(rows[0]) if rows else ['state_id'])
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(name,x):(H/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(s):return hashlib.sha256(str(s).encode()).hexdigest()
def families(pool):
 m=json.load(open(pool/'manifest.json')); out={}
 for r in m['files']:out.setdefault(r['family_id'],r)
 return out
def order(ids,tag):return sorted(ids,key=lambda x:sha(tag+'|'+x))

def kabsch(X,Y):
 x=X-X.mean(0);y=Y-Y.mean(0);U,_,Vt=np.linalg.svd(x.T@y);R=Vt.T@U.T
 if np.linalg.det(R)<0:Vt[-1]*=-1;R=Vt.T@U.T
 s=np.sum((x@R.T)*y)/np.sum(x*x);b=Y.mean(0)-s*(X.mean(0)@R.T)
 return s*R,b
def aniso_fit(X,Y):
 A0,b0=kabsch(X,Y);s0=np.linalg.svd(A0,compute_uv=False).mean();r0=Rotation.from_matrix(A0/s0).as_rotvec()
 def unpack(q):
  R=Rotation.from_rotvec(q[:3]).as_matrix();s=np.exp(q[3:6]);A=np.diag(s)@R;b=q[6:9];return A,b
 def fun(q):
  A,b=unpack(q);return (X@A.T+b-Y).ravel()
 q=np.r_[r0,np.log(np.repeat(max(s0,1e-4),3)),b0]
 lo=np.r_[[-math.pi]*3,[-math.log(10)]*3,[-3]*3];hi=np.r_[[math.pi]*3,[math.log(10)]*3,[3]*3]
 q=least_squares(fun,q,bounds=(lo,hi),max_nfev=50000,xtol=1e-13,ftol=1e-13,gtol=1e-13).x
 A,b=unpack(q);return A,b
def regularized_affine(X,Y,Aref,lamb=.25,max_cond=10):
 # Closed-form ridge toward the anisotropic map, followed by deterministic
 # singular-value clipping and an optimal translation refit.
 xc=X-X.mean(0);yc=Y-Y.mean(0)
 A=((np.linalg.solve(xc.T@xc+lamb*np.eye(3),xc.T@yc+lamb*Aref.T)).T)
 U,s,Vt=np.linalg.svd(A);floor=s.max()/max_cond;s=np.maximum(s,floor);A=U@np.diag(s)@Vt
 b=Y.mean(0)-X.mean(0)@A.T
 return A,b

def main():
 H.mkdir(parents=True,exist_ok=True)
 # Freeze source-isolated split without outcomes.
 old4=json.load(open(GEN/'double_bottleneck_state_panel.json'));dev_groups={x['family_id'] for x in old4}
 tr=families(POOL/'train_pool');va=families(POOL/'validation_pool');te=families(POOL/'untouched_test_pool')
 trs=order(tr,'db_mode_transfer_train')[:64];vas=order(va,'db_mode_transfer_val')[:16]
 tes=[x for x in order(te,'db_mode_transfer_test') if x not in dev_groups][:16]
 states=[]
 for split,ids,pool in [('train',trs,POOL/'train_pool'),('val',vas,POOL/'validation_pool'),('test',tes,POOL/'untouched_test_pool')]:
  for j,f in enumerate(ids):
   states.append(dict(state_id=f'DB_MODE_{split}_{j:03d}',split=split,family_id=f,source_group=f,dataset=str(pool),initial_flow_root=2026092906,future_root=2026092907,rng_namespace={'train':10000,'val':20000,'test':30000}[split]+j,selection='outcome-blind SHA256 order',h_schema='obs72+current_raw_flow8'))
 assert len(states)==96 and len({x['source_group'] for x in states})==96
 assert not ({x['source_group'] for x in states if x['split'] in ('val','test')} & dev_groups)
 dump('db_state_split.json',{'frozen_before_new_outcomes':True,'counts':{'train':64,'val':16,'test':16},'feature_schema':'flattened true-t0 observation[4,18] + frozen current raw Flow action[4,2] = 80D','transform_development_source_groups':sorted(dev_groups),'states':states})
 # Existing exact DB inventory is authoritative and provenance-aware.
 inv=[r for r in read(GEN/'exact_q64_inventory.csv') if r['scenario']=='DoubleBottleneck_4A']
 write('db_exact_q64_inventory.csv',inv)
 # Frozen Toy anchors and deterministic DB B63 clusters.
 cb=read(SH/'codebook_eta.csv');E=np.array([[float(r[f'eta{i}']) for i in (1,2,3)] for r in cb]);X=(E-AFF)/SCALE
 pos=np.unique(np.array([[float(r[f'eta{i}']) for i in (1,2,3)] for r in inv if r['B63']=='True']),axis=0);Z=(pos-AFF)/SCALE
 orderz=np.lexsort((Z[:,2],Z[:,1],Z[:,0]));init=Z[orderz[np.linspace(0,len(orderz)-1,12,dtype=int)]];C,label=kmeans2(Z,init,minit='matrix',iter=100)
 # Freeze the prior deterministic correspondence; it is geometry-only and
 # was computed before this experiment's new state outcomes.
 prior=read(DEF/'cross_scenario_mode_alignment.csv');assign=np.array([int(r['assigned_cluster']) for r in sorted(prior,key=lambda r:int(r['mode_id']))])
 Y=C[assign]
 As,bs=kabsch(X,Y);Aa,ba=aniso_fit(X,Y);Af,bf=regularized_affine(X,Y,Aa)
 # Eta3 is nonnegative in frozen E_bridge.  A global translation correction
 # is part of each transform (not a per-mode clip); it makes every anchor
 # feasible while preserving the shared affine form.
 for A,b in ((As,bs),(Aa,ba),(Af,bf)):
  min_eta3=float(np.min((X@A.T+b)*SCALE+AFF,axis=0)[2])
  if min_eta3<0:b[2]+=(-min_eta3+1e-6)/SCALE[2]
 models={'similarity':(As,bs),'anisotropic':(Aa,ba),'regularized_affine':(Af,bf)}
 # Bounded scenario-level residual is the final allowed form.  It maps to
 # cluster centers while retaining the globally shared affine component.
 pred=X@Af.T+bf;delta=Y-pred;dn=np.linalg.norm(delta,axis=1)
 assert np.max(dn)<=.20+1e-12
 models['affine_plus_mode_residual']=(Af,bf)
 out=[];res=[]
 for m in range(12):
  row={'mode_id':m,'toy_eta1':E[m,0],'toy_eta2':E[m,1],'toy_eta3':E[m,2],'assigned_db_cluster':int(assign[m]),'db_cluster_z1':Y[m,0],'db_cluster_z2':Y[m,1],'db_cluster_z3':Y[m,2],'db_eta1':(Y[m]*SCALE+AFF)[0],'db_eta2':(Y[m]*SCALE+AFF)[1],'db_eta3':(Y[m]*SCALE+AFF)[2],'delta_norm':dn[m]}
  out.append(row)
  for name,(A,b) in models.items():
   p=X[m]@A.T+b
   if name=='affine_plus_mode_residual':p=p+delta[m]
   res.append(dict(model=name,mode_id=m,residual_norm=float(np.linalg.norm(p-Y[m])),pred_z1=p[0],pred_z2=p[1],pred_z3=p[2],pred_eta1=(p*SCALE+AFF)[0],pred_eta2=(p*SCALE+AFF)[1],pred_eta3=(p*SCALE+AFF)[2]))
 write('mode_correspondence.csv',out);write('transform_residuals.csv',res)
 trans={}
 for name,(A,b) in models.items():
  P=X@A.T+b
  if name=='affine_plus_mode_residual':P=P+delta
  trans[name]={'A_normalized':A.tolist(),'b_normalized':b.tolist(),'condition_number':float(np.linalg.cond(A)),'residual_median':float(np.median(np.linalg.norm(P-Y,axis=1))),'residual_max':float(np.max(np.linalg.norm(P-Y,axis=1))),'eta':(P*SCALE+AFF).tolist()}
 trans['affine_plus_mode_residual']['delta_normalized']=delta.tolist();trans['affine_plus_mode_residual']['max_delta_norm']=float(dn.max())
 dump('toy_to_db_transform.json',{'status':'CANDIDATES_FROZEN_FOR_TRANSFORM_DEV_PILOT','normalization':{'center':AFF.tolist(),'scale':SCALE.tolist()},'correspondence_source':'159 exact B63 eta / four quarantined transform-development true-t0 states','mode_ids':list(range(12)),'candidates':trans})
 dump('working_state.json',{'status':'TRANSFORM_DEV_PILOT_READY','completed':['toy_freeze','db_inventory','h_schema_audit','state_split_freeze','transform_candidates'],'next_action':'exact Q64 transform-development pilot'})
 dump('evidence_index.json',{'toy_codebook':str(SH/'codebook_eta.csv'),'db_exact_source':str(GEN/'exact_q64_inventory.csv'),'prior_alignment':str(DEF/'cross_scenario_mode_alignment.csv'),'db_records':len(inv),'db_B63':sum(r['B63']=='True' for r in inv),'db_states':len(set(r['state_id'] for r in inv))})
 write('experiment_ledger.csv',[{'stage':'freeze_and_audit','status':'COMPLETE','new_continuations':0,'decision':'64/16/16 split; 80D DB h; four external transform-dev states'},{'stage':'transform_candidates','status':'COMPLETE','new_continuations':0,'decision':'similarity, anisotropic, cond-controlled affine, bounded per-mode residual'}])
 print(json.dumps({'states':len(states),'db_exact':len(inv),'db_B63':sum(r['B63']=='True' for r in inv),'transform_candidates':{k:{q:v[q] for q in ('condition_number','residual_median','residual_max')} for k,v in trans.items()}},indent=2))
if __name__=='__main__':main()
