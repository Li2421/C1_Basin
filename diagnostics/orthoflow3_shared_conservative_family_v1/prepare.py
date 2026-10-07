#!/usr/bin/env python3
"""Freeze exact evidence, state folds, and candidate-agnostic oracle subsets."""
import csv, json, hashlib, sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

HERE=Path(__file__).resolve().parent
D=HERE.parent
G=D/'orthoflow3_general_basin_geometry_v1'
EP=D/'orthoflow3_ep0082_failure_intrusion_v1'
POINT=D/'orthoflow3_true_t0_point_learning_v1'
BASIS=D/'double_bottleneck_eta_basis_redesign/tools/bases.py'
SHA='51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38'
AFF=np.array([.875,0,.375]); SCALE=np.array([.75,1,.75])
HS=np.unique(np.array(json.load(open(D/'orthoflow3_t0_multiball_basin_learning_v1/geometry_constants.json'))['halfspaces']),axis=0)
TOY=[f'T0_WIDE_perm{i:02d}_ep{e:04d}' for i,e in enumerate((82,217,182,205,195,179,74,139))]
DB=[f'DB_T0_{i}' for i in range(4)]
STATES=TOY+DB

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(open(p)))
def write(p,rows,fields=None):
 p=HERE/p;p.parent.mkdir(parents=True,exist_ok=True)
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(p,x):
 p=HERE/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def eta(r):return np.array([float(r[f'eta{i}']) for i in (1,2,3)])
def key(v):return np.asarray(v,dtype='<f8').tobytes().hex()
def inside(v):
 z=(np.atleast_2d(v)-AFF)/SCALE
 return np.all(z@HS[:,:3].T+HS[:,3]<=1e-10,axis=1)
def hval(s):return int(hashlib.sha256(s.encode()).hexdigest()[:16],16)

def merge_inventory():
 assert sha(BASIS)==SHA
 base=read(G/'exact_q64_inventory.csv'); store={(r['state_id'],r['eta_key']):dict(r) for r in base}
 added=0; conflicts=[]
 if (EP/'exact_q64_ep0082.csv').exists():
  for r in read(EP/'exact_q64_ep0082.csv'):
   k=(r['state_id'],r['eta_key']); old=store.get(k)
   if old:
    if int(old['successes'])!=int(r['successes']) or int(old['trials'])!=int(r['trials']):conflicts.append(k)
    continue
   if int(r['trials'])!=64:continue
   # Keep the general-audit schema and mark these post-audit exact points as heldout.
   row={c:r.get(c,'') for c in base[0]};row.update(geometry_role='heldout',phases='ep0082_intrusion_round1',raw_phases='ep0082_intrusion_round1',retained_validation_batches='')
   store[k]=row;added+=1
 assert not conflicts,conflicts
 out=[]
 for k,r in sorted(store.items()):
  r['in_E_bridge']=str(bool(inside(eta(r))[0]))
  out.append(r)
 fields=list(base[0]);write('exact_q64_inventory.csv',out,fields)
 dump('inventory_integrity.json',dict(basis_sha256=sha(BASIS),general_inventory_sha256=sha(G/'exact_q64_inventory.csv'),ep0082_inventory_sha256=sha(EP/'exact_q64_ep0082.csv') if (EP/'exact_q64_ep0082.csv').exists() else None,
  general_exact=len(base),supplemental_ep0082_exact=added,merged_exact=len(out),conflicts=0,in_domain=sum(r['in_E_bridge']=='True' for r in out),B63=sum(r['B63']=='True' for r in out)))
 return out,added

def select_oracle(rr,budget,target_keys):
 # Exclude previous analytic-candidate retained-validation points and the new ep0082
 # topology slice from fitting. They remain strict heldout evidence.
 def eligible(r):
  p=(r.get('phases','')+';'+r.get('retained_validation_batches','')).lower()
  bad=('retained_validation','minimal_validation','polyhedral_validation','bounded_validation','ep0082_intrusion')
  return r['in_E_bridge']=='True' and not any(x in p for x in bad)
 cand=[r for r in rr if eligible(r)]
 assert len(cand)>=budget,(rr[0]['state_id'],len(cand),budget)
 bykey={r['eta_key']:r for r in cand}; selected=[]
 # One defensible exact robust anchor counts toward the oracle budget.
 anchor=None
 for k in target_keys:
  if k in bykey and bykey[k]['B63']=='True':anchor=bykey[k];break
 if anchor is None:
  pos=[r for r in cand if r['B63']=='True'];neg=[r for r in cand if r['B63']=='False']
  assert pos
  if neg:
   N=np.array([eta(r) for r in neg]);anchor=max(pos,key=lambda r:(float(np.min(np.linalg.norm((N-eta(r))/SCALE,axis=1))),-hval(r['eta_key'])))
  else:anchor=min(pos,key=lambda r:r['eta_key'])
 selected.append(anchor)
 # Fixed nearby/far negative evidence, allowed by the preregistered oracle policy.
 neg=[r for r in cand if r['B63']=='False']
 if neg:
  d=[np.linalg.norm((eta(r)-eta(anchor))/SCALE) for r in neg]
  for j in (int(np.argmin(d)),int(np.argmax(d))):
   if neg[j] not in selected:selected.append(neg[j])
 # Then deterministic candidate-agnostic maximin eta coverage. Common/cross-transfer
 # records receive a stable small priority bonus but labels do not enter this step.
 while len(selected)<budget:
  S=np.array([(eta(r)-AFF)/SCALE for r in selected])
  best=None
  for r in cand:
   if r in selected:continue
   z=(eta(r)-AFF)/SCALE;dist=float(np.min(np.linalg.norm(S-z,axis=1)))
   phase=(r.get('phases','')+';'+r.get('sources','')).lower()
   bonus=.02 if ('cross_transfer' in phase or 'common_core' in phase) else .01 if ('coverage' in phase or 'geometry1' in phase or 'initial_q64' in phase) else 0
   score=(dist+bonus,-hval(r['eta_key']))
   if best is None or score>best[0]:best=(score,r)
  selected.append(best[1])
 assert len({r['eta_key'] for r in selected})==budget
 return selected

def main():
 if (HERE/'cv_folds.json').exists():raise FileExistsError('Frozen evidence protocol already exists')
 inv,added=merge_inventory();dense=[r for r in inv if r['state_id'] in STATES and r['in_E_bridge']=='True']
 assert set(STATES)=={r['state_id'] for r in dense}
 # Fixed scenario-stratified outer folds, frozen before any new family fitting.
 folds=[
  {'fold':0,'heldout':[TOY[0],TOY[3],TOY[6],DB[0]]},
  {'fold':1,'heldout':[TOY[1],TOY[4],TOY[7],DB[1]]},
  {'fold':2,'heldout':[TOY[2],TOY[5],DB[2],DB[3]]},
 ]
 for f in folds:f['development']=[s for s in STATES if s not in f['heldout']]
 dump('cv_folds.json',dict(protocol='Three frozen state-level outer folds; each includes both compatible scenarios. Global p/erosion/cut count selected on development states only.',folds=folds))
 target_rows=read(POINT/'selected_eta_targets.csv');target_keys=defaultdict(list)
 for r in target_rows:
  if r['state_id'] in TOY:
   v=np.array([float(r[f'target_eta{i}']) for i in (1,2,3)]);target_keys[r['state_id']].append(key(v))
 records=[];sub={}
 inventory=[]
 for sid in STATES:
  rr=[r for r in dense if r['state_id']==sid];pos=sum(r['B63']=='True' for r in rr);scenario=rr[0]['scenario']
  inventory.append(dict(state_id=sid,scenario=scenario,source_group=rr[0]['source_group'],exact_Q64=len(rr),B63=pos,non_B63=len(rr)-pos,h=rr[0]['h'],conditioning=rr[0]['conditioning'],controller_semantics_hash=rr[0]['controller_semantics_hash']))
  sel32=select_oracle(rr,32,target_keys[sid]);sub[sid]={}
  for b in (16,24,32):
   keys=[r['eta_key'] for r in sel32[:b]];sub[sid][str(b)]=keys
   for rank,r in enumerate(sel32[:b]):records.append(dict(state_id=sid,scenario=scenario,budget=b,rank=rank,eta_key=r['eta_key'],B63=r['B63'],Q64=r['Q64'],phases=r['phases']))
 write('scenario_state_inventory.csv',inventory)
 dump('oracle_budget_subsamples.json',dict(selection='One known robust anchor; nearest/farthest known negatives if available; candidate-agnostic normalized-eta maximin with fixed provenance bonus. Nested16/24/32. Prior retained validations and new ep0082 slice excluded from fitting.',states=sub))
 write('oracle_budget_subsamples.csv',records)
 dump('evidence_freeze.json',dict(states=12,scenarios=sorted(set(r['scenario'] for r in inventory)),merged_exact=len(inv),panel_exact=len(dense),supplemental_ep0082_exact=added,folds_sha256=sha(HERE/'cv_folds.json'),oracle_subsamples_sha256=sha(HERE/'oracle_budget_subsamples.json')))
 print(json.dumps(dict(panel_exact=len(dense),B63=sum(r['B63']=='True' for r in dense),added_ep0082=added,states=len(STATES))))

if __name__=='__main__':main()
