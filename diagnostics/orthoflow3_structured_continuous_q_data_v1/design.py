#!/usr/bin/env python3
"""Freeze structured Toy state×eta matrix and emit global-cache requests (no rollout)."""
from __future__ import annotations
import csv,hashlib,json,sqlite3,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import qmc

ROOT=Path('/home/zhihan/research/Basin_C1');D=ROOT/'diagnostics';H=Path(__file__).parent
TOY=D/'orthoflow3_shared_eta_codebook_v1';AUD=D/'orthoflow3_fixed_eta_state_to_q_identifiability_v2'
DB=ROOT/'shared_rollout_db/rollout.sqlite';sys.path.insert(0,str(ROOT))
from shared_rollout_db.src.rollout_db import eta_identity,canonical

CENTER=np.array([.875,0.,.375]);SCALE=np.array([.75,1.,.75]);CTL='ctl_df736b67f6410260d87812e0a76946af7152d147c9a0a25189d1908a92567b34'
QUOTA={'train':{'global':18,'transition':14,'boundary':8},'val':{'global':6,'transition':4,'boundary':2},'test':{'global':16,'transition':9,'boundary':7}}
STATE_N={'train':48,'val':12,'test':16}

def dump(name,x):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def write(name,rows):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True);fields=list(rows[0]) if rows else ['empty']
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def h(s):return hashlib.sha256(s.encode()).hexdigest()
def z_of(e):return (np.asarray(e,float)-CENTER)/SCALE
def in_bridge(e):
 x,y,z=np.asarray(e,float);return -1e-12<=x<=1.25+1e-12 and abs(y)<=min(.5,x)+1e-12 and -1e-12<=z<=min(.75,1.5*x)+1e-12
def farthest(points,n,tag):
 p=np.asarray(points,float);assert len(p)>=n
 first=min(range(len(p)),key=lambda i:h(tag+'|'+','.join(f'{v:.17g}' for v in p[i])))
 chosen=[first];d=np.linalg.norm(p-p[first],axis=1);d[first]=-1
 while len(chosen)<n:
  mx=d.max();cand=np.where(np.isclose(d,mx,rtol=0,atol=1e-14))[0];i=min(cand,key=lambda j:h(tag+'|'+str(j)))
  chosen.append(int(i));d=np.minimum(d,np.linalg.norm(p-p[i],axis=1));d[chosen]=-1
 return chosen

def freeze_states():
 src=json.load(open(TOY/'state_split.json'))['states'];out=[]
 for sp in ('train','val','test'):
  q=[x for x in src if x['split']==sp];q=sorted(q,key=lambda x:h('structured-q-v1|'+sp+'|'+x['source_group']+'|'+x['state_id']))[:STATE_N[sp]]
  assert len(q)==STATE_N[sp];out.extend(q)
 groups={sp:{x['source_group'] for x in out if x['split']==sp} for sp in STATE_N}
 assert not groups['train']&groups['val'] and not groups['train']&groups['test'] and not groups['val']&groups['test']
 payload={'frozen_before_eta_design':True,'selection':'deterministic SHA256 order within prior source-isolated split; outcome-blind',
          'counts':STATE_N,'source_group_overlap':{'train_val':0,'train_test':0,'val_test':0},'states':out}
 dump('toy_state_split.json',payload);return out

def transition_pool():
 rr=list(csv.DictReader(open(AUD/'coverage_audit_toy.csv')));out=[]
 for r in rr:
  if r['state_dependent_b15']=='True':
   out.append({'eta_uid':r['eta_uid'],'eta':np.array([float(r['eta1']),float(r['eta2']),float(r['eta3'])]),
               'train_q_std':float(r['train_q_std']),'train_b15_prevalence':float(r['train_b15_prevalence']),
               'train_clear_failure_count':int(r['train_clear_failure_count']),'train_robust_count':int(r['train_robust_count'])})
 assert len(out)==27
 return out

def assign_transition(pool):
 z=np.asarray([z_of(x['eta']) for x in pool]);tr=farthest(z,QUOTA['train']['transition'],'transition-train')
 rem=[i for i in range(len(pool)) if i not in tr];vv=farthest(z[rem],QUOTA['val']['transition'],'transition-val');va=[rem[i] for i in vv];te=[i for i in rem if i not in va]
 assert len(te)==QUOTA['test']['transition']
 ans=[]
 for sp,ix in [('train',tr),('val',va),('test',te)]:
  for i in ix:ans.append({**pool[i],'split':sp,'probe_type':'transition','origin':'TRAIN-only repeated exact eta with failure/B15 state variation'})
 return ans

def global_scaffold(trans):
 sob=qmc.Sobol(3,scramble=False).random_base2(13);cand=[]
 for u in sob[1:]:
  e=np.array([1.25*u[0],u[1]-.5,.75*u[2]])
  if in_bridge(e):cand.append(e)
 tz=np.asarray([z_of(x['eta']) for x in trans]);cand=[e for e in cand if np.min(np.linalg.norm(tz-z_of(e),axis=1))>=.12]
 sel=farthest(np.asarray([z_of(e) for e in cand]),sum(QUOTA[s]['global'] for s in QUOTA),'global-scaffold')
 pts=[cand[i] for i in sel];order=sorted(range(len(pts)),key=lambda i:h('global-split|'+','.join(f'{v:.17g}' for v in pts[i])))
 ans=[];pos=0
 for sp in ('train','val','test'):
  for i in order[pos:pos+QUOTA[sp]['global']]:ans.append({'eta':pts[i],'split':sp,'probe_type':'global','origin':'unscrambled Sobol E_bridge scaffold; outcome-blind'})
  pos+=QUOTA[sp]['global']
 return ans

def boundary_probes(trans,glob):
 # Fixed directions/radii; coordinates use TRAIN-only transition evidence, never state-specific outcomes.
 dirs=[]
 for a in np.eye(3):dirs.extend([a,-a])
 dirs += [np.array(x,float)/np.linalg.norm(x) for x in [(1,1,0),(1,-1,0),(1,0,1),(1,0,-1),(0,1,1),(0,1,-1),(1,1,1),(1,-1,1)]]
 existing={sp:[z_of(x['eta']) for x in trans+glob if x['split']==sp] for sp in QUOTA};ans=[]
 for sp in ('train','val','test'):
  anchors=[x for x in trans if x['split']==sp];cand=[]
  for ai,a in enumerate(anchors):
   az=z_of(a['eta'])
   for ri,radius in enumerate((.075,.10,.125,.15)):
    for di,d in enumerate(dirs):
     z=az+radius*d;e=CENTER+SCALE*z
     if not in_bridge(e):continue
     # Keep spatial groups split clean with a conservative 0.055 normalized buffer.
     other=[q for osp,v in existing.items() if osp!=sp for q in v]
     if other and np.min(np.linalg.norm(np.asarray(other)-z,axis=1))<.055:continue
     cand.append((e,ai,ri,di))
  # Deterministic farthest coverage, then add immediately so later splits cannot approach it.
  ix=farthest(np.asarray([z_of(x[0]) for x in cand]),QUOTA[sp]['boundary'],'boundary-'+sp)
  for j in ix:
   e,ai,ri,di=cand[j];rec={'eta':e,'split':sp,'probe_type':'boundary','origin':'fixed normalized local cross/direction around TRAIN-evidence transition eta',
                           'boundary_anchor_eta_uid':anchors[ai]['eta_uid'],'boundary_radius_norm':(.075,.10,.125,.15)[ri],'boundary_direction_index':di}
   ans.append(rec);existing[sp].append(z_of(e))
 # Validate every cross-split eta separation.
 allp=trans+glob+ans
 for a in allp:
  for b in allp:
   if a is b or a['split']==b['split']:continue
   assert np.linalg.norm(z_of(a['eta'])-z_of(b['eta']))>=.05-1e-12
 return ans

def annotate(etas):
 out=[]
 for i,x in enumerate(etas):
  e=np.asarray(x.pop('eta'),float);eid=eta_identity(e)[0]
  z=z_of(e);out.append({**x,'probe_id':f"{x['split'][0].upper()}{i:03d}",'eta_uid':eid,'eta1':e[0],'eta2':e[1],'eta3':e[2],'z1':z[0],'z2':z[1],'z3':z[2]})
 return out

def state_uids(states):
 con=sqlite3.connect(DB);con.row_factory=sqlite3.Row;out={}
 for s in states:
  rr=con.execute('''SELECT s.state_uid,s.identity_quality,COUNT(r.rollout_uid)n FROM state_alias a JOIN state s USING(state_uid)
    LEFT JOIN rollout r ON r.state_uid=s.state_uid AND r.controller_uid=? WHERE a.alias=? AND s.scenario_uid=(SELECT scenario_uid FROM scenario WHERE name='ToyGiveWay')
    AND s.identity_quality IN ('CONDITIONING_EXACT','SOURCE_GROUP_STABLE') GROUP BY s.state_uid ORDER BY (s.identity_quality='CONDITIONING_EXACT') DESC,n DESC,s.state_uid''',(CTL,s['state_id'])).fetchall()
  if not rr:raise RuntimeError(('no exact DB state alias',s['state_id']))
  out[s['state_id']]=rr[0]['state_uid']
 con.close();return out

def requests(states,etas):
 su=state_uids(states);req=[];index=[]
 S={sp:[s for s in states if s['split']==sp] for sp in STATE_N};E={sp:[e for e in etas if e['split']==sp] for sp in QUOTA}
 partitions=[('TRAIN_TRAIN',S['train'],E['train'],None),('VAL_VAL',S['val'],E['val'],8),('TESTSTATE_TRAINETA',S['test'],E['train'],16),('TRAINSTATE_TESTETA',S['train'],E['test'],16),('TESTSTATE_TESTETA',S['test'],E['test'],16)]
 for part,ss,ee,fixedn in partitions:
  for s in ss:
   for e in ee:
    n=fixedn if fixedn is not None else (4 if e['probe_type']=='global' else 8)
    seed_keys=[canonical({'future_index':i}) for i in range(n)]
    ri=len(req);req.append({'state_uid':su[s['state_id']],'eta_uid':e['eta_uid'],'controller_uid':CTL,'seed_keys':seed_keys})
    index.append({'request_index':ri,'matrix_partition':part,'state_id':s['state_id'],'state_uid':su[s['state_id']],'state_split':s['split'],'probe_id':e['probe_id'],'eta_uid':e['eta_uid'],
                  'eta_split':e['split'],'probe_type':e['probe_type'],'eta':[e['eta1'],e['eta2'],e['eta3']],'target_trials':n})
 dump('planned_rollouts.json',{'task':'ORTHOFLOW3_STRUCTURED_CONTINUOUS_Q_DATA_V1','controller_uid':CTL,'seed_policy':'Basin_C1_standard_16_matched_continuations_v1','requests':req})
 dump('request_index.json',index)
 return req,index

def main():
 H.mkdir(parents=True,exist_ok=True);states=freeze_states();trans=assign_transition(transition_pool());glob=global_scaffold(trans);bound=boundary_probes(trans,glob);etas=annotate(trans+glob+bound)
 counts=defaultdict(lambda:defaultdict(int))
 for e in etas:counts[e['split']][e['probe_type']]+=1
 assert {s:dict(counts[s]) for s in counts}==QUOTA
 train=np.asarray([[e['z1'],e['z2'],e['z3']] for e in etas if e['split']=='train'])
 test=np.asarray([[e['z1'],e['z2'],e['z3']] for e in etas if e['split']=='test']);d=cdist(test,train).min(1)
 assert d.min()>=.05-1e-12
 write('toy_eta_probe_types.csv',etas)
 dump('toy_eta_split.json',{'normalization':{'center':CENTER.tolist(),'scale':SCALE.tolist()},'selection':'spatially separated fixed shared panel; all outcome-aware choices use TRAIN evidence only',
   'counts':{s:{'total':sum(counts[s].values()),**dict(counts[s])} for s in counts},'test_to_train_distance':{'min':float(d.min()),'q25':float(np.quantile(d,.25)),'median':float(np.median(d)),'q75':float(np.quantile(d,.75)),'max':float(d.max())},
   'assignments':etas})
 req,index=requests(states,etas)
 expected=sum(len(r['seed_keys']) for r in req)
 (H/'data_design.md').write_text(f'''# Structured continuous-Q data design\n\nState panel is a deterministic outcome-blind subset of the prior source-isolated Toy inventory: 48 TRAIN, 12 VAL, 16 TEST. Source-group overlap is zero. TEST states were frozen before eta design.\n\nThe fixed eta panel has 40 TRAIN, 12 VAL, and 32 TEST probes. Thirty-two TEST probes are required by the predeclared K=32 finite-candidate evaluation. TRAIN probes comprise 18 global E_bridge Sobol scaffold, 14 TRAIN-only state-dependent transition eta, and 8 fixed local boundary probes. No probe moves per state.\n\nEta split is spatial: minimum normalized TEST-to-TRAIN distance is {d.min():.6f}. Transition candidates use only TRAIN outcomes from the fixed-eta audit; scaffold is outcome-blind; boundary offsets are fixed geometry around TRAIN-evidence transition eta.\n\nRequested matrix partitions: TRAIN×TRAIN, VAL×VAL, unseen-state×seen-eta, seen-state×unseen-eta, and unseen-state×unseen-eta. Requested seed records before cache reuse: {expected}. TRAIN global pairs request Q4, TRAIN transition/boundary Q8, VAL Q8, and every TEST-bearing pair Q16.\n''')
 dump('working_state.json',{'status':'DESIGN_FROZEN_PREFLIGHT_PENDING','completed':['state_split','eta_panel','eta_spatial_split','planned_requests'],'new_rollout':0,'next_action':'global cache preflight'})
 write('experiment_ledger.csv',[{'stage':'design','status':'COMPLETE','requested_continuations':expected,'new_rollouts':0,'detail':'Frozen before rollout outcomes'}])
 print(json.dumps({'states':STATE_N,'etas':{s:dict(counts[s]) for s in counts},'requests':len(req),'requested_continuations':expected,'test_to_train_eta_distance':json.load(open(H/'toy_eta_split.json'))['test_to_train_distance']},indent=2))
if __name__=='__main__':main()
