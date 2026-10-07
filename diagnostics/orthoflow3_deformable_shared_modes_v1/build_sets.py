#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
H=Path(__file__).parent;D=H.parent;SH=D/'orthoflow3_shared_eta_codebook_v1'
AFF=np.array([.875,0.,.375]);SCALE=np.array([.75,1.,.75])
def read(p):return list(csv.DictReader(open(p)))
def write(name,rows,fields=None):
 p=H/name;fields=fields or (list(rows[0]) if rows else ['state_id'])
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(name,x):(H/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def status(k,n):
 if (n,k) in ((64,63),(64,64),(32,31),(32,32),(16,16),(8,8)):return 'positive'
 if (n==64 and k<=60) or (n==32 and k<=29) or (n==16 and k<=14) or (n==8 and k<=6):return 'negative'
 return 'ambiguous'
def main():
 rr=[]
 for p in sorted((H/'local_raw').glob('shard*.jsonl')):rr += [json.loads(x) for x in open(p) if x.strip()]
 expect=json.load(open(H/'frozen_manifest.json'))['continuations']
 if len(rr)!=expect:raise RuntimeError(('incomplete local probes',len(rr),expect))
 by=defaultdict(list)
 for r in rr:by[(r['state_id'],int(r['mode_id']),int(r['candidate_index']))].append(r)
 manifest={(r['state_id'],int(r['mode_id']),int(r['candidate_index'])):r for r in read(H/'local_probe_manifest.csv')};new=[]
 for key,v in sorted(by.items()):
  q=manifest[key];out=Counter(x['outcome'] for x in v);k=sum(x['success'] for x in v);new.append(dict(state_id=key[0],assigned_mode=key[1],normalized_distance_to_anchor=q['normalized_radius'],eta1=q['eta1'],eta2=q['eta2'],eta3=q['eta3'],successes=k,trials=len(v),label_status=status(k,len(v)),provenance='frozen_local_sobol',deadlock=out['safe_deadlock'],timeout=out['timeout'],collision=out['collision']))
 rows=read(H/'local_mode_dataset_cached.csv')+new;write('local_mode_dataset.csv',rows);pos=[r for r in rows if r['label_status']=='positive'];neg=[r for r in rows if r['label_status']=='negative'];write('mode_local_positive_sets.csv',pos);write('mode_local_negative_sets.csv',neg)
 sel=[]
 # Reconstruct frozen selected mode from probe manifest plus cached selection logic.
 import sys;sys.path.insert(0,str(H));from prepare import selected_modes
 states,sm=selected_modes();pairs=[r for r in sm if r['split'] in ('train','val') and int(r['selected_mode'])>=0];X=np.load(H/'frozen_arrays.npz');ids=list(X['state_ids']);ix={str(s):i for i,s in enumerate(ids)};anchors=X['anchors'];Ps=[];Ns=[];meta=[];summary=[]
 for r in pairs:
  sid=r['state_id'];m=int(r['selected_mode']);p=[q for q in pos if q['state_id']==sid and int(q['assigned_mode'])==m];n=[q for q in neg if q['state_id']==sid and int(q['assigned_mode'])==m];zp=np.array([[(float(q[f'eta{i}'])-AFF[i-1])/SCALE[i-1] for i in (1,2,3)] for q in p]);zn=np.array([[(float(q[f'eta{i}'])-AFF[i-1])/SCALE[i-1] for i in (1,2,3)] for q in n]);sep=max([np.linalg.norm(zp[i]-zp[j]) for i in range(len(zp)) for j in range(i)]+[0.]);usable=len(zp)>=2 and sep>=.05
  summary.append(dict(state_id=sid,split=r['split'],mode_id=m,positive_count=len(zp),negative_count=len(zn),max_positive_separation=sep,status='LOCAL_SET_USABLE' if usable else 'LOCAL_SET_UNDERRESOLVED'))
  if usable:meta.append((sid,r['split'],m,ix[sid]));Ps.append(zp);Ns.append(zn)
 write('local_set_summary.csv',summary)
 if not meta:raise RuntimeError('no usable local sets')
 mp=max(len(x) for x in Ps);mn=max(max(len(x),1) for x in Ns);P=np.zeros((len(meta),mp,3));PM=np.zeros((len(meta),mp));N=np.zeros((len(meta),mn,3));NM=np.zeros((len(meta),mn))
 for i,(p,n) in enumerate(zip(Ps,Ns)):P[i,:len(p)]=p;PM[i,:len(p)]=1
 for i,n in enumerate(Ns):
  if len(n):N[i,:len(n)]=n;NM[i,:len(n)]=1
 np.savez_compressed(H/'local_sets.npz',x=np.stack([X['x'][q[3]] for q in meta]),state_ids=np.array([q[0] for q in meta]),splits=np.array([q[1] for q in meta]),modes=np.array([q[2] for q in meta]),anchors=np.stack([(anchors[q[2]]-AFF)/SCALE for q in meta]),P=P,Pmask=PM,N=N,Nmask=NM,halfspace_A=X['halfspace_A'],halfspace_b=X['halfspace_b'])
 dump('local_set_gate.json',{'selected_pairs':len(pairs),'usable_pairs':len(meta),'train_usable':sum(q[1]=='train' for q in meta),'val_usable':sum(q[1]=='val' for q in meta),'underresolved':sum(r['status']=='LOCAL_SET_UNDERRESOLVED' for r in summary),'positive_points':len(pos),'negative_points':len(neg),'ambiguous_points':sum(r['label_status']=='ambiguous' for r in rows),'collisions':sum(int(r.get('collision',0) or 0) for r in new)})
 w=json.load(open(H/'working_state.json'));w.update(status='LOCAL_SETS_BUILT',completed=w['completed']+['local_probe_rollouts','local_set_construction'],next_action='train shared residual seeds');dump('working_state.json',w);print(json.dumps(json.load(open(H/'local_set_gate.json')),indent=2))
if __name__=='__main__':main()

