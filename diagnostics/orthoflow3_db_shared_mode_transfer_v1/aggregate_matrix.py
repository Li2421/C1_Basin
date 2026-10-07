#!/usr/bin/env python3
import csv,json,hashlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
H=Path(__file__).parent
def write(name,rows,fields=None):
 fields=fields or list(rows[0]);
 with open(H/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def aggregate(v):
 out=Counter(str(r['outcome']) for r in v);succ=[r for r in v if r['success']];k=len(succ);n=len(v)
 return dict(successes=k,trials=n,empirical_Q=k/n,B63=(n==64 and k>=63),strong_proxy=(k>=15 if n==16 else k>=30 if n==32 else k>=63),deadlock=sum(out[q] for q in ('strict_deadlock','safe_deadlock','deadlock')),timeout=out['timeout'],collision=sum('collision' in str(r['outcome']) for r in v),J_def_mean=float(np.mean([r['J_def'] for r in v])),episode_length_mean=float(np.mean([r['episode_steps'] for r in v])))
def main():
 split=json.load(open(H/'db_state_split.json'));meta={s['state_id']:s for s in split['states']};rows=[]
 for p in (H/'raw').glob('matrix*.jsonl'):
  rows += [json.loads(x) for x in open(p) if x.strip()]
 # One exact projection solver failure is not a scientific outcome.  Its
 # entire matched future block is replaced deterministically by future 32.
 rows=[r for r in rows if r.get('scientific_outcome_valid') is True]
 rows=[r for r in rows if not (r['state_id']=='DB_MODE_val_008' and int(r['future_index'])==31)]
 key=lambda r:(r['state_id'],r['controller'],int(r['mode_id']),int(r['future_index']))
 ded={key(r):r for r in rows};rows=list(ded.values());expected={'train':16,'val':32,'test':64};by=defaultdict(list)
 for r in rows:by[(r['state_id'],r['controller'],int(r['mode_id']))].append(r)
 mode=[]
 for sid,s in meta.items():
  for m in range(12):
   v=by[(sid,'transformed',m)];n=expected[s['split']]
   if len(v)!=n:raise RuntimeError(('incomplete transformed matrix',sid,m,len(v),n))
   e=v[0]['eta'];mode.append(dict(state_id=sid,split=s['split'],source_group=s['source_group'],mode_id=m,eta1=e[0],eta2=e[1],eta3=e[2],**aggregate(v)))
 write('transformed_codebook_coverage.csv',mode)
 for sp,fn in [('train','train_mode_counts.csv'),('val','val_mode_counts.csv'),('test','test_mode_q64.csv')]:write(fn,[r for r in mode if r['split']==sp])
 raw=[]
 for sid,s in meta.items():
  if s['split']!='test':continue
  for m in range(12):
   v=by[(sid,'raw_toy',m)]
   if len(v)!=64:raise RuntimeError(('incomplete raw matrix',sid,m,len(v)))
   e=v[0]['eta'];raw.append(dict(state_id=sid,split='test',mode_id=m,eta1=e[0],eta2=e[1],eta3=e[2],**aggregate(v)))
 write('raw_toy_codebook_coverage.csv',raw)
 safety=[]
 for sid,s in meta.items():
  if s['split'] not in ('val','test'):continue
  v=by[(sid,'safety',-1)];n=expected[s['split']]
  if len(v)!=n:raise RuntimeError(('incomplete safety',sid,len(v),n))
  safety.append(dict(state_id=sid,split=s['split'],mode_id=-1,eta1=0,eta2=0,eta3=0,**aggregate(v)))
 write('safety_mode_counts.csv',safety)
 # Stable DB h feature: exact obs72 + exact frozen current Flow action8.
 ids=[];splits=[];feat=[]
 for s in split['states']:
  c=json.load(open(H/'raw'/f"conditioning_{s['state_id']}.json"));x=np.asarray(c['h_feature'],np.float64);assert x.shape==(80,);ids.append(s['state_id']);splits.append(s['split']);feat.append(x)
 feat=np.asarray(feat);train=np.array(splits)=='train';mean=feat[train].mean(0);std=feat[train].std(0);std=np.maximum(std,1e-6);x=(feat-mean)/std
 np.savez_compressed(H/'state_features.npz',state_ids=np.array(ids),splits=np.array(splits),features=feat,x=x.astype(np.float32),mean=mean,std=std)
 (H/'feature_normalization.json').write_text(json.dumps({'dimension':80,'schema':'obs72+current_raw_flow8','mean':mean.tolist(),'std':std.tolist(),'train_only':True},indent=2)+'\n')
 (H/'matrix_summary.json').write_text(json.dumps({'raw_valid_records_after_matched_repair_filter':len(rows),'deduplicated_records':len(ded),'transformed_pairs':len(mode),'raw_toy_test_pairs':len(raw),'safety_pairs':len(safety),'feature_dim':80,'numerical_repair_manifest':'numerical_repair_manifest.json'},indent=2)+'\n')
 print((H/'matrix_summary.json').read_text())
if __name__=='__main__':main()
