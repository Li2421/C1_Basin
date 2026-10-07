#!/usr/bin/env python3
"""Post-hoc, rollout-free diagnosis of continuous critic failure modes."""
from __future__ import annotations
import csv,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

H=Path(__file__).parent
rows=pq.read_table(H/'pair_table.parquet').to_pylist()
manifest=json.load(open(H/'dataset_manifest.json'));fs=np.load(H/'feature_store.npz')

def qbin(q):
 return 'failure<=.5' if q<=.5 else ('intermediate_.5-.9' if q<.9 else ('near_.9-.9375' if q<15/16 else 'robust>=.9375'))
def summarize(vals):
 a=np.asarray(vals,float);return {'n':len(a),'min':float(a.min()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'mean':float(a.mean()) if len(a) else None,'max':float(a.max()) if len(a) else None}
def write(name,data):
 with open(H/name,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]) if data else ['empty']);w.writeheader();w.writerows(data)

out={}; degree_rows=[]; group_rows=[]
for sc in ('Toy','DB'):
 tr=[r for r in rows if r['scenario']==sc and r['sampled_train']]
 d=[r for r in rows if r['scenario']==sc and r['regime']=='D_unseen_state_unseen_eta']
 by_eta=defaultdict(list);by_state=defaultdict(list)
 for r in tr:by_eta[r['eta_uid']].append(r);by_state[r['state_uid']].append(r)
 eta_deg=[len({r['state_uid'] for r in v}) for v in by_eta.values()]
 pair_deg=[len({q['state_uid'] for q in by_eta[r['eta_uid']]}) for r in tr]
 variable=[]
 for uid,v in by_eta.items():
  qs=[r['empirical_q'] for r in v]
  variable.append({'scenario':sc,'eta_uid':uid,'states':len({r['state_uid'] for r in v}),'pairs':len(v),'q_mean':np.mean(qs),'q_std':np.std(qs),'q_min':min(qs),'q_max':max(qs),'crosses_failure_robust':min(qs)<=.5 and max(qs)>=15/16})
 degree_rows+=variable
 trials=Counter(('1' if r['n_trials']==1 else '2-7' if r['n_trials']<8 else '8-15' if r['n_trials']<16 else '>=16') for r in tr)
 train_bins=Counter(qbin(r['empirical_q']) for r in tr);d_bins=Counter(qbin(r['empirical_q']) for r in d)
 out[sc]={
  'independent_train_states':len(by_state),'train_pairs':len(tr),'D_pairs':len(d),
  'pairs_per_state':summarize([len(v) for v in by_state.values()]),'eta_state_degree':summarize(eta_deg),
  'fraction_train_pairs_eta_seen_one_state':float(np.mean(np.asarray(pair_deg)==1)),
  'fraction_train_pairs_eta_seen_ge5_states':float(np.mean(np.asarray(pair_deg)>=5)),
  'unique_train_eta':len(by_eta),'eta_with_ge5_states':sum(x>=5 for x in eta_deg),'eta_with_ge20_states':sum(x>=20 for x in eta_deg),
  'eta_with_state_dependent_extremes':sum(r['crosses_failure_robust'] for r in variable),
  'trial_bins':dict(trials),'train_q_bins':dict(train_bins),'D_q_bins':dict(d_bins),
  'train_q_mean':float(np.mean([r['empirical_q'] for r in tr])),'D_q_mean':float(np.mean([r['empirical_q'] for r in d]))}
 # h geometry
 key='toy' if sc=='Toy' else 'db'; raw=fs[key].astype(float); norm=manifest['state_normalization'][sc]
 x=(raw-np.asarray(norm['mean']))/np.asarray(norm['std'])
 states=json.load(open(H/'state_split.json'))['states']; sm={r['feature_index']:r['state_split'] for r in states if r['scenario']==sc}
 ixtr=np.asarray([i for i in range(len(x)) if sm.get(i)=='train']);ixte=np.asarray([i for i in range(len(x)) if sm.get(i)=='test'])
 # RMS-normalized Euclidean distance; robust to feature dimension.
 dist=cKDTree(x[ixtr]/np.sqrt(x.shape[1])).query(x[ixte]/np.sqrt(x.shape[1]))[0]
 xc=x[ixtr]-x[ixtr].mean(0);sv=np.linalg.svd(xc,compute_uv=False);energy=np.cumsum(sv**2)/np.sum(sv**2)
 out[sc]['h_feature']={'dimension':x.shape[1],'variable_dimensions':int(np.sum(np.std(raw[ixtr],0)>1e-8)),'matrix_rank':int(np.linalg.matrix_rank(xc)),
  'pca_dims_95pct':int(np.searchsorted(energy,.95)+1),'test_nearest_train_rms_distance':summarize(dist)}
 # D eta-group composition
 bg=defaultdict(list)
 for r in d:bg[r['eta_group']].append(r)
 for g,v in bg.items():
  group_rows.append({'scenario':sc,'eta_group':g,'pairs':len(v),'states':len({r['state_uid'] for r in v}),'unique_eta':len({r['eta_uid'] for r in v}),
   'q_mean':np.mean([r['empirical_q'] for r in v]),'q_std':np.std([r['empirical_q'] for r in v]),'failure_pairs':sum(r['empirical_q']<=.5 for r in v),'robust_pairs':sum(r['empirical_q']>=15/16 for r in v)})

write('diagnostic_eta_degrees.csv',degree_rows);write('diagnostic_D_eta_groups.csv',group_rows)
training=list(csv.DictReader(open(H/'training_summary.csv')))
out['optimization']={m:{'best_steps':[int(r['best_step']) for r in training if r['model']==m],'total_steps':[int(r['steps']) for r in training if r['model']==m],'val_nll':[float(r['val_nll']) for r in training if r['model']==m]} for m in sorted({r['model'] for r in training})}
json.dump(out,open(H/'failure_diagnosis.json','w'),indent=2,sort_keys=True)
print(json.dumps(out,indent=2))
