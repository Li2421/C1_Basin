#!/usr/bin/env python3
"""Build authoritative structured aggregates and matched historical sparse controls after postflight."""
from __future__ import annotations
import csv,hashlib,json,sqlite3
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path('/home/zhihan/research/Basin_C1');D=ROOT/'diagnostics';H=Path(__file__).parent;DB=ROOT/'shared_rollout_db/rollout.sqlite'
OLD=D/'orthoflow3_continuous_basin_critic_v1';TOY=D/'orthoflow3_shared_eta_codebook_v1'
def dump(n,x):(H/n).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def h(x):return hashlib.sha256(x.encode()).hexdigest()

def structured():
 idx=json.load(open(H/'request_index.json'));req=json.load(open(H/'planned_rollouts.json'))['requests'];assert len(idx)==len(req)
 states={x['state_id']:x for x in json.load(open(H/'toy_state_split.json'))['states']};features=np.load(TOY/'state_features.npz')['features']
 con=sqlite3.connect(DB);con.row_factory=sqlite3.Row;rows=[];missing=[]
 for meta,rq in zip(idx,req):
  q=','.join('?'*len(rq['seed_keys']))
  got=con.execute(f'''SELECT seed_key,success,deadlock,timeout,collision FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?
    AND conflict_quarantined=0 AND numerical_failure=0 AND compatibility_quality='EXACT_REUSE' AND seed_key IN ({q})''',[rq['state_uid'],rq['eta_uid'],rq['controller_uid'],*rq['seed_keys']]).fetchall()
  found={x['seed_key'] for x in got};miss=[x for x in rq['seed_keys'] if x not in found]
  if miss:missing.append({'state_id':meta['state_id'],'eta_uid':meta['eta_uid'],'missing':miss});continue
  # Preserve every compatible exact historical seed for stronger empirical evaluation.
  allr=con.execute('''SELECT success,deadlock,timeout,collision FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?
    AND conflict_quarantined=0 AND numerical_failure=0 AND compatibility_quality='EXACT_REUSE' ''',(rq['state_uid'],rq['eta_uid'],rq['controller_uid'])).fetchall()
  st=states[meta['state_id']];k=sum(x['success'] for x in allr);n=len(allr)
  rows.append({**meta,'n_requested':len(rq['seed_keys']),'n_trials':n,'n_success':k,'n_deadlock':sum(x['deadlock'] for x in allr),'n_timeout':sum(x['timeout'] for x in allr),
               'n_collision':sum(x['collision'] for x in allr),'empirical_q':k/n,'n_eff':min(n,16),'feature_index':st['dataset_index'],'h_raw':features[st['dataset_index']].astype(float).tolist()})
 con.close()
 if missing:raise RuntimeError(('postflight incomplete',len(missing),missing[:2]))
 pq.write_table(pa.Table.from_pylist(rows),H/'structured_pair_table.parquet',compression='zstd')
 return rows

def sparse_controls(struct):
 old=[r for r in pq.read_table(OLD/'pair_table.parquet').to_pylist() if r['scenario']=='Toy' and r['state_split']=='train']
 train_states={r['state_id'] for r in struct if r['matrix_partition']=='TRAIN_TRAIN'}
 # Apply the *current* spatial eta holdout to the historical sparse pool.  The
 # old critic's eta_split cannot be reused here: several current TEST probes
 # were previously training eta, which would make the matched control a false
 # unseen-eta comparison.  Radius 0.05 is the frozen spatial grouping radius.
 assignments=json.load(open(H/'toy_eta_split.json'))['assignments']
 test_z=np.asarray([[r['z1'],r['z2'],r['z3']] for r in assignments if r['split']=='test'],dtype=float)
 def zn(r):
  e=r['eta'] if 'eta' in r else [r['eta1'],r['eta2'],r['eta3']]
  return np.asarray([(e[0]-.875)/.75,e[1],(e[2]-.375)/.75],dtype=float)
 pool=[r for r in old if r['state_id'] in train_states and float(np.min(np.linalg.norm(test_z-zn(r),axis=1)))>0.05+1e-12]
 # Primary: exact pair-count/state-count/Q-bin match. Matching eta count too would force a
 # complete 48x40 matrix and mathematically destroy the sparse control condition.
 target=[r for r in struct if r['matrix_partition']=='TRAIN_TRAIN'];N=len(target);bins=np.array([0,.25,.5,.75,.9375,1.0000001])
 def bi(q):return int(np.clip(np.searchsorted(bins,q,side='right')-1,0,len(bins)-2))
 want=Counter(bi(r['empirical_q']) for r in target);by=defaultdict(list)
 for r in pool:by[bi(r['empirical_q'])].append(r)
 # The historical pool cannot match all 1,920 structured rows after enforcing
 # the current eta holdout (it lacks enough failures/intermediate outcomes).
 # Compare equal deterministic subsets with *identical* Q-bin counts instead
 # of silently filling the sparse control with robust positives.
 match_counts={b:min(want[b],len(by[b])) for b in want}
 def select_counts(source,counts,prefix):
  chosen=[];used=set();rem=dict(counts)
  # Guarantee every state while respecting the frozen bin quotas.
  for sid in sorted(train_states,key=lambda x:h(prefix+'|state|'+x)):
   cand=[r for r in source if r['state_id']==sid and (r['state_uid'],r['eta_uid']) not in used and rem.get(bi(r['empirical_q']),0)>0]
   if not cand:raise RuntimeError(('cannot retain all states under matched bins',prefix,sid,rem))
   r=min(cand,key=lambda x:(-rem[bi(x['empirical_q'])],h(prefix+'|cover|'+x['state_uid']+'|'+x['eta_uid'])))
   chosen.append(r);used.add((r['state_uid'],r['eta_uid']));rem[bi(r['empirical_q'])]-=1
  for b,n in sorted(rem.items()):
   cand=[r for r in source if bi(r['empirical_q'])==b and (r['state_uid'],r['eta_uid']) not in used]
   cand.sort(key=lambda x:h(prefix+'|bin|'+x['state_uid']+'|'+x['eta_uid']))
   if len(cand)<n:raise RuntimeError(('bin quota underflow',prefix,b,n,len(cand)))
   for r in cand[:n]:chosen.append(r);used.add((r['state_uid'],r['eta_uid']))
  assert Counter(bi(r['empirical_q']) for r in chosen)==Counter(counts)
  assert len({r['state_id'] for r in chosen})==48
  return chosen
 chosen=select_counts(pool,match_counts,'sparse')
 structured_matched=select_counts(target,match_counts,'structured')
 pq.write_table(pa.Table.from_pylist(chosen),H/'sparse_matched_control.parquet',compression='zstd')
 pq.write_table(pa.Table.from_pylist(structured_matched),H/'structured_matched_control_basis.parquet',compression='zstd')
 # Secondary eta-count matched control: 40 historical eta with maximal state coverage.
 deg=Counter(r['eta_uid'] for r in pool);eta40={e for e,_ in sorted(deg.items(),key=lambda x:(-x[1],h('eta40|'+x[0])))[:40]};eta_ctrl=[r for r in pool if r['eta_uid'] in eta40]
 eta_match_counts=Counter(bi(r['empirical_q']) for r in eta_ctrl)
 structured_eta_matched=select_counts(target,eta_match_counts,'structured-eta-count')
 pq.write_table(pa.Table.from_pylist(eta_ctrl),H/'sparse_eta_matched_control.parquet',compression='zstd')
 pq.write_table(pa.Table.from_pylist(structured_eta_matched),H/'structured_eta_matched_control_basis.parquet',compression='zstd')
 def info(rr):
  uniq={r['eta_uid']:r for r in rr};nearest=[float(np.min(np.linalg.norm(test_z-zn(r),axis=1))) for r in uniq.values()]
  return {'pairs':len(rr),'states':len({r['state_id'] for r in rr}),'eta':len(uniq),'eta_degree_median':float(np.median(list(Counter(r['eta_uid'] for r in rr).values()))),
          'exact_current_test_eta_overlap':len(set(uniq)&{r['eta_uid'] for r in assignments if r['split']=='test'}),'min_train_to_current_test_eta_distance':min(nearest) if nearest else None,
          'q_bin_counts':dict(Counter(bi(r['empirical_q']) for r in rr)),'q_mean':float(np.mean([r['empirical_q'] for r in rr]))}
 dump('sparse_control_design.json',{'structured_full':info(target),'structured_matched_basis':info(structured_matched),'sparse_pair_matched':info(chosen),'structured_eta_matched_basis':info(structured_eta_matched),'eta_matched_secondary':info(eta_ctrl),
  'matched_q_bin_counts':match_counts,
  'eta_count_matched_q_bin_counts':eta_match_counts,
  'constraint':'A 1,920-pair Q-distribution match is impossible after applying the current eta holdout because the historical sparse pool lacks enough failure/intermediate pairs. Control 1 uses equal maximum-overlap subsets with identical pair count, 48 states, and Q bins but lets sparse eta diversity differ. Control 2 additionally matches eta count (40), pair count, 48 states, and Q bins, though exact eta coordinates necessarily differ because historical sparse observations do not cover the new fixed panel.'})

def main():
 rows=structured();sparse_controls(rows)
 print(json.dumps({'structured_pairs':len(rows),'partitions':dict(Counter(r['matrix_partition'] for r in rows)),'sparse':json.load(open(H/'sparse_control_design.json'))},indent=2))
if __name__=='__main__':main()
