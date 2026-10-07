"""Distinguish exact eta lookup from state-conditioned learning in local test."""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0,'/home/zhihan/research/Basin_C1')
from shared_rollout_db.src.rollout_db import eta_identity

ROOT=Path('/home/zhihan/research/Basin_C1')
OUT=Path(__file__).resolve().parent
PRIOR=ROOT/'diagnostics/orthoflow3_toy_critic_local_state_probe_v1'
STRUCT=ROOT/'diagnostics/orthoflow3_structured_continuous_q_data_v1'
NORM=ROOT/'diagnostics/orthoflow3_continuous_basin_critic_v1/dataset_manifest.json'
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def csvwrite(p,r):
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def main():
 old=read(PRIOR/'frozen_proposals.json')
 samples={}
 for ref in old['references']:
  ep=ref['episode_index']
  for kind,eta in ref['eta'].items():samples[(ep,kind)]={'eta':eta,'uid':eta_identity(eta)[0],'q':[ref['original_bad_q16' if kind=='frozen_bad' else 'original_good_q16']]}
 for row in csv.DictReader((PRIOR/'per_state_eta_results.csv').open()):
  if float(row['offset_m'])==0:continue
  samples[(int(row['parent_episode_index']),row['kind'])]['q'].append(float(row['Q16']))
 assert len(samples)==10 and all(len(v['q'])==3 for v in samples.values())
 normal=read(NORM)['eta_normalization'];ec=np.asarray(normal['center']);es=np.asarray(normal['scale'])
 wide=pq.read_table(STRUCT/'sparse_matched_control.parquet').to_pylist()
 struct=[r for r in pq.read_table(STRUCT/'structured_pair_table.parquet').to_pylist() if r['matrix_partition']=='TRAIN_TRAIN']
 sk={(r['state_uid'],r['eta_uid']) for r in struct}
 wide=[r for r in wide if (r['state_uid'],r['eta_uid']) not in sk]
 assert len(struct)==1920 and len(wide)==956
 oldpairs=[{'eta_uid':r['eta_uid'],'eta':[float(x) for x in r['eta']],
            'q':float(r['empirical_q']),'source':'structured'} for r in struct]
 oldpairs +=[{'eta_uid':r['eta_uid'],'eta':[float(r[k]) for k in ('eta1','eta2','eta3')],
              'q':float(r['empirical_q']),'source':'wide'} for r in wide]
 old_uids={r['eta_uid'] for r in oldpairs}
 z=np.asarray([(np.asarray(r['eta'])-ec)/es for r in oldpairs])
 audit=[]
 for (ep,kind),v in sorted(samples.items()):
  target=(np.asarray(v['eta'])-ec)/es
  dist=np.linalg.norm(z-target,axis=1)
  ix=np.argsort(dist)[:20]
  audit.append({'parent_episode_index':ep,'kind':kind,'eta_uid':v['uid'],
    'exact_eta_in_original_train':int(v['uid'] in old_uids),
    'nearest_normalized_eta_distance':float(dist[ix[0]]),
    'nearest5_normalized_eta_distance_max':float(dist[ix[4]]),
    'nearest5_train_Q_mean':float(np.mean([oldpairs[i]['q'] for i in ix[:5]])),
    'nearest20_train_Q_mean':float(np.mean([oldpairs[i]['q'] for i in ix])),
    'local_train_Q_mean':float(np.mean(v['q']))})
 testpred=list(csv.DictReader((OUT/'test_pair_predictions.csv').open()))
 target=[r for r in testpred if r['variant']=='frozen_pretrained']
 assert len(target)==20
 eta_only_correct=0;eta_only_selected_q=[]
 eta_only_pred=[];eta_only_true=[]
 for i in range(0,20,2):
  a,b=target[i:i+2]
  assert a['parent_episode_index']==b['parent_episode_index'] and a['offset_m']==b['offset_m']
  ep=int(a['parent_episode_index']);pa=float(np.mean(samples[(ep,a['kind'])]['q']));pb=float(np.mean(samples[(ep,b['kind'])]['q']))
  eta_only_pred.extend([pa,pb]);eta_only_true.extend([float(a['Q16']),float(b['Q16'])])
  pick=a if pa>=pb else b
  eta_only_correct+=float(pick['Q16'])>=15/16
  eta_only_selected_q.append(float(pick['Q16']))
 csvwrite(OUT/'eta_support_audit.csv',audit)
 result={'target_exact_eta_count':len({v['uid'] for v in samples.values()}),
   'target_exact_eta_present_in_original_TRAIN':sum(r['exact_eta_in_original_train'] for r in audit),
   'eta_only_train_local_prior_TEST_B15':int(eta_only_correct),
   'eta_only_train_local_prior_TEST_mean_Q16':float(np.mean(eta_only_selected_q)),
   'eta_only_train_local_prior_TEST_MAE':float(np.mean(abs(np.asarray(eta_only_pred)-np.asarray(eta_only_true)))),
   'eta_only_train_local_prior_TEST_soft_label_NLL':float(np.mean(-np.asarray(eta_only_true)*np.log(np.clip(eta_only_pred,1e-7,1-1e-7))-(1-np.asarray(eta_only_true))*np.log(np.clip(1-np.asarray(eta_only_pred),1e-7,1-1e-7)))),
   'nearest_original_TRAIN_eta_distance_min':min(r['nearest_normalized_eta_distance'] for r in audit),
   'nearest_original_TRAIN_eta_distance_median':float(np.median([r['nearest_normalized_eta_distance'] for r in audit])),
   'nearest_original_TRAIN_eta_distance_max':max(r['nearest_normalized_eta_distance'] for r in audit),
   'interpretation':'Same exact eta were used in local TRAIN and held-out physical perturbations. Eta-only local prior solves top-1 in this targeted two-eta task; compare its TEST MAE/NLL against the neural local model to assess state-conditioned Q variation, especially episode 115.'}
 write(OUT/'support_control.json',result)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
