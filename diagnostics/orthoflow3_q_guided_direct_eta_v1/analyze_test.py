#!/usr/bin/env python3
"""Freeze held-out TEST comparison, Q exploitation, and actor output shift."""
from __future__ import annotations
import csv,json,hashlib
from collections import Counter,defaultdict
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_q_guided_direct_eta_v1';BASE=ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1';QDIR=ROOT/'diagnostics/orthoflow3_q_learnability_v2';SCALE=np.array([.75,1.,.75])
class Q(nn.Module):
 @nn.compact
 def __call__(self,x):x=nn.silu(nn.Dense(256)(x));x=nn.silu(nn.Dense(256)(x));return nn.Dense(1)(x).squeeze(-1)
def load(root,stem):
 out=[]
 for p in sorted((root/'raw'/stem).glob('shard*.jsonl')):
  if p.stem.removeprefix('shard').isdigit():out += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 return out
def write(path,rows,fields=None):
 fields=fields or list(rows[0]);f=path.open('w',newline='');w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.close()
def main():
 plan=json.loads((HERE/'heldout_test_plan.json').read_text());guided=load(HERE,'heldout_test_plan')
 if len(guided)!=len(plan['tasks']):raise RuntimeError(('test incomplete',len(guided),len(plan['tasks'])))
 states=json.loads((HERE/'eligible_state_manifest.json').read_text())['selected_states'];sm={r['state_id']:r for r in states};test=[r for r in states if r['split']=='test'];targets={r['state_id']:r for r in csv.DictReader(open(BASE/'robust_lowj_targets.csv')) if r['split']=='test'};base={r['state_id']:r for r in csv.DictReader(open(BASE/'heldout_64seed_closedloop.csv'))};comp={r['state_id']:r for r in csv.DictReader(open(BASE/'heldout_target_vs_prediction.csv'))}
 qbase=load(QDIR,'base_rollout_plan');zraw=[r for r in qbase if r['probe_id']=='zero']+load(QDIR,'zero_followup_plan')+load(BASE,'zero_completion_plan');zby=defaultdict(dict)
 for r in zraw:zby[r['state_id']][int(r['future_index'])]=r
 braw=load(BASE,'test_prediction_plan');bby=defaultdict(list)
 for r in braw:bby[r['state_id']].append(r)
 by=defaultdict(list)
 for r in guided:by[r['state_id']].append(r)
 # Frozen Q and support cloud.
 qn=json.loads((QDIR/'normalization.json').read_text());hm=np.asarray(qn['h_mean']);hs=np.asarray(qn['h_std']);ec=np.asarray(qn['eta_center']);es=np.asarray(qn['eta_scale']);features=np.asarray(np.load(HERE/'conditioning_features.npz')['features']);qsel=json.loads((QDIR/'selected_checkpoint.json').read_text());qm=Q();tmp=qm.init(jax.random.PRNGKey(0),jnp.zeros((1,217)));qp=serialization.from_bytes(tmp,Path(qsel['checkpoint']).read_bytes());tau=float(json.loads((QDIR/'frozen_q_manifest.json').read_text())['zero_feasibility_threshold'])
 cloud=[]
 for r in csv.DictReader(open(QDIR/'eta_probe_cloud.csv')):cloud.append((r['probe_id'],np.array([r['eta1'],r['eta2'],r['eta3']],float)))
 train_base=load(QDIR,'base_rollout_plan');train_q=defaultdict(list)
 for r in train_base:
  if r['split']=='train':train_q[r['probe_id']].append(int(r['success']))
 def qhat(sid,eta):x=np.r_[((features[sm[sid]['feature_index']]-hm)/hs),(np.asarray(eta)-ec)/es].astype(np.float32);return float(jax.nn.sigmoid(qm.apply(qp,jnp.asarray(x)[None]))[0])
 def support(eta):
  ds=[np.linalg.norm((np.asarray(eta)-e)/es) for _,e in cloud];i=int(np.argmin(ds));pid=cloud[i][0];return pid,float(ds[i]),float(np.mean(train_q[pid]))
 results=[];pair=[];zero=[];active=[];exploit=[];shift=[]
 for st in test:
  sid=st['state_id'];rr=by[sid];assert len(rr)==64;eta=np.asarray(rr[0]['eta']);succ=sum(r['success'] for r in rr);sj=[r['J_def'] for r in rr if r['success']];out=[r['outcome'] for r in rr];b=base[sid];be=np.array([b['pred_eta1'],b['pred_eta2'],b['pred_eta3']],float);pid,dist,nq=support(eta);q=qhat(sid,eta);bq=qhat(sid,be);isexp=q>=tau and succ<63
  common={'state_id':sid,'target_kind':targets[sid]['target_kind']};zr=list(zby[sid].values());br=bby[sid];assert len(zr)==len(br)==64;zout=[r['outcome'] for r in zr];bsj=[r['J_def'] for r in br if r['success']]
  results += [{**common,'controller':'zero','successes':int(comp[sid]['zero_successes']),'trials':64,'Q64':int(comp[sid]['zero_successes'])/64,'B63':comp[sid]['zero_B63']=='True','deadlock':sum(x=='safe_deadlock' for x in zout),'timeout':sum(x=='timeout' for x in zout),'collision':sum(x=='collision' for x in zout),'mean_successful_J_def':0.,'median_successful_J_def':0.},{**common,'controller':'baseline','successes':int(b['pred_successes']),'trials':64,'Q64':float(b['pred_Q64']),'B63':b['pred_B63']=='True','deadlock':int(b['deadlocks']),'timeout':int(b['timeouts']),'collision':int(b['collisions']),'mean_successful_J_def':b['pred_mean_successful_J_def'],'median_successful_J_def':float(np.median(bsj)) if bsj else ''},{**common,'controller':'guided','successes':succ,'trials':64,'Q64':succ/64,'B63':succ>=63,'deadlock':sum(x=='safe_deadlock' for x in out),'timeout':sum(x=='timeout' for x in out),'collision':sum(x=='collision' for x in out),'mean_successful_J_def':float(np.mean(sj)) if sj else '','median_successful_J_def':float(np.median(sj)) if sj else ''}]
  bb=b['pred_B63']=='True';gb=succ>=63;pair.append({**common,'baseline_B63':bb,'guided_B63':gb,'transition':'BOTH_B63' if bb and gb else 'BASELINE_B63_GUIDED_FAIL' if bb else 'BASELINE_FAIL_GUIDED_B63' if gb else 'BOTH_FAIL','baseline_Q64':float(b['pred_Q64']),'guided_Q64':succ/64})
  teta=np.array([targets[sid]['eta1'],targets[sid]['eta2'],targets[sid]['eta3']],float);rec={**common,'baseline_eta':json.dumps(be.tolist()),'guided_eta':json.dumps(eta.tolist()),'baseline_eta_norm':float(np.linalg.norm(be)),'guided_eta_norm':float(np.linalg.norm(eta)),'baseline_Qhat':bq,'guided_Qhat':q,'baseline_Q64':float(b['pred_Q64']),'guided_Q64':succ/64,'baseline_B63':bb,'guided_B63':gb,'baseline_target_distance':float(np.linalg.norm((be-teta)/SCALE)),'guided_target_distance':float(np.linalg.norm((eta-teta)/SCALE)),'guided_failure_category':max(Counter(out),key=Counter(out).get) if succ<63 else 'success'}
  if targets[sid]['target_kind']=='ZERO':
   nnorm=np.linalg.norm(eta/SCALE);rec['guided_zero_class']='ZERO_EXACT_OR_NEAR_ZERO' if nnorm<=.05 and gb else 'ZERO_HARMLESS_NONZERO' if gb else 'ZERO_HARMFUL_FALSE_ACTIVATION';zero.append(rec)
  else:active.append(rec)
  exploit.append({**common,'Qhat':q,'tau':tau,'true_Q64':succ/64,'true_B63':gb,'critic_exploitation_candidate':isexp,'eta1':eta[0],'eta2':eta[1],'eta3':eta[2],'eta_norm':float(np.linalg.norm(eta)),'clipping':False,'nearest_cloud_probe':pid,'nearest_cloud_normalized_distance':dist,'nearest_train_candidate_empirical_Q':nq,'overestimate_Qhat_minus_true':q-succ/64})
  shift.append({**common,'baseline_eta1':be[0],'baseline_eta2':be[1],'baseline_eta3':be[2],'baseline_norm':float(np.linalg.norm(be)),'guided_eta1':eta[0],'guided_eta2':eta[1],'guided_eta3':eta[2],'guided_norm':float(np.linalg.norm(eta)),'baseline_Qhat':bq,'guided_Qhat':q,'guided_nearest_cloud_distance':dist,'guided_clipping':False})
 write(HERE/'heldout_test_results.csv',results);write(HERE/'heldout_pairwise_baseline_vs_guided.csv',pair);write(HERE/'heldout_zero_audit.csv',zero);write(HERE/'heldout_active_audit.csv',active);write(HERE/'critic_exploitation_audit.csv',exploit);write(HERE/'actor_output_shift.csv',shift)
 summary={'baseline_B63':sum(r['baseline_B63'] for r in pair),'guided_B63':sum(r['guided_B63'] for r in pair),'baseline_mean_Q64':float(np.mean([r['baseline_Q64'] for r in pair])),'guided_mean_Q64':float(np.mean([r['guided_Q64'] for r in pair])),'transitions':dict(__import__('collections').Counter(r['transition'] for r in pair)),'baseline_zero_harmful':sum((not r['baseline_B63']) for r in zero),'guided_zero_harmful':sum(r['guided_zero_class']=='ZERO_HARMFUL_FALSE_ACTIVATION' for r in zero),'guided_zero_harmless_nonzero':sum(r['guided_zero_class']=='ZERO_HARMLESS_NONZERO' for r in zero),'guided_zero_near':sum(r['guided_zero_class']=='ZERO_EXACT_OR_NEAR_ZERO' for r in zero),'baseline_active_B63':sum(r['baseline_B63'] for r in active),'guided_active_B63':sum(r['guided_B63'] for r in active),'critic_exploitation_cases':sum(r['critic_exploitation_candidate'] for r in exploit),'max_Q_overestimate':max(r['overestimate_Qhat_minus_true'] for r in exploit)};(HERE/'heldout_test_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
