#!/usr/bin/env python3
"""Freeze TEST closed-loop results and run optional read-only Q diagnostic."""
from __future__ import annotations
import csv,json,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
import flax.linen as nn
from flax import serialization
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1';Q=ROOT/'diagnostics/orthoflow3_q_learnability_v2';SCALE=np.array([.75,1.,.75])
class QMLP(nn.Module):
 @nn.compact
 def __call__(self,x):x=nn.silu(nn.Dense(256)(x));x=nn.silu(nn.Dense(256)(x));return nn.Dense(1)(x).squeeze(-1)
def load(root,stem):
 out=[]
 for p in sorted((root/'raw'/stem).glob('shard*.jsonl')):
  if p.stem.removeprefix('shard').isdigit():out += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 return out
def write_csv(path,rows,fields=None):
 fields=fields or list(rows[0]);f=path.open('w',newline='');w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.close()
def main():
 targets=list(csv.DictReader(open(HERE/'robust_lowj_targets.csv')));targets={r['state_id']:r for r in targets if r['split']=='test'};plan=json.loads((HERE/'test_prediction_plan.json').read_text());pred=load(HERE,'test_prediction_plan')
 if len(pred)!=len(plan['tasks']):raise RuntimeError(('prediction plan incomplete',len(pred),len(plan['tasks'])))
 base=load(Q,'base_rollout_plan');zero=base+load(Q,'zero_followup_plan')+load(HERE,'zero_completion_plan');prom=[]
 for p in sorted((HERE/'raw').glob('active_promotion_round*_plan')):prom+=load(HERE,p.name)
 allcand=base+prom;by_pred=defaultdict(list);by_zero=defaultdict(dict);by_cand=defaultdict(lambda:defaultdict(dict))
 for r in pred:by_pred[r['state_id']].append(r)
 for r in zero:
  if r['probe_id']=='zero':by_zero[r['state_id']][int(r['future_index'])]=r
 for r in allcand:by_cand[r['state_id']][r['probe_id']][int(r['future_index'])]=r
 states=json.loads((HERE/'eligible_state_manifest.json').read_text())['selected_states'];sm={r['state_id']:r for r in states};h=np.asarray(np.load(HERE/'conditioning_features.npz')['features']);qn=json.loads((Q/'normalization.json').read_text());hm=np.asarray(qn['h_mean']);hs=np.asarray(qn['h_std']);ec=np.asarray(qn['eta_center']);es=np.asarray(qn['eta_scale']);qs=json.loads((Q/'selected_checkpoint.json').read_text());qm=QMLP();template=qm.init(jax.random.PRNGKey(0),jnp.zeros((1,217)));qp=serialization.from_bytes(template,Path(qs['checkpoint']).read_bytes())
 def qhat(sid,eta):x=np.r_[((h[sm[sid]['feature_index']]-hm)/hs),(np.asarray(eta)-ec)/es].astype(np.float32);return float(jax.nn.sigmoid(qm.apply(qp,jnp.asarray(x)[None]))[0])
 closed=[];comp=[];zrows=[];arows=[];qrows=[]
 for sid,t in targets.items():
  pr=by_pred[sid];assert len(pr)==64
  eta=np.asarray(pr[0]['eta'],float);ps=sum(r['success'] for r in pr);pj=[r['J_def'] for r in pr if r['success']];zr=list(by_zero[sid].values());assert len(zr)==64;zs=sum(r['success'] for r in zr)
  tr=zr if t['probe_id']=='zero' else list(by_cand[sid][t['probe_id']].values());assert len(tr)==64,(sid,t['probe_id'],len(tr));ts=sum(r['success'] for r in tr)
  q=qhat(sid,eta);row={'state_id':sid,'target_kind':t['target_kind'],'target_status':t['target_status'],'pred_eta1':eta[0],'pred_eta2':eta[1],'pred_eta3':eta[2],'pred_eta_norm':float(np.linalg.norm(eta)),'pred_normalized_eta_norm':float(np.linalg.norm(eta/SCALE)),'pred_successes':ps,'pred_trials':64,'pred_Q64':ps/64,'pred_B63':ps>=63,'pred_mean_successful_J_def':float(np.mean(pj)) if pj else '','deadlocks':sum(r['outcome']=='safe_deadlock' for r in pr),'timeouts':sum(r['outcome']=='timeout' for r in pr),'collisions':sum(r['outcome']=='collision' for r in pr),'other_numerical':sum(r['outcome']=='other_numerical' for r in pr),'frozen_Qhat_readonly':q};closed.append(row)
  comp.append({'state_id':sid,'target_kind':t['target_kind'],'target_probe_id':t['probe_id'],'target_successes':ts,'target_B63':ts>=63,'prediction_successes':ps,'prediction_B63':ps>=63,'zero_successes':zs,'zero_B63':zs>=63})
  qrows.append({'state_id':sid,'target_kind':t['target_kind'],'pred_B63':ps>=63,'true_Q64':ps/64,'frozen_Qhat':q,'Qhat_below_0.5':q<.5})
  if t['target_kind']=='ZERO':zrows.append({'state_id':sid,'pred_eta_norm':float(np.linalg.norm(eta)),'visibly_nonzero':bool(np.linalg.norm(eta)>1e-6),'zero_B63':zs>=63,'pred_Q64':ps/64,'pred_B63':ps>=63,'classification':'ZERO_HARMLESS_NONZERO' if ps>=63 else 'ZERO_HARMFUL_FALSE_ACTIVATION','pred_mean_successful_J_def':float(np.mean(pj)) if pj else ''})
  else:arows.append({'state_id':sid,'pred_eta_normalized_norm':float(np.linalg.norm(eta/SCALE)),'pred_Q64':ps/64,'pred_B63':ps>=63,'target_Q64':ts/64,'classification':'ACTIVE_SUCCESS' if ps>=63 else ('ACTIVE_NEAR_ZERO_MISS' if np.linalg.norm(eta/SCALE)<=.05 else 'ACTIVE_OUTSIDE_BASIN')})
 write_csv(HERE/'heldout_64seed_closedloop.csv',closed);write_csv(HERE/'heldout_target_vs_prediction.csv',comp);write_csv(HERE/'zero_false_activation.csv',zrows,fields=['state_id','pred_eta_norm','visibly_nonzero','zero_B63','pred_Q64','pred_B63','classification','pred_mean_successful_J_def']);write_csv(HERE/'active_target_performance.csv',arows,fields=['state_id','pred_eta_normalized_norm','pred_Q64','pred_B63','target_Q64','classification']);write_csv(HERE/'frozen_q_readonly_diagnostic.csv',qrows)
 summary={'resolved_test_states':len(closed),'predicted_B63_states':sum(r['pred_B63'] for r in closed),'mean_pred_Q64':float(np.mean([r['pred_Q64'] for r in closed])),'target_B63_states':sum(r['target_B63'] for r in comp),'zero_B63_states':sum(r['zero_B63'] for r in comp),'zero_harmful_false_activation':sum(r['classification']=='ZERO_HARMFUL_FALSE_ACTIVATION' for r in zrows),'zero_harmless_nonzero':sum(r['classification']=='ZERO_HARMLESS_NONZERO' for r in zrows),'active_success':sum(r['classification']=='ACTIVE_SUCCESS' for r in arows),'active_basin_miss':sum(r['classification']!='ACTIVE_SUCCESS' for r in arows),'active_near_zero_miss':sum(r['classification']=='ACTIVE_NEAR_ZERO_MISS' for r in arows),'Qhat_mean_on_pred_success':float(np.mean([r['frozen_Qhat'] for r in qrows if r['pred_B63']])) if any(r['pred_B63'] for r in qrows) else None,'Qhat_mean_on_pred_miss':float(np.mean([r['frozen_Qhat'] for r in qrows if not r['pred_B63']])) if any(not r['pred_B63'] for r in qrows) else None,'Qhat_below_half_on_misses':sum((not r['pred_B63']) and r['Qhat_below_0.5'] for r in qrows)};(HERE/'heldout_closedloop_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
