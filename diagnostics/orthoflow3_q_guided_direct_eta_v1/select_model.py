#!/usr/bin/env python3
"""Analyze matched VAL stages, promote top three, and freeze final actor."""
from __future__ import annotations
import argparse,csv,json,hashlib
from collections import defaultdict
from pathlib import Path
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_q_guided_direct_eta_v1';BASE=ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def load(stem):
 out=[]
 for p in sorted((HERE/'raw'/stem).glob('shard*.jsonl')):
  if p.stem.removeprefix('shard').isdigit():out += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 return out
def write(path,rows):
 f=path.open('w',newline='');w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.close()
def summarize(rows,runs,trials):
 target={r['state_id']:r for r in csv.DictReader(open(BASE/'robust_lowj_targets.csv')) if r['split']=='val'};by=defaultdict(lambda:defaultdict(list))
 for r in rows:by[r['actor_id']][r['state_id']].append(r)
 out=[]
 for run in runs:
  aid=run['actor_id'];states=by[aid]
  if len(states)!=20 or any(len(v)!=trials for v in states.values()):raise RuntimeError((aid,len(states),[len(v) for v in states.values()]))
  succ={sid:sum(x['success'] for x in rr) for sid,rr in states.items()};z=[sid for sid in succ if target[sid]['target_kind']=='ZERO'];a=[sid for sid in succ if target[sid]['target_kind']=='ACTIVE'];oc=[x['outcome'] for rr in states.values() for x in rr]
  out.append({'actor_id':aid,'lambda_Q':float(run['lambda_Q']),'seed':int(run['seed']),'trials_per_state':trials,'total_successes':sum(succ.values()),'total_trials':20*trials,'states_all_success':sum(v==trials for v in succ.values()),'states_ge31of32':sum(v>=31 for v in succ.values()) if trials==32 else '','zero_total_successes':sum(succ[s] for s in z),'zero_trials':len(z)*trials,'zero_harmful_states':sum(succ[s]<(31 if trials==32 else trials) for s in z),'active_total_successes':sum(succ[s] for s in a),'active_trials':len(a)*trials,'deadlock':sum(x=='safe_deadlock' for x in oc),'timeout':sum(x=='timeout' for x in oc),'collision':sum(x=='collision' for x in oc),'other_numerical':sum(x=='other_numerical' for x in oc),'val_eta_mse':float(run['val_eta_mse'])})
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',type=int,choices=(1,2),required=True);a=ap.parse_args();runs=list(csv.DictReader(open(HERE/'all_training_runs.csv')))
 if a.stage==1:
  rows=load('validation_stage1_plan');plan=json.loads((HERE/'validation_stage1_plan.json').read_text());
  if len(rows)!=len(plan['tasks']):raise RuntimeError(('stage1 incomplete',len(rows),len(plan['tasks'])))
  summary=summarize(rows,runs,8);ranked=sorted(summary,key=lambda r:(-r['total_successes'],-r['states_all_success'],r['zero_harmful_states'],r['val_eta_mse'],r['lambda_Q'],r['seed']));promoted=ranked[:3];write(HERE/'validation_stage1.csv',summary)
  # Reuse exact actor eta from Stage-1 plan and add future indices 8..31.
  lookup={}
  for t in plan['tasks']:lookup.setdefault((t['actor_id'],t['state_id']),t)
  tasks=[]
  for pr in promoted:
   for (aid,sid),t in lookup.items():
    if aid!=pr['actor_id']:continue
    for fi in range(8,32):tasks.append({**{k:t[k] for k in ('candidate_id','state_id','split','probe_id','actor_id','lambda_Q','seed','eta')},'task_id':f"val32add__{aid}__{sid}__f{fi:02d}",'future_index':fi})
  dump(HERE/'validation_stage2_plan.json',{'schema':'q_guided_actor_val_stage2_v1','orthoflow3_sha256':'51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38','future_root_seed':2026092715,'promoted':promoted,'tasks':tasks,'matched_added_future_indices':list(range(8,32))});print(json.dumps({'promoted':promoted,'tasks':len(tasks)},indent=2));return
 # Stage 2.
 s1=load('validation_stage1_plan');s2=load('validation_stage2_plan');p2=json.loads((HERE/'validation_stage2_plan.json').read_text());
 if len(s2)!=len(p2['tasks']):raise RuntimeError(('stage2 incomplete',len(s2),len(p2['tasks'])))
 ids={x['actor_id'] for x in p2['promoted']};rr=[r for r in runs if r['actor_id'] in ids];summary=summarize([r for r in s1 if r['actor_id'] in ids]+s2,rr,32);ranked=sorted(summary,key=lambda r:(-r['total_successes'],-r['states_ge31of32'],r['zero_harmful_states'],-r['active_total_successes'],r['val_eta_mse'],r['lambda_Q'],r['seed']));chosen=ranked[0];write(HERE/'validation_stage2.csv',summary);run=next(r for r in runs if r['actor_id']==chosen['actor_id']);selection={'selection_frozen_before_test':True,'selected_actor_id':chosen['actor_id'],'lambda_Q':float(run['lambda_Q']),'seed':int(run['seed']),'checkpoint':run['checkpoint'],'checkpoint_sha256':run['checkpoint_sha256'],'lexicographic_rule':['total successes /640','states >=31/32','ZERO states below31/32','ACTIVE total successes','VAL eta MSE','lower lambda','lower seed'],'validation_stage2_selected_row':chosen};dump(HERE/'model_selection.json',{'all_promoted':ranked,'selection':selection});dump(HERE/'selected_checkpoint.json',selection);(HERE/'selected_checkpoint_sha256.txt').write_text(run['checkpoint_sha256']+'\n')
 # Freeze TEST eta predictions without touching outcomes.
 import sys;sys.path.insert(0,str(ROOT));import jax,jax.numpy as jnp,numpy as np;from flax import serialization
 from train_all import G
 states=json.loads((HERE/'eligible_state_manifest.json').read_text())['selected_states'];test=[x for x in states if x['split']=='test'];features=np.asarray(np.load(HERE/'conditioning_features.npz')['features']);norm=json.loads((BASE/'normalization.json').read_text());hm=np.asarray(norm['h_mean']);hs=np.asarray(norm['h_std']);x=np.stack([(features[s['feature_index']]-hm)/hs for s in test]).astype(np.float32);m=G();tmp=m.init(jax.random.PRNGKey(0),jnp.zeros((1,214)));pp=serialization.from_bytes(tmp,Path(run['checkpoint']).read_bytes());etas=np.asarray(m.apply(pp,jnp.asarray(x)));tasks=[]
 for st,eta in zip(test,etas):
  for fi in range(64):tasks.append({'task_id':f"guided64__{st['state_id']}__f{fi:02d}",'candidate_id':f"{st['state_id']}__guided",'state_id':st['state_id'],'split':'test','probe_id':'q_guided','actor_id':chosen['actor_id'],'eta':eta.tolist(),'future_index':fi})
 dump(HERE/'heldout_test_plan.json',{'schema':'q_guided_actor_test64_v1','orthoflow3_sha256':'51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38','future_root_seed':2026092702,'checkpoint_sha256':run['checkpoint_sha256'],'tasks':tasks});print(json.dumps({'selected':selection,'test_tasks':len(tasks)},indent=2))
if __name__=='__main__':main()
