#!/usr/bin/env python3
"""Analyze B63 promotions, finalize targets, or freeze one next candidate/state."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1';Q=ROOT/'diagnostics/orthoflow3_q_learnability_v2'
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def load(root,stem):
 out=[]
 for p in sorted((root/'raw'/stem).glob('shard*.jsonl')):
  if p.stem.removeprefix('shard').isdigit():out += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--round',type=int,required=True);a=ap.parse_args()
 states=json.loads((HERE/'eligible_state_manifest.json').read_text())['selected_states'];pre=json.loads((HERE/'active_promotion_round1_plan.json').read_text());rank=pre['rankings']
 base=load(Q,'base_rollout_plan');zero=[r for r in base if r['probe_id']=='zero']+load(Q,'zero_followup_plan')+load(HERE,'zero_completion_plan')
 zr=defaultdict(dict)
 for r in zero:zr[r['state_id']][int(r['future_index'])]=r
 cr=defaultdict(lambda:defaultdict(dict))
 for r in base:
  if r['probe_id']!='zero':cr[r['state_id']][r['probe_id']][int(r['future_index'])]=r
 for rnd in range(1,a.round+1):
  stem='active_promotion_round1_plan' if rnd==1 else f'active_promotion_round{rnd}_plan'
  for r in load(HERE,stem):cr[r['state_id']][r['probe_id']][int(r['future_index'])]=r
 targets=[];promotion=[];under=[]
 for st in states:
  sid=st['state_id'];zs=sum(r['success'] for r in zr[sid].values());
  if len(zr[sid])!=64:raise RuntimeError((sid,'zero incomplete'))
  if zs>=63:targets.append({'state_id':sid,'split':st['split'],'target_kind':'ZERO','target_status':'RESOLVED_ZERO_B63','probe_id':'zero','eta1':0.,'eta2':0.,'eta3':0.,'successes':zs,'trials':64,'mean_successful_jdef':0.});continue
  confirmed=[];tested=[]
  for item in rank[sid]:
   rr=cr[sid][item['probe_id']]
   if len(rr)==64:
    succ=[r for r in rr.values() if r['success']];b=len(succ)>=63;mean=sum(r['J_def'] for r in succ)/len(succ) if succ else None
    tested.append(item['probe_id']);promotion.append({'state_id':sid,'split':st['split'],'probe_id':item['probe_id'],'eta':json.dumps(item['eta']),'successes':len(succ),'trials':64,'B63':b,'mean_successful_jdef':mean,'screen_q':item['screen_q'],'eta_cloud_index':item['eta_cloud_index']})
    if b:confirmed.append((mean,item,len(succ)))
  if len(confirmed)>=2:
   mean,item,successes=min(confirmed,key=lambda x:(x[0],x[1]['eta_cloud_index']));targets.append({'state_id':sid,'split':st['split'],'target_kind':'ACTIVE','target_status':'RESOLVED_TWO_B63','probe_id':item['probe_id'],'eta1':item['eta'][0],'eta2':item['eta'][1],'eta3':item['eta'][2],'successes':successes,'trials':64,'mean_successful_jdef':mean})
  elif any(x['probe_id'] not in tested for x in rank[sid]):
   under.append((st,confirmed,tested))
  elif len(confirmed)==1:
   mean,item,successes=confirmed[0];targets.append({'state_id':sid,'split':st['split'],'target_kind':'ACTIVE','target_status':'TARGET_SINGLE_ROBUST','probe_id':item['probe_id'],'eta1':item['eta'][0],'eta2':item['eta'][1],'eta3':item['eta'][2],'successes':successes,'trials':64,'mean_successful_jdef':mean})
  else:
   targets.append({'state_id':sid,'split':st['split'],'target_kind':'UNRESOLVED','target_status':'TARGET_UNRESOLVED','probe_id':'','eta1':'','eta2':'','eta3':'','successes':0,'trials':64,'mean_successful_jdef':''})
 if promotion:
  with open(HERE/'target_promotion_results.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(promotion[0]));w.writeheader();w.writerows(promotion)
 if not under:
  with open(HERE/'robust_lowj_targets.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(targets[0]));w.writeheader();w.writerows(targets)
  summary={sp:{'ZERO':sum(r['split']==sp and r['target_kind']=='ZERO' for r in targets),'ACTIVE':sum(r['split']==sp and r['target_status']=='RESOLVED_TWO_B63' for r in targets),'TARGET_SINGLE_ROBUST':sum(r['split']==sp and r['target_status']=='TARGET_SINGLE_ROBUST' for r in targets),'TARGET_UNRESOLVED':sum(r['split']==sp and r['target_status']=='TARGET_UNRESOLVED' for r in targets)} for sp in ('train','val','test')};summary['train_resolved_fraction']=(summary['train']['ZERO']+summary['train']['ACTIVE']+summary['train']['TARGET_SINGLE_ROBUST'])/80;summary['status']='READY_FOR_TRAINING' if summary['train_resolved_fraction']>=.9 else 'DIRECT_ETA_TARGET_DATA_UNDERRESOLVED';dump(HERE/'target_resolution_summary.json',summary);print(json.dumps(summary,indent=2));return
 next_round=a.round+1;tasks=[]
 for st,confirmed,tested in under:
  remaining=[x for x in rank[st['state_id']] if x['probe_id'] not in tested]
  if not remaining:continue
  item=remaining[0];existing=set(cr[st['state_id']][item['probe_id']])
  for fi in range(64):
   if fi not in existing:tasks.append({'task_id':f"promote{next_round}__{st['state_id']}__{item['probe_id']}__f{fi:02d}",'candidate_id':f"{st['state_id']}__{item['probe_id']}",'state_id':st['state_id'],'split':st['split'],'probe_id':item['probe_id'],'eta':item['eta'],'future_index':fi,'promotion_round':next_round})
 prior_new=len(load(HERE,'zero_completion_plan'))+sum(len(load(HERE,'active_promotion_round1_plan' if r==1 else f'active_promotion_round{r}_plan')) for r in range(1,a.round+1));projected=prior_new+len(tasks)+1280+400
 status='READY' if projected<=15000 else 'DIRECT_ETA_TARGET_BUILD_REQUIRES_APPROVAL'
 plan={'schema':f'direct_eta_active_promotion_round{next_round}_v1','orthoflow3_sha256':'51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38','future_root_seed':2026092702,'tasks':tasks,'maximum_new_continuations':len(tasks),'underresolved_states':len(under),'projected_total_with_test_and_wide':projected,'status':status}
 dump(HERE/f'active_promotion_round{next_round}_plan.json',plan);print(json.dumps({'underresolved':len(under),'confirmed0':sum(len(c)==0 for _,c,_ in under),'confirmed1':sum(len(c)==1 for _,c,_ in under),'next_tasks':len(tasks),'projected_total':projected,'status':status},indent=2))
if __name__=='__main__':main()
