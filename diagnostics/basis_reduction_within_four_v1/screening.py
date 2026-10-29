"""Durable fixed-menu Oracle screening, reduction, and conditional expansion."""
from dispatch import *
from evidence import run_requests,legacy_case
from combinations import lower
from budget import used
import copy,traceback
from execution_control import PriorityPause

def selected_entries(all_scenes=False):
 entries=read(HERE/'SCENES.json')['entries']
 if all_scenes:return entries
 names={x['scene']for x in read(HERE/'REPRESENTATIVE17_FROZEN.json')['selected']}
 return[e for e in entries if e['scene']['name']in names]

def ident(e,ctl,si):return [e['type_id'],e['level'],ctl['type_code'],si]

def historical_four():
 path=HERE/'SEARCH_four_8_screen.json'
 if path.exists()and(HERE/'HISTORICAL_FOUR_PREFLIGHT_CERT.json').exists():return read(path)
 entries=selected_entries();allowed={(e['type_id'],e['level'])for e in entries};old=read(HERE/'inputs/v3/SEARCH_STATE.json');groups=[]
 for source in old['groups']:
  if source['family']!=4 or(source['type_id'],source['level'])not in allowed:continue
  key=[source['type_id'],source['level'],source['controller'],source['state_index']];queries=[]
  for q in source['queries']:
   uid=q['rollout_uid'];raw=ROOT/'diagnostics/scene_types_oracle_coverage_v3/raw'/f'{uid}.json';r=read(raw)if raw.exists()else q
   queries.append(dict(index=q['index'],query_uid=q.get('target_query_uid',uid),source_rollout_uid=uid,result=r,historical=True))
  g=dict(identity=key,scene=source['scene'],success=source['success'],searched=source['searched'],first_success=source['first_success'],queries=queries,gap=None)
  if g['success']:g['solution']=queries[source['first_success']-1]
  groups.append(g)
 assert len(groups)==272 and all(g['success']or g['searched']==64 for g in groups)
 state=dict(code='four_8',scope='screen',groups=groups,pending=None,completed_stages=[],historical=True,complete64=True)
 dump(path,state)
 # Official preflight records the existing identities; no old execution is
 # submitted and no new source identity is claimed for these historical labels.
 requests=[];sources={q['source_rollout_uid']for g in groups for q in g['queries']}
 with db.connect(True)as con:
  for uid in sorted(sources):
   row=con.execute('SELECT state_uid,eta_uid,controller_uid,seed_key,conflict_quarantined FROM rollout WHERE rollout_uid=?',(uid,)).fetchone()
   assert row and not row['conflict_quarantined'],('Historical source not in official DB',uid)
   requests.append({k:row[k]for k in('state_uid','eta_uid','controller_uid','seed_key')})
 planned=HERE/'historical_four_8_all64_planned.json';dump(planned,dict(requests=requests,source_rollout_uids=sorted(sources),note='Read-only reanalysis of archived v3 labels, including previously certified v1 horizon-transfer successes. Source and v3 target identities remain distinct.'))
 preflight=planner.preflight(planned);dump(HERE/'historical_four_8_all64_preflight.json',preflight)
 assert preflight['summary']['ambiguous']==0
 m=read(HERE/'REUSE_MANIFEST.json')if(HERE/'REUSE_MANIFEST.json').exists()else dict(records={})
 for g in groups:
  for q in g['queries']:m['records'][q['query_uid']]=dict(target_query_uid=q['query_uid'],source_rollout_uid=q['source_rollout_uid'],kind='archived_v3_reference_with_original_provenance',stage='historical_four_8_all64')
 dump(HERE/'REUSE_MANIFEST.json',m);dump(HERE/'HISTORICAL_FOUR_PREFLIGHT_CERT.json',dict(passed=True,sources=len(sources),queries=sum(len(g['queries'])for g in groups),exact_source_preflight=preflight['summary'],v3_historical_aliases=sum(q['query_uid']!=q['source_rollout_uid']for g in groups for q in g['queries'])));return state

def search(combo,cap,make,scope='screen',floor=80000):
 code=combo['code'];path=HERE/f'SEARCH_{code}_{scope}.json';entries=selected_entries(scope=='all');controls=controllers()
 if path.exists():state=read(path)
 else:
  inherited={}
  if scope=='all'and(HERE/f'SEARCH_{code}_screen.json').exists():inherited={tuple(g['identity']):g for g in read(HERE/f'SEARCH_{code}_screen.json')['groups']}
  groups=[]
  for e in entries:
   for si in range(8):
    for ctl in controls:
     key=ident(e,ctl,si)
     groups.append(copy.deepcopy(inherited[tuple(key)])if tuple(key)in inherited else dict(identity=key,scene=e['scene']['name'],success=False,searched=0,first_success=None,queries=[],gap=None))
  state=dict(code=code,scope=scope,groups=groups,pending=None,completed_stages=[],complete64=False);dump(path,state)
 gm={tuple(g['identity']):g for g in state['groups']};scenes={(e['type_id'],e['level']):e['scene']for e in entries};ctls={c['type_code']:c for c in controls}
 def request(g,k):
  t,l,ctl,si=g['identity'];return make(scenes[(t,l)],ctls[ctl],si,combo['candidate_pool'][k-1],purpose='basis_reduction_'+scope)
 def consume():
  pending=state['pending'];cases=[]
  for item in pending['frontier']:
   c=request(gm[tuple(item['identity'])],item['index']);assert c['rollout_uid']==item['query_uid'];cases.append(c)
  rows=run_requests(cases,pending['stage'],eligible=combo.get('no_passage_preserves_four',False),ceiling=floor);labels=[]
  for item in pending['frontier']:
   g=gm[tuple(item['identity'])];r=rows.get(item['query_uid'])
   if r is None:g['gap']='infrastructure_incomplete_after_three_retries';continue
   assert not g['success']and g['searched']==item['index']-1
   q=dict(index=item['index'],query_uid=item['query_uid'],result=r);g['queries'].append(q);g['searched']=item['index']
   if r['success']:g.update(success=True,first_success=item['index'],solution=q)
   labels.append(dict(identity=g['identity'],**q))
  dump(HERE/(pending['stage']+'_labels.json'),labels);state['completed_stages'].append(pending['stage']);state['pending']=None;state['complete64']=all(g['success']or g['searched']==64 for g in state['groups']);dump(path,state)
 if state['pending']:consume()
 from value_review import review
 def prune_if_unhelpful():
  decision=review(state)
  if decision is None:return False
  state['user_early_stop']=decision;dump(path,state)
  dump(HERE/f'SCREENED_{code}.json',dict(reason=decision['reason'],user_authorized_early_stop=True,budget=decision['available_complete_budget'],review=decision))
  from analysis_results import write_report
  write_report();return True
 if prune_if_unhelpful():return state
 if code.startswith(('level3_','level2_'))and(HERE/'CHAIN_SCHEDULING_POLICY.json').exists():
  from chain_search import run as run_chain
  checkpoints=[8,12,16]if cap==16 else[cap]
  for checkpoint in checkpoints:
   run_chain(state,path,request,combo,checkpoint,floor)
   if prune_if_unhelpful():return state
  return state
 for k in range(1,cap+1):
  if prune_if_unhelpful():return state
  active=[g for g in state['groups']if not g['success']and not g['gap']and g['searched']<k]
  if not active:continue
  frontier=[]
  for g in active:
   assert g['searched']==k-1
   c=request(g,k);frontier.append(dict(identity=g['identity'],index=k,query_uid=c['rollout_uid']))
  state['pending']=dict(stage=f'{code}_{scope}_k{k:02d}',frontier=frontier);dump(path,state);consume()
 from analysis_results import write_report
 write_report();return state

def fixed(combo,make,scope='screen',plus=False,floor=80000):
 code=combo['code'];suffix='_plus'if plus else'';path=HERE/f'FIXED_{code}_{scope}{suffix}.json'
 if path.exists()and read(path)['complete']:return read(path)
 cases=[];records=[];sp=combo['fixed_plus']if plus else combo['fixed']
 for e in selected_entries(scope=='all'):
  for si in range(12):
   for ctl in controllers():
    c=make(e['scene'],ctl,si,sp,purpose='fixed_parameter_'+scope+suffix);cases.append(c);records.append(dict(identity=ident(e,ctl,si),query_uid=c['rollout_uid'],spec=sp))
 rows=run_requests(cases,f'fixed_{code}_{scope}{suffix}',eligible=combo.get('no_passage_preserves_four',False),ceiling=floor);data=dict(code=code,scope=scope,plus=plus,complete=len(rows)==len(cases),records=[dict(x,result=rows.get(x['query_uid']))for x in records]);dump(path,data);return data

def run_level(combos,make,floor=80000):
 from analysis_results import write_report,metrics
 # Every combination gets16 before any new64 extension in this level.
 for combo in combos:
  if combo['code']=='four_8':historical_four()
  else:search(combo,16,make,floor=floor)
 write_report();reference=metrics('reference');threshold=reference['coverage']['16']-10
 for combo in combos:
  code=combo['code'];m=metrics(code)
  marker=HERE/f'SCREENED_{code}.json'
  if marker.exists()and read(marker).get('user_authorized_early_stop'):continue
  if code=='reference'or m['coverage']['16']>=threshold:
   if code!='four_8':search(combo,64,make,floor=floor)
  else:
   dump(HERE/f'SCREENED_{code}.json',dict(reason='16次筛选未通过',coverage16=m['coverage']['16'],threshold=threshold,historical64_retained=code=='four_8'))
 write_report()
 for combo in combos:
  marker=HERE/f'SCREENED_{combo["code"]}.json'
  if marker.exists()and(combo['code'].startswith(('level3_','level2_')))and(HERE/'USER_EARLY_STOP_POLICY.json').exists():
   dump(HERE/f'FIXED_SKIPPED_USER_{combo["code"]}.json',dict(reason='用户授权价值早停后，不再为该弱候选追加固定参数对局。',retained_results=True))
  else:fixed(combo,make,floor=floor)
 write_report()

def best_qualified():
 from analysis_results import qualified,metrics
 combos=[c for c in read(HERE/'COMBINATIONS.json')['combinations']if c['dimension']<=4 and qualified(c['code'])]
 return sorted(combos,key=lambda c:(c['dimension'],-metrics(c['code'])['coverage']['64'],c['code']))

def main(make):
 from analysis_results import write_report,metrics,qualified,finalize
 from allscene import run_full
 data=read(HERE/'COMBINATIONS.json');level4=[c for c in data['combinations']if c['dimension']>=4]
 try:
  run_level(level4,make)
  # Definition-effect evidence belongs to the reference/first-level priority,
  # before lower-dimensional searches can consume the remaining hard budget.
  from corrections_analysis import run as corrections
  corrections(make)
  parents=[c for c in level4 if c['dimension']==4 and qualified(c['code'])]
  if parents:
   three=lower(3,parents,{c['code']:metrics(c['code'])['coverage']['64']for c in parents});run_level(three,make)
   parents3=[c for c in three if qualified(c['code'])]
   if parents3:
    two=lower(2,parents3,{c['code']:metrics(c['code'])['coverage']['64']for c in parents3})
    chosen=best_qualified();reserve=12000 if chosen and metrics(chosen[0]['code'])['hard64']>=36 else 0
    try:run_level(two,make,floor=80000-reserve)
    except PriorityPause:
     with(HERE/'ITERATION_LOG.md').open('a')as f:f.write('\n按预算优先级暂缓两系数级，保留12000局给已达标最小组合全场景；未完成项不记为失败。\n')
  chosen=best_qualified();minimum=chosen[0]if chosen else None
  fixed(next(c for c in read(HERE/'COMBINATIONS.json')['combinations']if c['code']=='reference'),make,plus=True)
  if minimum:fixed(minimum,make,plus=True)
  # Trigger only the completed, qualifying minimum, then the best four-free
  # comparator when different and resources permit. No ad-hoc new family.
  targets=[]
  if minimum and metrics(minimum['code'])['hard64']>=36:targets.append(minimum)
  fours=[c for c in chosen if c['dimension']==4]
  if fours:
   best4=min(fours,key=lambda c:(-metrics(c['code'])['coverage']['64'],c['code']))
   if metrics(best4['code'])['hard64']>=36 and best4['code']not in[c['code']for c in targets]:targets.append(best4)
  if targets:run_full(targets[0],make)
  # Complete a priority-paused two-free level after the first full test.
  data=read(HERE/'COMBINATIONS.json')
  pending2=[c for c in data['combinations']if c['dimension']==2 and not metrics(c['code'])['search_stage_finished']]
  if pending2:
   try:run_level([c for c in data['combinations']if c['dimension']==2],make)
   except PriorityPause:pass
  current=best_qualified();final_min=current[0]if current else None
  if final_min and(not minimum or final_min['code']!=minimum['code']):fixed(final_min,make,plus=True)
  if final_min and metrics(final_min['code'])['hard64']>=36 and final_min['code']not in[c['code']for c in targets]:run_full(final_min,make)
  if len(targets)>1:run_full(targets[1],make)
  finalize()
 except RuntimeError as exc:
  if'actual new rollout budget exhausted'not in str(exc):raise
  with(HERE/'GAPS.md').open('a')as f:f.write('\n达到80000局新运行硬上限；未完成请求保留，不算控制失败。\n')
  finalize(budget_exhausted=True)
 except PriorityPause:
  finalize(budget_exhausted=used()>=80000)
