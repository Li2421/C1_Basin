"""Official exact lookup and explicitly certified cross-version semantic reuse."""
from dispatch import *
from execution_control import PriorityPause
from budget import used
_PREFIX_ALIASES=None
_SOURCE_RECORDS={}

def archived_sources(old_uids,stage):
 global _PREFIX_ALIASES
 if _PREFIX_ALIASES is None:
  _PREFIX_ALIASES={q['target_query_uid']:q['rollout_uid']for g in read(HERE/'inputs/v3/SEARCH_STATE.json')['groups']if g['family']==4 for q in g['queries']if q.get('target_query_uid',q['rollout_uid'])!=q['rollout_uid']and q['success']}
 selected={uid:_PREFIX_ALIASES[uid]for uid in old_uids if uid in _PREFIX_ALIASES}
 if not selected:return{}
 requests=[]
 with db.connect(True)as con:
  for uid in sorted(set(selected.values())):
   row=con.execute('SELECT state_uid,eta_uid,controller_uid,seed_key,success,conflict_quarantined FROM rollout WHERE rollout_uid=?',(uid,)).fetchone()
   assert row and row['success']and not row['conflict_quarantined'],('Invalid archived prefix source',uid)
   requests.append({k:row[k]for k in('state_uid','eta_uid','controller_uid','seed_key')})
   if uid not in _SOURCE_RECORDS:
    source=con.execute('SELECT sf.path,rs.source_line FROM rollout_source rs JOIN source_file sf ON rs.source_uid=sf.source_uid WHERE rs.rollout_uid=? LIMIT 1',(uid,)).fetchone();assert source
    with Path(source['path']).open()as f:_SOURCE_RECORDS[uid]=next(json.loads(line)for i,line in enumerate(f,1)if i==source['source_line'])
 suffix=digest(sorted(selected.items()))[:12];path=HERE/f'{stage}_archived_{suffix}_planned.json';dump(path,dict(requests=requests,v3_target_to_source=selected,certificate='inputs/v3/PREFIX_CHECK.json'))
 pf=planner.preflight(path);assert pf['summary']['ambiguous']==0;dump(HERE/f'{stage}_archived_{suffix}_preflight.json',pf)
 return{target:_SOURCE_RECORDS[source]for target,source in selected.items()}

def legacy_case(scene,ctl,si,spec):
 c=case(scene,ctl,si,spec,purpose='v3_historical_lookup');old=read(HERE/'inputs/V3_CONTROLLER_TEMPLATE.json')
 c['controller_payload']['execution_sources']=old['execution_sources']
 if'exact_acceleration'in old:c['controller_payload']['exact_acceleration']=old['exact_acceleration']
 c['controller_uid']=db.uid('ctl',dict(scenario=c['scenario_uid'],payload=c['controller_payload']))
 c['rollout_uid']=db.uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']))
 return c

def run_requests(cases,stage,eligible=False,worker_entry='worker_recorded.py',ceiling=80000,resolve_only=False):
 cases=list({c['rollout_uid']:c for c in cases}.values());rows={};missing=[];old_requests=[];links=[]
 certified=(HERE/'EQUIVALENCE.json').exists()and read(HERE/'EQUIVALENCE.json')['passed']
 no_pass=(HERE/'NO_PASSAGE_EQUIVALENCE.json').exists()and read(HERE/'NO_PASSAGE_EQUIVALENCE.json')['passed']
 passages=read(HERE/'PASSAGES.json')['by_scene'];scenes={e['scene']['name']:e['scene']for e in read(HERE/'SCENES.json')['entries']}
 # New exact identities are always preflighted, even when semantic aliasing
 # will eliminate actual execution; a certified alias is never an exact hit.
 _,_,new_done=register(cases,stage+'_new_preflight')
 revision=read(HERE/'REVISION_EQUIVALENT.json')['records']if(HERE/'REVISION_EQUIVALENT.json').exists()else{}
 import_cert=(HERE/'IMPORT_RELOCATION_CERT.json').exists()and read(HERE/'IMPORT_RELOCATION_CERT.json')['passed']
 for c in cases:
  if c['rollout_uid']in new_done:rows[c['rollout_uid']]=new_done[c['rollout_uid']];links.append(dict(target_query_uid=c['rollout_uid'],source_rollout_uid=c['rollout_uid'],kind='official_exact_reuse'));continue
  if c['rollout_uid']in revision and(not revision[c['rollout_uid']].get('requires_import_certificate')or import_cert):
   bridge=revision[c['rollout_uid']];old_requests.append(bridge['source_case']);links.append(dict(target_query_uid=c['rollout_uid'],source_rollout_uid=bridge['source_case']['rollout_uid'],kind='certified_semantic_alias_pending',revision_certificate='REVISION_EQUIVALENT.json'));continue
  sp=c['spec'];can_alias=eligible and certified and(not sp.get('one_way')or(no_pass and not passages[c['scene']]['passages']))and not sp.get('legacy_admission')
  if can_alias:
   base={k:v for k,v in sp.items()if k!='one_way'}
   # Match explicit-zero dictionary representation in the archived pool.
   for old in read(HERE/'FAMILIES_64.json')['families'][4]['candidate_pool']:
    if old['eta']==base['eta']and all(old['coefficients'].get(n,0.)==base['coefficients'].get(n,0.)for n in('T1','C2')):base=old;break
   old=legacy_case(scenes[c['scene']],c['controller_payload']['nominal'],c['state_index'],base);old_requests.append(old);links.append(dict(target_query_uid=c['rollout_uid'],source_rollout_uid=old['rollout_uid'],kind='certified_semantic_alias_pending'));continue
  missing.append(c)
 old_done={}
 if old_requests:
  # A resumed stage can have more new exact hits than its first invocation.
  # Keep each historical subset immutable under its own content identity.
  subset=digest([c['rollout_uid']for c in old_requests])[:12]
  _,_,old_done=register(old_requests,stage+'_historical_'+subset)
 archived=archived_sources([link['source_rollout_uid']for link in links if link['kind']=='certified_semantic_alias_pending'and link['source_rollout_uid']not in old_done],stage)
 requested={c['rollout_uid']:c for c in cases}
 for link in links:
  if link['kind']!='certified_semantic_alias_pending':continue
  uid=link['target_query_uid'];r=old_done.get(link['source_rollout_uid'])
  if r is None and link['source_rollout_uid']in archived:
   link['v3_archived_query_uid']=link['source_rollout_uid'];r=archived[link['source_rollout_uid']];link['source_rollout_uid']=r['rollout_uid'];link['prefix_certificate']='inputs/v3/PREFIX_CHECK.json'
  if r is None:missing.append(requested[uid]);link['kind']='historical_missing_new_execution_required'
  else:
   link['kind']='certified_semantic_alias';link['certificate']=link.get('revision_certificate')or('NO_PASSAGE_EQUIVALENCE.json'if requested[uid]['spec'].get('one_way')else'EQUIVALENCE.json')
   if requested[uid]['spec'].get('one_way')and not passages[requested[uid]['scene']]['passages']:
    link['logical_admission_history']=dict(encoding='constant empty state for every step',steps=r['episode_length'],passage_turns=[],goal_yielders=[],reason='Frozen geometry contains no passage; sourceNPZ is immutable and need not contain the later diagnostic field.')
   rows[uid]=dict(r,logical_query_uid=uid,reuse_reason=link['kind'])
 if missing and not resolve_only:
  if used()+len(missing)>ceiling:raise PriorityPause('Insufficient new-launch allowance after exact and certified reuse; priority reservation retained')
  for r in execute(missing,stage,worker_entry=worker_entry):rows[r['rollout_uid']]=r
 manifest=HERE/'REUSE_MANIFEST.json';m=read(manifest)if manifest.exists()else dict(records={})
 for link in links:
  if link['kind']in('official_exact_reuse','certified_semantic_alias'):m['records'][link['target_query_uid']]=dict(link,stage=stage)
 dump(manifest,m);dump(HERE/f'{stage}_all_requests.json',dict(exact_configs=cases,records=links))
 return (rows,missing)if resolve_only else rows
