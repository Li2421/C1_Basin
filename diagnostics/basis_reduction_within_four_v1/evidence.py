"""Official exact lookup and explicitly certified cross-version semantic reuse."""
from dispatch import *

def legacy_case(scene,ctl,si,spec):
 c=case(scene,ctl,si,spec,purpose='v3_historical_lookup');old=read(HERE/'inputs/V3_CONTROLLER_TEMPLATE.json')
 c['controller_payload']['execution_sources']=old['execution_sources']
 if'exact_acceleration'in old:c['controller_payload']['exact_acceleration']=old['exact_acceleration']
 c['controller_uid']=db.uid('ctl',dict(scenario=c['scenario_uid'],payload=c['controller_payload']))
 c['rollout_uid']=db.uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']))
 return c

def run_requests(cases,stage,eligible=False):
 cases=list({c['rollout_uid']:c for c in cases}.values());rows={};missing=[];old_requests=[];links=[]
 certified=(HERE/'EQUIVALENCE.json').exists()and read(HERE/'EQUIVALENCE.json')['passed']
 no_pass=(HERE/'NO_PASSAGE_EQUIVALENCE.json').exists()and read(HERE/'NO_PASSAGE_EQUIVALENCE.json')['passed']
 passages=read(HERE/'PASSAGES.json')['by_scene'];scenes={e['scene']['name']:e['scene']for e in read(HERE/'SCENES.json')['entries']}
 # New exact identities are always preflighted, even when semantic aliasing
 # will eliminate actual execution; a certified alias is never an exact hit.
 _,_,new_done=register(cases,stage+'_new_preflight')
 for c in cases:
  if c['rollout_uid']in new_done:rows[c['rollout_uid']]=new_done[c['rollout_uid']];links.append(dict(target_query_uid=c['rollout_uid'],source_rollout_uid=c['rollout_uid'],kind='official_exact_reuse'));continue
  sp=c['spec'];can_alias=eligible and certified and(not sp.get('one_way')or(no_pass and not passages[c['scene']]['passages']))and not sp.get('legacy_admission')
  if can_alias:
   base={k:v for k,v in sp.items()if k!='one_way'}
   # Match explicit-zero dictionary representation in the archived pool.
   for old in read(HERE/'FAMILIES_64.json')['families'][4]['candidate_pool']:
    if old['eta']==base['eta']and all(old['coefficients'].get(n,0.)==base['coefficients'].get(n,0.)for n in('T1','C2')):base=old;break
   old=legacy_case(scenes[c['scene']],c['controller_payload']['nominal'],c['state_index'],base);old_requests.append(old);links.append(dict(target_query_uid=c['rollout_uid'],source_rollout_uid=old['rollout_uid'],kind='certified_semantic_alias_pending'));continue
  missing.append(c)
 old_done={}
 if old_requests:_,_,old_done=register(old_requests,stage+'_historical_preflight')
 requested={c['rollout_uid']:c for c in cases}
 for link in links:
  if link['kind']!='certified_semantic_alias_pending':continue
  uid=link['target_query_uid'];r=old_done.get(link['source_rollout_uid'])
  if r is None:missing.append(requested[uid]);link['kind']='historical_missing_new_execution_required'
  else:
   link['kind']='certified_semantic_alias';link['certificate']='NO_PASSAGE_EQUIVALENCE.json'if requested[uid]['spec'].get('one_way')else'EQUIVALENCE.json'
   rows[uid]=dict(r,logical_query_uid=uid,reuse_reason=link['kind'])
 if missing:
  for r in execute(missing,stage):rows[r['rollout_uid']]=r
 manifest=HERE/'REUSE_MANIFEST.json';m=read(manifest)if manifest.exists()else dict(records={})
 for link in links:
  if link['kind']in('official_exact_reuse','certified_semantic_alias'):m['records'][link['target_query_uid']]=dict(link,stage=stage)
 dump(manifest,m);dump(HERE/f'{stage}_all_requests.json',dict(exact_configs=cases,records=links))
 return rows
