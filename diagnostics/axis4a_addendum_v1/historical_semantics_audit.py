"""Check cross-directory semantic candidates without opening round30 intermediate results."""
from common import *
from plan import signature

def main():
 names=read(HERE/'ORIGINAL_PROFILES.json');wanted_cp={v['payload']['flow_checkpoint_sha256'] for v in names.values()};targets={signature(s['physical']):s['state_uid'] for s in read(HERE/'EVAL_STATES_ORIGINAL.json')};matched={};profiles={};sources={}
 with rollout_db.connect(True) as db:
  for s in db.execute('SELECT state_uid,physical_state_json FROM state'):
   try:key=signature(json.loads(s['physical_state_json']))
   except (TypeError,ValueError):key=None
   if key in targets:matched[s['state_uid']]=targets[key]
  for sid in matched:
   sql='SELECT DISTINCT c.controller_uid,c.flow_checkpoint_sha256,c.config_json,e.path,e.experiment_uid FROM rollout r JOIN controller_config c ON r.controller_uid=c.controller_uid JOIN experiment e ON r.experiment_uid=e.experiment_uid WHERE r.state_uid=? AND e.path NOT LIKE ? AND e.experiment_uid!=?'
   for q in db.execute(sql,(sid,'%teacher_student_macflow%',EXP)):
    if q['flow_checkpoint_sha256'] not in wanted_cp:continue
    v=json.loads(q['config_json']);profiles[q['controller_uid']]=v;sources.setdefault(q['controller_uid'],set()).add(q['path'])
 out=[];nonbaseline=[]
 for cid,v in profiles.items():
  config=v.get('evaluation_config') or v.get('physical_config') or v.get('config') or v.get('cbf') or {};radius=config.get('obstacle_radius')
  item=dict(controller_uid=cid,source_paths=sorted(sources[cid]),radius=radius,horizon=v.get('horizon'),execution=v.get('round35_execution'),config=config,profile_keys=sorted(v));out.append(item)
  if radius is not None and radius!=.9:nonbaseline.append(item)
 write(HERE/'HISTORICAL_SEMANTIC_CANDIDATES.json',dict(utc=utc(),matched_physical_state_identities=len(matched),matching_checkpoint_profiles=len(profiles),candidate_profiles=out,nonbaseline_geometry_candidates=nonbaseline,round30_result_payloads_opened=False,scope='Metadata only, exact physical state and same frozencheckpoint. No label invalidation from provenance/encoding alone. All rawoldH700baseline labels are explicitR10/R35references.'))
 assert not nonbaseline,'Inspect potential old changed-radius compatible evidence before any more missing execution'
 print('Historical samephysical/samecheckpoint profiles',len(profiles),'nonbaseline geometry',len(nonbaseline),flush=True)
if __name__=='__main__':main()
