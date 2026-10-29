"""Conservative evidence migration after the goal-yield end-condition fix."""
from core import *
from runner import sources

def main():
 cases={}
 for folder in [HERE,HERE/'inputs/revision1',HERE/'inputs/import_relocation']:
  for path in folder.glob('*_planned.json'):
   for c in read(path).get('exact_configs',[]):cases[c['rollout_uid']]=c
 valid={};invalid=[];unmapped=[]
 for path in(HERE/'raw').glob('*.json'):
  r=read(path);uid=r['rollout_uid'];c=cases.get(uid)
  if c is None:unmapped.append(uid);continue
  current=sources(c['spec']);prior=c['controller_payload']['execution_sources'];differences={k for k in current.keys()|prior.keys()if current.get(k)!=prior.get(k)}
  allowed={str(HERE/name)for name in ['admission.py','core.py','runner.py','generated/backend/exact_projection_retry.py']}
  assert differences and differences<=allowed,(uid,differences)
  with np.load(r['trace'])as tr:history=json.loads(str(tr['admission_json']))
  activated=any(any(m.get('goal_yield',[]))for step in history for m in step)
  admission_changed=prior.get(str(HERE/'admission.py'))!=current.get(str(HERE/'admission.py'))
  if activated and admission_changed:
   invalid.append(dict(rollout_uid=uid,purpose=r['purpose'],reason='Goal-yield branch activated; conservatively rerun under corrected end condition.',trace=r['trace']));continue
  target=json.loads(json.dumps(c));target['controller_payload']['execution_sources']=current
  target['controller_uid']=db.uid('ctl',dict(scenario=target['scenario_uid'],payload=target['controller_payload']))
  target['rollout_uid']=db.uid('roll',dict(state=target['state_uid'],eta=target['eta_uid'],controller=target['controller_uid'],seed=target['seed_key']))
  valid[target['rollout_uid']]=dict(source_case=c,source_trace=r['trace'],requires_import_certificate=True,reason='Byte-identical retry module relocated under its canonical name. If goal-yield implementation differs, complete trace has no activation, proving the changed branch was never entered. Current goal-yield traces remain valid under unchanged numerical code.')
 dump(HERE/'REVISION_EQUIVALENT.json',dict(records=valid,count=len(valid),prior_definitions=['inputs/revision1/DESIGN_FREEZE.json','inputs/import_relocation/DESIGN_FREEZE.json'],current_definition='DESIGN_FREEZE.json',rule='If admission implementation changed, any goal_yield activation conservatively requires rerun. Otherwise only byte-identical dependency relocation changed executed code provenance. All aliases require the current200-trajectory import relocation certificate.'))
 dump(HERE/'INVALIDATED_ROLLOUTS.json',dict(records=invalid,count=len(invalid),unmapped=unmapped,old_records_preserved=True,scope='Obsolete for current corrected definition; still valid historical evidence for the explicitly versioned old implementation.'))
 print('Revision bridge valid',len(valid),'affected',len(invalid),'unmapped',len(unmapped))

if __name__=='__main__':main()
