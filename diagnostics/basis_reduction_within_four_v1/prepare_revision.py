"""Conservative evidence migration after the goal-yield end-condition fix."""
from core import *
from runner import sources

def main():
 cases={}
 for folder in [HERE,HERE/'inputs/revision1']:
  for path in folder.glob('*_planned.json'):
   for c in read(path).get('exact_configs',[]):cases[c['rollout_uid']]=c
 valid={};invalid=[];unmapped=[]
 for path in(HERE/'raw').glob('*.json'):
  r=read(path);uid=r['rollout_uid'];c=cases.get(uid)
  if c is None:unmapped.append(uid);continue
  current=sources(c['spec']);prior=c['controller_payload']['execution_sources'];differences={k for k in current.keys()|prior.keys()if current.get(k)!=prior.get(k)}
  assert differences=={str(HERE/'admission.py')},(uid,differences)
  with np.load(r['trace'])as tr:history=json.loads(str(tr['admission_json']))
  activated=any(any(m.get('goal_yield',[]))for step in history for m in step)
  if activated:
   invalid.append(dict(rollout_uid=uid,purpose=r['purpose'],reason='Goal-yield branch activated; conservatively rerun under corrected end condition.',trace=r['trace']));continue
  target=json.loads(json.dumps(c));target['controller_payload']['execution_sources']=current
  target['controller_uid']=db.uid('ctl',dict(scenario=target['scenario_uid'],payload=target['controller_payload']))
  target['rollout_uid']=db.uid('roll',dict(state=target['state_uid'],eta=target['eta_uid'],controller=target['controller_uid'],seed=target['seed_key']))
  valid[target['rollout_uid']]=dict(source_case=c,source_trace=r['trace'],reason='No goal-yield activation in complete saved trace; the only modified control branch was never entered; all old numerical statements in executed branches are identical.')
 dump(HERE/'REVISION_EQUIVALENT.json',dict(records=valid,count=len(valid),changed_file='admission.py',prior_definition='inputs/revision1/DESIGN_FREEZE.json',current_definition='DESIGN_FREEZE.json',rule='Any prior goal_yield activation is conservatively treated as affected and rerun; no activation guarantees persistence logic never changes the action.'))
 dump(HERE/'INVALIDATED_ROLLOUTS.json',dict(records=invalid,count=len(invalid),unmapped=unmapped,old_records_preserved=True,scope='Obsolete for current corrected definition; still valid historical evidence for the explicitly versioned old implementation.'))
 print('Revision bridge valid',len(valid),'affected',len(invalid),'unmapped',len(unmapped))

if __name__=='__main__':main()
