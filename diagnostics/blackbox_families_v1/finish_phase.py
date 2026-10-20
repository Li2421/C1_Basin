from common import *
import subprocess,fcntl
phase=sys.argv[1]
assert len(list((HERE/phase/'raw').glob('*.json')))==len(read(HERE/f'{phase}_MISSING_CASES.json'))
with (HERE/'OFFICIAL_MERGER.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with (HERE/f'{phase}_MERGE.log').open('w') as out:subprocess.run([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
post=planner.preflight(HERE/f'{phase}_PLANNED_ROLLOUTS.json');write(HERE/f'{phase}_POSTFLIGHT.json',post)
if not phase.startswith('probe_'):
 from shared_rollout_db.src.rollout_db import connect
 with connect(True) as db:
  absent=[];unknown=[]
  for c in read(HERE/f'{phase}_CASES.json'):
   r=db.execute('SELECT numerical_failure FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
   if r is None:absent.append(c['rollout_uid'])
   elif r['numerical_failure']:unknown.append(c['rollout_uid'])
 write(HERE/f'{phase}_POSTFLIGHT_UNKNOWN.json',dict(utc=utc(),absent_rows=absent,unknown_preserved=unknown,no_unknown_reruns=True))
 if phase!='gate_flow':assert not absent,absent
if phase in ['gate_flow','gate_flow_corrected']:
 checks=[]
 for c in read(HERE/f'{phase}_CASES.json'):
  r=read(HERE/phase/'raw'/f"{c['rollout_uid']}.json");old=read(c['reference_path']);assert sha(c['reference_path'])==c['reference_sha256'];a=np.load(r['trace_path'])['positions'];b=np.load(old['trace_path'])['positions'];v=dict(reference_path=c['reference_path'],first_action=np.asarray(r['first_action'],np.float64).tobytes()==np.asarray(old['first_action'],np.float64).tobytes(),positions=a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),outcome=r['outcome']==old['outcome'],steps=r['episode_length']==old['episode_length']);v['passed']=all(v[k] for k in ['first_action','positions','outcome','steps']);checks.append(v)
 write(HERE/'FLOW_REPRODUCTION_GATE.json',dict(utc=utc(),passed=len(checks)==16 and all(c['passed'] for c in checks),checks=checks));print('Flow reproduction',all(c['passed'] for c in checks))
print(phase,post['summary'])
