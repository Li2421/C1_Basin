"""Audit original-student and teacher evidence without creating trajectories."""
import sys,pathlib,json,hashlib,struct,datetime
H=pathlib.Path(__file__).resolve().parent;R=H.parents[1];D=R/'diagnostics';sys.dont_write_bytecode=True;sys.path.insert(0,str(R))
import shared_rollout_db
import shared_rollout_db.src.cache_writer as writer
from shared_rollout_db.src.rollout_db import connect,canonical
from shared_rollout_db.src.planner import preflight
assert pathlib.Path(shared_rollout_db.__file__).resolve().is_relative_to(R/'shared_rollout_db')
assert (writer.ROOT/'journals').resolve()==R/'shared_rollout_db/journals'
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(n,x):(H/n).write_text(json.dumps(x,indent=2))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def sig(p):
 return b''.join(struct.pack('<d',float(z)) for k in ['positions','velocities','goals'] for row in p[k] for z in row)
def main():
 states=read(H/'EVAL_STATES.json');checks=[];files=[]
 for path in [D/'teacher_student_macflow_v1c/main_protocol.json',D/'teacher_student_macflow_v1d/fair_protocol.json',D/'teacher_student_macflow_v1/teacher_protocol.json',D/'decentralized_pipeline_v1/main_protocol.json']:
  old=read(path)['states'];assert len(old)==32
  for a,b in zip(states,old):assert sig(a['physical'])==sig(b['physical']) and a['uid']==b['uid'],path
  checks.append(dict(path=str(path),sha256=sha(path),physical_bytes_and_rng_state_alias_match=True));files.append(path)
 cases=[c for c in read(D/'teacher_student_macflow_v1c/main_CASES.json') if c['arm'] in ['nominal only','safety baseline','F10','F13','S300','S500']]+[c for c in read(D/'teacher_student_macflow_v1d/fair_CASES.json') if c.get('kind')=='student'];assert len(cases)==6272
 students=read(H/'STUDENTS.json');teachers=read(H/'TEACHERS.json');rows=[]
 with connect(True) as db:
  for c in cases:
   i=next(i for i,t in enumerate(teachers) if t['name']==c['base_controller_name']);ctl=db.execute('SELECT * FROM controller_config WHERE controller_uid=?',(c['controller_uid'],)).fetchone();row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert ctl['flow_checkpoint_sha256']==students[str(i)]['checkpoint_sha256'];assert row and row['compatibility_quality']=='EXACT_REUSE' and not row['conflict_quarantined'];rows.append(dict(row))
 manifest=H/'V0_PLANNED_ROLLOUTS.json';write(manifest.name,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in cases]));pf=preflight(manifest);assert pf['summary']['ambiguous']==0;write('V0_CACHE_PREFLIGHT.json',pf);write('V0_DB_ROWS.json',rows)
 choices={}
 for folder in ['teacher_student_macflow_v1','teacher_student_macflow_v1b','teacher_student_macflow_v1c']:
  path=D/folder/'FROZEN_CHOICES.json'
  if path.exists():choices.update(read(path)['controllers']);files.append(path)
 assert all(t['name'] in choices for t in teachers);write('V0_FROZEN_CHOICES_REUSE.json',choices)
 write('REUSE_AUDIT.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS',paired_state_checks=checks,original_student_rows=len(rows),preserved_unknown=sum(r['numerical_failure'] for r in rows),sources={str(p):sha(p) for p in files},teacher_note='Distributed F10/F13/S300/S500 labels in round28, nominal/safety in round30. No exact fair-RHR teacher reference; report gap. No new teacher rollouts.',rollouts_generated=0));print('reuse audit PASS',len(rows))
if __name__=='__main__':main()
