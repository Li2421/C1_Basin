from common import *
from shared_rollout_db.src.rollout_db import connect
from audit_utils import source_inventory
import subprocess,shlex
G=guard()
def merge(phase):
 p=read(HERE/f'{phase}_protocol.json');cases=read(HERE/f'{phase}_CASES.json');missing=read(HERE/f'{phase}_MISSING_CASES.json');assert all((HERE/phase/'raw'/f"{c['rollout_uid']}.json").exists() for c in missing)
 ps=subprocess.check_output(['ps','-eo','pid,args'],text=True);sq=subprocess.check_output(['squeue','-u',os.environ.get('USER','zhihan'),'-h','-o','%i|%j|%T|%C|%b'],text=True);active=[]
 for line in ps.splitlines()[1:]:
  try:tokens=shlex.split(line.split(None,1)[1])
  except (ValueError,IndexError):continue
  if tokens and 'python' in Path(tokens[0]).name and 'shared_rollout_db.src.merger' in tokens:active.append(line)
 assert not active and 'merger' not in sq.lower(),active
 write(HERE/f'{phase}_MERGER_PREFLIGHT.json',dict(utc=utc(),ps=ps,squeue=sq,active_mergers=active,guard=guard()))
 cmd=[sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',p['experiment_uid']];r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'});(HERE/f'{phase}_MERGE.log').write_text(r.stdout+r.stderr);assert r.returncode==0
 if phase=='probe':
  with connect(True) as db:assert db.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?',(p['experiment_uid'],)).fetchone()[0]==0
  seal('probe_MERGE_RECEIPT.json',[HERE/'probe_MERGE.log',HERE/'probe_MERGER_PREFLIGHT.json'],task_rollout_rows=0);return
 found=[]
 with connect(True) as db:
  for c in cases:
   row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert row and not row['conflict_quarantined'] and row['compatibility_quality']=='EXACT_REUSE';found.append(dict(row))
 pf=planner.preflight(HERE/f'{phase}_PLANNED_ROLLOUTS.json');write(HERE/f'{phase}_CACHE_POSTFLIGHT.json',pf);write(HERE/f'{phase}_DB_ROWS.json',found);unknown=sum(r['numerical_failure'] for r in found);assert pf['summary']['ambiguous']==0 and pf['summary']['genuinely_missing']==unknown
 seal(f'{phase}_MERGE_RECEIPT.json',[HERE/f'{phase}_DB_ROWS.json',HERE/f'{phase}_CACHE_POSTFLIGHT.json',HERE/f'{phase}_MERGER_PREFLIGHT.json'],records=len(found),unknown=unknown,official_command=cmd)
 source_inventory('imports_postflight_'+phase+'.json')
if __name__=='__main__':merge(sys.argv[1])
