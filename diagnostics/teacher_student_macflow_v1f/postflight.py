from core import *
from guard import guard
from shared_rollout_db.src.rollout_db import connect
from shared_rollout_db.src.planner import preflight
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
 found=[]
 with connect(True) as db:
  for c in cases:
   row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert row and not row['conflict_quarantined'] and row['compatibility_quality']=='EXACT_REUSE';found.append(dict(row))
 pf=preflight(HERE/f'{phase}_PLANNED_ROLLOUTS.json');write(HERE/f'{phase}_CACHE_POSTFLIGHT.json',pf);write(HERE/f'{phase}_DB_ROWS.json',found);unknown=sum(r['numerical_failure'] for r in found);assert pf['summary']['ambiguous']==0 and pf['summary']['genuinely_missing']==unknown
 seal(f'{phase}_MERGE_RECEIPT.json',[HERE/f'{phase}_DB_ROWS.json',HERE/f'{phase}_CACHE_POSTFLIGHT.json',HERE/f'{phase}_MERGER_PREFLIGHT.json'],records=len(found),unknown=unknown,official_command=cmd)
 if phase=='pilot':
  byid={r['rollout_uid']:r for r in found};rates={};eligible=[];excluded=[]
  for name in p['controller_order']:
   rates[name]={}
   for arm in ['safety baseline','S500']:
    rr=[byid[c['rollout_uid']] for c in cases if c['base_controller_name']==name and c['arm']==arm];assert len(rr)==16;rates[name][arm]=dict(lower=sum(r['success'] and not r['numerical_failure'] for r in rr)/16,upper=sum(r['success'] or r['numerical_failure'] for r in rr)/16,unknown=sum(r['numerical_failure'] for r in rr))
   a,b=[rates[name][arm]['lower'] for arm in ['safety baseline','S500']]
   if (a<.15 and b<.15) or (a>.98 and b>.98):excluded.append(dict(name=name,reason='both_low' if a<.15 else 'both_high'))
   else:eligible.append(name)
  write(HERE/'CAPABILITY_GATE.json',dict(utc=utc(),eligible=eligible,excluded=excluded,rates=rates));seal('CAPABILITY_FROZEN.receipt.json',[HERE/'CAPABILITY_GATE.json',HERE/'pilot_DB_ROWS.json'],before_main_labels=True)
 source_inventory(f'imports_postflight_{phase}.json')
if __name__=='__main__':merge(sys.argv[1])
