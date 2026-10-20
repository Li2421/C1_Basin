from common import *
import subprocess,fcntl,time

def queue():return subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C'],text=True)
def running(stage):return any(x.split('|')[1]=='r37_'+stage for x in queue().splitlines())
def call(script,*args):
 with (HERE/(script.replace('.py','')+'_'+('_'.join(map(str,args)) or 'main')+'.log')).open('a') as f:subprocess.run([sys.executable,'-B',str(HERE/script),*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def lookup(c,stage):
 for other in [stage,'A','GA3','B','gate']:
  f=HERE/other/'raw'/f"{c['rollout_uid']}.json"
  if f.exists():return f
 with rollout_db.connect(True) as db:
  row=db.execute('SELECT original_source_file FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
 if row:
  # Official source normally a merged journal: locate the exact raw hash/UID.
  f=Path(row[0]);f=f if f.is_absolute() else ROOT/f
  if f.suffix in ['.jsonl','.merged']:
   if not f.exists() and f.suffix=='.jsonl':f=f.with_suffix('.merged')
   for line in f.read_text().splitlines():
    q=json.loads(line)
    if q.get('case',{}).get('rollout_uid')==c['rollout_uid']:
     out=HERE/'reused_raw'/f"{c['rollout_uid']}.json";write(out,q);return out
  elif f.exists() and f.suffix=='.json':return f
 raise RuntimeError('Cannot locate exact existing evidence '+c['rollout_uid'])
def merge(stage):
 rows=read(HERE/f'{stage}_CASES.json');ps_before=None;q_before=None
 with (HERE/'OFFICIAL_MERGER.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  while True:
   ps_before=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True);q_before=queue();busy=[x for x in ps_before.splitlines() if '-m shared_rollout_db.src.merger' in x and 'bash -c' not in x];jobbusy=any('merge' in x.split('|')[1].lower() and x.split('|')[2] in ['RUNNING','CONFIGURING','COMPLETING'] for x in q_before.splitlines())
   if not busy and not jobbusy:break
   time.sleep(5)
  write(HERE/f'{stage}_MERGER_PREFLIGHT.json',dict(utc=utc(),ps=ps_before,squeue=q_before,no_other_merger=True))
  with (HERE/f'{stage}_MERGE.log').open('a') as f:subprocess.run([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 refs=[];unknown=0;fresh=0
 with rollout_db.connect(True) as db:
  for c in rows:
   f=lookup(c,stage);q=read(f);r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert r and not r['conflict_quarantined'],c['rollout_uid'];assert r['raw_record_hash']==hashlib.sha256(rollout_db.canonical(q).encode()).hexdigest();assert all(q[k]==r[k] for k in ['success','collision','timeout','deadlock','numerical_failure']);refs.append(dict(rollout_uid=c['rollout_uid'],raw_path=str(f),raw_sha256=sha(f),source_experiment_uid=q['experiment_uid']));unknown+=q['numerical_failure'];fresh+=q['experiment_uid']==EXP
 write(HERE/f'{stage}_RAW_INDEX.json',refs);write(HERE/f'{stage}_POSTFLIGHT.json',dict(utc=utc(),verified=len(rows),unknown=unknown,fresh_experiment_rows=fresh,planner=planner.preflight(HERE/f'{stage}_PLANNED_ROLLOUTS.json')));seal(f'{stage}_NUMERICAL_COMPLETE.receipt.json',[HERE/f'{stage}_CASES.json',HERE/f'{stage}_RAW_INDEX.json',HERE/f'{stage}_POSTFLIGHT.json'],records=len(rows),unknown=unknown)
def wait(stage):
 if (HERE/f'{stage}_NUMERICAL_COMPLETE.receipt.json').exists():return
 while running(stage):time.sleep(5)
 assert all((HERE/stage/'raw'/f"{c['rollout_uid']}.json").exists() for c in read(HERE/f'{stage}_MISSING_CASES.json')),'No automatic retry of failed or missing run'
 merge(stage)
if __name__=='__main__':wait(sys.argv[1])
