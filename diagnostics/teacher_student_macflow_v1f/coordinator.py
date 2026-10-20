"""Persistent stage DAG. Only schedules work permitted by current user-wide quota."""
import os,sys,json,pathlib,subprocess,time,hashlib,datetime
H=pathlib.Path(__file__).resolve().parent;R=H.parents[1];P='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python';os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.path.insert(0,str(R))
import shared_rollout_db
import shared_rollout_db.src.cache_writer as writer
assert pathlib.Path(shared_rollout_db.__file__).resolve().is_relative_to(R/'shared_rollout_db')
assert (writer.ROOT/'journals').resolve()==R/'shared_rollout_db/journals'
from quota import census,submit
os.environ['TMPDIR']=str(H/'tmp');os.environ['MPLCONFIGDIR']=str(H/'mplconfig');os.environ['XDG_CACHE_HOME']=str(H/'cache')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(name,x):
 p=H/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
STATE=H/'coordinator_state.json';state=read(STATE) if STATE.exists() else {};gaps=read(H/'GAPS.json') if (H/'GAPS.json').exists() else []
def save():write('coordinator_state.json',state);write('GAPS.json',gaps)
def status(stage):
 c=census();write('RUN_STATUS.json',dict(utc=utc(),status='RUNNING',stage=stage,quota=c,gaps=gaps,completed_tasks=sum(v.get('state')=='done' for v in state.values()),active_tasks={k:v for k,v in state.items() if v.get('state')=='running'}))
def task(key,script,args,output,gpu=False):
 return dict(key=key,script=script,args=[str(x) for x in args],output=str(H/output),gpu=gpu)
def run_tasks(tasks,stage):
 while True:
  c=census();active={r['job'] for r in c['rows']};allfinished=True
  for t in tasks:
   k=t['key'];s=state.setdefault(k,dict(state='pending',task=t))
   if s['state']=='running' and s['job'] not in active:
    if pathlib.Path(t['output']).exists():s['state']='done'
    else:s['state']='gap';gaps.append(dict(utc=utc(),task=t,reason='Slurm job ended without required completion artifact; no automatic retry',job=s['job']))
    save()
   if s['state'] in ['done','gap']:continue
   allfinished=False
   if s['state']=='pending':
    args=[str(H/('gpu.sbatch' if t['gpu'] else 'cpu.sbatch'))]+([] if t['gpu'] else [t['script']])+t['args'];job=submit(args,gpu=int(t['gpu']))
    if job:s.update(state='running',job=job,submitted=utc());save()
  status(stage)
  if allfinished:return
  time.sleep(20)
def run_one(key,script,args,output):
 run_tasks([task(key,script,args,output)],key)
 return state[key]['state']=='done'
def freeze_impl():
 files=[p for p in H.rglob('*.py') if '/students/' not in str(p)]+list(H.glob('*.sbatch'))+[H/'PREREGISTRATION.md',H/'PREREGISTRATION.receipt.json',H/'UNIT_TESTS.json',H/'COPY_PROVENANCE.json',H/'COPY_DIFF.patch']
 write('IMPLEMENTATION_FROZEN.receipt.json',dict(utc=utc(),files={str(p):sha(p) for p in files},before_training=True,before_new_evaluation=True))
def training():
 # Each variant/teacher is independent; all final versions freeze before pilot labels.
 todo=[('V1',i) for i in range(14)]+[('V3',i) for i in range(14)]
 while True:
  c=census();active={r['job'] for r in c['rows']};pending=[];terminal=True
  for variant,i in todo:
   chain=[]
   if variant=='V3':chain.append(task(f'{variant}_{i}_train0','train.py',[i,variant,0],f'students/{variant}/{i}/offline.json',True))
   for rd in [1,2,3]:
    chain.extend([task(f'{variant}_{i}_collect{rd}','collect.py',[i,variant,rd],f'dagger/{variant}/{i}/round{rd}/COMPLETE.json'),task(f'{variant}_{i}_train{rd}','train.py',[i,variant,rd],f'students/{variant}/{i}/round{rd}.json',True)])
   chain.append(task(f'{variant}_{i}_gate','gates.py',[i,variant],f'gates/{variant}_{i}.json'))
   for t in chain:
    k=t['key'];s=state.setdefault(k,dict(state='pending',task=t))
    if s['state']=='running' and s['job'] not in active:
     if pathlib.Path(t['output']).exists():s['state']='done'
     else:s['state']='gap';gaps.append(dict(utc=utc(),task=t,job=s['job'],reason='No completion artifact; retain any partial evidence; dependent stages excluded'))
     save()
    if s['state']=='done':continue
    if s['state']=='gap':break
    terminal=False
    if s['state']=='pending':pending.append(t)
    break
  status('training_DAG')
  if terminal:return
  for t in pending:
   args=[str(H/('gpu.sbatch' if t['gpu'] else 'cpu.sbatch'))]+([] if t['gpu'] else [t['script']])+t['args'];job=submit(args,gpu=int(t['gpu']))
   if job:state[t['key']].update(state='running',job=job,submitted=utc());save()
  time.sleep(20)
def phase(name):
 if not run_one('design_'+name,'design.py',[name],name+'_PREFLIGHT.receipt.json'):return False
 workers=[task(f'{name}_worker{i}','evaluate.py',[name,i,18],f'{name}/worker{i}.json') for i in range(18)];run_tasks(workers,name)
 if any(state[t['key']]['state']=='gap' for t in workers):return False
 return run_one('postflight_'+name,'postflight.py',[name],name+'_MERGE_RECEIPT.json')
def audit():
 # Official and protected trees are only read by this utility.
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
 with (H/'integrity_end.log').open('w') as f:subprocess.run([P,'-B',str(H/'integrity.py'),'END'],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 sources={};inconsistent=[]
 for p in H.glob('imports*.json'):
  for path,h in read(p).get('imported_sources',{}).items():
   if pathlib.Path(path).is_relative_to(H):continue
   if path in sources and sources[path]!=h:inconsistent.append(path)
   sources[path]=h
 changed={p:dict(at_import=h,current=sha(p)) for p,h in sources.items() if sha(p)!=h}
 write('IMPORTED_SOURCE_AUDIT.json',dict(utc=utc(),source_sha256=sources,changed=changed,inconsistent=inconsistent,passed=not changed and not inconsistent))
def main():
 (H/'COORDINATOR_PID').write_text(str(os.getpid()))
 if not run_one('unit_tests','tests.py',[],'UNIT_TESTS.json'):raise RuntimeError('unit tests blocked')
 if not run_one('replay','replay.py',[],'ROUND28_REPLAY_GATE.receipt.json'):raise RuntimeError('mandatory reproduction blocked; no student rollout')
 if not (H/'IMPLEMENTATION_FROZEN.receipt.json').exists():freeze_impl()
 training()
 if not list((H/'gates').glob('V*.json')):raise RuntimeError('All training variants blocked')
 if not run_one('distillation_freeze','gates.py',['freeze'],'TRAINING_FROZEN.receipt.json'):raise RuntimeError('distillation freeze blocked')
 if not phase('pilot'):raise RuntimeError('pilot stage incomplete')
 if not read(H/'CAPABILITY_GATE.json')['eligible']:raise RuntimeError('No student passed both gates; main endpoints unavailable')
 if not run_one('probe_design','probes.py',['design'],'probe_PREFLIGHT.receipt.json'):raise RuntimeError('probe design blocked')
 tasks=[task(f'probe_worker{i}','probes.py',['worker',i,18],f'probe_worker{i}.json') for i in range(18)];run_tasks(tasks,'probes')
 if any(state[t['key']]['state']=='gap' for t in tasks):raise RuntimeError('probe workers incomplete')
 for key,script,args,out in [('probe_freeze','probes.py',['freeze'],'R5_FROZEN.receipt.json'),('freeze_choices','freeze_choices.py',[],'CHOICES_FROZEN.receipt.json')]:
  if not run_one(key,script,args,out):raise RuntimeError(key+' blocked')
 if not phase('main'):raise RuntimeError('main stage incomplete; preserve unknown and do not rerun')
 if not run_one('analysis','analysis.py',[],'results.json'):raise RuntimeError('analysis blocked')
 # Plotting is an optional independent descriptive delivery; same CPU quota.
 plotenv='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python'
 if pathlib.Path(plotenv).exists():
  while census()['cpu']>=18:status('await_plot_quota');time.sleep(20)
  with (H/'plots.log').open('w') as f:r=subprocess.run([plotenv,'-B',str(H/'plot_training.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:gaps.append(dict(stage='training_curves',reason='Plotting failed; histories retained'))
 audit()
 if not run_one('deliver','deliver.py',[],'DELIVERY.receipt.json'):raise RuntimeError('delivery blocked')
 audit();write('RUN_STATUS.json',dict(utc=utc(),status='COMPLETE_WITH_GAPS' if gaps else 'COMPLETE',gaps=gaps,resource_exception='RESOURCE_EXCEPTION.json'))
if __name__=='__main__':
 try:main()
 except Exception as e:
  gaps.append(dict(utc=utc(),stage='coordinator',type=type(e).__name__,reason=str(e)));save();audit();write('RUN_STATUS.json',dict(utc=utc(),status='INCOMPLETE',gaps=gaps));raise
