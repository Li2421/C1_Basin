"""Independent native statistical calls; fixed RNG and no controller/physics calls."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from statistics_correction_header_20261023 import FIXED,LAYERS,SEMANTICS,diagnostic_counts,matched,class_splits,wandering_rates
from statistics_header_reader_20261023 import summarize
import signal,subprocess,re

def task_manifest():
 tasks=[]
 for scene in NAMES:
  tasks.append(dict(kind='scene',scene=scene))
  for sem in ('original','revised'):
   for layer in LAYERS:tasks.append(dict(kind='split',scene=scene,semantics=sem,layer=layer))
 for kind in ('old_fixed_pool','old_excluded_pool','new_fixed_pool','old_learned_pool','new_learned_pool','old_hover','new_hover'):tasks.append(dict(kind=kind))
 return {uid('numeric',t):t for t in tasks}

def calculate(task,old,new,census,classes,old_sources,new_sources):
 kind=task['kind'];arms=[(l,m) for l in LAYERS for m in FIXED]
 if kind=='scene':
  scene=task['scene'];previous=read(HERE/'COMPLETED_SCENE_RESULTS.json')['results'][scene];crossfit=read(HERE/'COMPLETED_FIXED_CROSSFIT.json')['results'][scene]
  learned=diagnostic_counts(previous['learned_evaluation'],old)
  if scene=='lit_circle_five':learned=dict(status='N/A',reason='用户决定停止：修正语义修订；已落盘旧标签保留，不执行旧语义评估')
  entry=dict(fixed_full=diagnostic_counts(previous['fixed_full'],old),excluding_controller_failures=summarize(old,arms,[scene],True,census),learned_evaluation=learned,crossfit_fixed_oracle=crossfit,semantics=SEMANTICS['original'],legacy_exclusion_note='Historical threshold-only exclusion; superseded by instance_class_gain_splits')
  subset=matched(new,scene)
  entry['revised']=dict(fixed_full=summarize(new,arms,[scene]),learned_evaluation=summarize(subset,[('distributed',m) for m in (*FIXED,'learned_scene')],[scene]),semantics=SEMANTICS['revised'],central_scene_learning='N/A: not requested for revised semantics')
  return entry
 if kind=='split':
  rows=old if task['semantics']=='original' else new;layer=task['layer'];scene=task['scene']
  selected=[r for r in rows if r['scene']==scene and r['layer']==layer]
  return class_splits(selected,scene,classes)[layer]
 if kind in ('old_fixed_pool','old_excluded_pool','new_fixed_pool'):
  rows=new if kind=='new_fixed_pool' else old
  return summarize(rows,arms,list(NAMES),kind=='old_excluded_pool',census)
 if kind in ('old_learned_pool','new_learned_pool'):
  rows=new if kind=='new_learned_pool' else old;scenes=[s for s in NAMES if matched(rows,s)];subset=[r for s in scenes for r in matched(rows,s)]
  return summarize(subset,[('distributed',m) for m in (*FIXED,'learned_scene')],scenes)
 if kind in ('old_hover','new_hover'):return wandering_rates(new if kind=='new_hover' else old,old_sources+new_sources)
 raise ValueError(task)

def worker(key):
 # Pin to this job's own allocation, never the local coordinator core.
 assert os.environ.get('SLURM_JOB_ID'),'Native numeric tasks require a Slurm allocation'
 details=subprocess.check_output(['scontrol','show','job','-dd',os.environ['SLURM_JOB_ID']],text=True)
 assigned=set()
 for value in re.findall(r'CPU_IDs=(\S+)',details):
     for item in value.split(','):
         if '-' in item:
             left,right=map(int,item.split('-'));assigned.update(range(left,right+1))
         elif item.isdigit():assigned.add(int(item))
 assert assigned,'Missing allocated CPU IDs'
 chosen=min(assigned)
 for thread in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(thread.name),{chosen})
 # A resource signal never interrupts the current statistical call.
 signal.signal(signal.SIGUSR1,lambda *args:None);signal.signal(signal.SIGTERM,lambda *args:None)
 verify_receipt('CORRECTION_PARALLEL_NUMERICS_READY_20261023.receipt.json');verify_receipt('PARALLEL_NUMERIC_INPUTS_20261023.receipt.json')
 manifest=task_manifest();task=manifest[key];folder=HERE/'parallel_numeric_fragments_20261023';folder.mkdir(exist_ok=True);target=folder/(key+'.json');proof=folder/(key+'.receipt.json');input_hash=sha(HERE/'PARALLEL_NUMERIC_INPUTS_20261023.receipt.json')
 if proof.exists():verify_receipt(str(proof.relative_to(HERE)));return
 if target.exists():
  result=read(target);assert result['task']==task and result['input_receipt_sha256']==input_hash
 else:
  old=read(HERE/'COMPACT_MAIN_RESULTS.json');new=read(HERE/'COMPACT_REVISED_MAIN_RESULTS_20261021.json');census=read(HERE/'CONTROLLER_LEVEL_FAILURES.json');classes=read(HERE/'INSTANCE_CLASSES_REVISED.json')['instances']
  # Source lists are only needed by the two saved-raw hover tasks.
  ss=read(HERE/'MIGRATION_RESULT_SOURCES.json') if task['kind'].endswith('_hover') else [];ns=read(HERE/'REVISED_MIGRATION_RESULT_SOURCES_20261021.json') if task['kind'].endswith('_hover') else []
  result=dict(task=task,input_receipt_sha256=input_hash,result=calculate(task,old,new,census,classes,ss,ns))
  write(target,result)
 seal(str(proof.relative_to(HERE)),[target,HERE/'PARALLEL_NUMERIC_INPUTS_20261023.receipt.json',HERE/'CORRECTION_PARALLEL_NUMERICS_READY_20261023.receipt.json'],bootstrap_iterations=5000,native_rng_unchanged=True,posthoc_only=True)
 print('Numeric task complete',key,task,flush=True)
if __name__=='__main__':worker(sys.argv[1])
