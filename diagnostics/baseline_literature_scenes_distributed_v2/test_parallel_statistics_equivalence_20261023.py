"""Engineering equivalence of diagnostic reads and independent native calls."""
from experiment import *
from atomic_write import write
from reclassify_instances_20261021 import small_diagnosis
from statistics_correction_revised_20261021 import class_splits as serial_splits
from statistics_parallel_tasks_20261023 import calculate,task_manifest
from statistics import bootstrap as old_bootstrap
from statistics_header_reader_20261023 import bootstrap as new_bootstrap
import ast,numpy as np,time,collections
os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
started=time.monotonic();paths=sorted((HERE/'DIAGNOSTICS').glob('*.json'));paths=paths[::max(1,len(paths)//128)][:128];keys=('local','canonical_joint','actual_product_joint','grid');read_checks=0
for path in paths:
 data=path.read_bytes();a=json.loads(data);b=small_diagnosis(data)
 for key in keys:assert a.get(key)==b.get(key),(path,key)
 read_checks+=1
# The full numerical function definitions are byte-for-byte the native AST.
old=ast.parse((HERE/'statistics.py').read_text());new=ast.parse((HERE/'statistics_header_reader_20261023.py').read_text());names=('bootstrap','contrast','summarize','controller_census','crossfit_oracle')
for name in names:
 a=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name);b=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name);assert ast.dump(a)==ast.dump(b),name
rng=np.random.default_rng(2026102301);M=rng.integers(0,3,(3,4,8)).astype(float)/2;types=np.asarray(['a','a','b']);a,d=old_bootstrap(M,types);b,e=new_bootstrap(M,types);assert a.tobytes()==b.tobytes() and d.tobytes()==e.tobytes()
# Native paired inference sees identical records when a scene/layer is handed to a different process.
scene='lit_narrow_passage';controllers=[r for r in read(HERE/'CONTROLLER_MANIFEST.json') if r['controller'] in ('DM0','DM1')];assert len(controllers)==2
rows=[];classes=[]
for layer in ('central','distributed'):
 for ci,ctl in enumerate(controllers):
  classes.append(dict(scene=scene,layer=layer,controller=ctl['controller'],remove_in_revised_exclusion=ci==0,classification='controller级失败' if ci==0 else '安全层导致的全局失败'))
  for state in range(2):
   for method in ('safety','fixed_eta10','fixed_eta13','human_progress','learned_scene'):
    for seed in (0,1):
     success=int((state+seed+ci+len(method))%3!=0)
     rows.append(dict(scene=scene,layer=layer,controller=ctl['controller'],type=ctl['type_code'],N=3,state=state,seed=seed,method=method,success=success,upper=success,collision=0,timeout=1-success,unknown=0,deadlock_episode=1-success,wall_episode=0,deadlock_agents=1-success,wall_agents=0,canonical_joint_episode=0,actual_joint_episode=1-success,rollout_uid='engineering_only'))
serial=serial_splits(rows,scene,classes)
for layer in ('central','distributed'):
 task=dict(kind='split',scene=scene,semantics='revised',layer=layer);parallel=calculate(task,rows,rows,[],classes,[],[])
 assert canonical(serial[layer])==canonical(parallel),layer
result=dict(utc=utc(),passed=True,saved_diagnostic_header_reads_bitwise_fields_equal=read_checks,native_numeric_AST_identical=list(names),bootstrap5000_draw_arrays_bitwise_identical=True,serial_parallel_layer_gain_split_JSON_identical=True,engineering_fixture_only=True,planned_independent_numeric_tasks=len(task_manifest()),new_rollouts=0,new_training=0,seconds=time.monotonic()-started)
write(HERE/'PARALLEL_STATISTICS_EQUIVALENCE_20261023.json',result);print(json.dumps(result),flush=True)
