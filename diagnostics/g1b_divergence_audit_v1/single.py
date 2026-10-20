import os,sys,json,hashlib,datetime,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['JAX_PLATFORMS']='cpu';(HERE/'tmp').mkdir(exist_ok=True);os.environ['TMPDIR']=str(HERE/'tmp')
sys.path[:0]=[str(ROOT/'diagnostics/four_controller_axis_v2'),str(ROOT)]
import numpy as np,jax
from adapter import historical_state,intervention_runtime
states64=[dict(stage='after_adapter_import',x64=bool(jax.config.x64_enabled))]
from new_benchmark_common import basin_dataset_v1 as bd
from new_benchmark_common.safety_eta3 import ScenarioRuntime
oldpath=ROOT/'diagnostics/orthoflow3_controller_intervention_generalization_v1/intervention.py';spec=importlib.util.spec_from_file_location('historical_intervention',oldpath);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);states64.append(dict(stage='after_old_import',x64=bool(jax.config.x64_enabled)))
r19=ROOT/'diagnostics/four_controller_axis_v2';gate=json.loads((r19/'G1B_REPORT.json').read_text());uid=next(x['rollout_uid'] for x in gate['results'] if not x['passed']);r=next(x for x in json.loads((r19/'G1A_SELECTION.json').read_text())['records'] if x['rollout_uid']==uid);index=r['original']['initial_state_index'];states=json.loads((ROOT/'diagnostics/orthoflow3_loso_partial_count_v1/states.json').read_text());state=historical_state(states[index],index)
class Isolated(bd.TrainingRuntime):
 def _register_training_identities(self):pass
mode=sys.argv[1]
if mode.startswith('old'):
 out=ROOT/'diagnostics/orthoflow3_controller_intervention_generalization_v1'/('balanced_expansion' if r['experiment_uid'].endswith('balanced_v1') else 'second_variant');old.OUT=out;profile=json.loads((out/'protocol.json').read_text())['profiles']['four_way_intersection'];orig=bd.TrainingRuntime;bd.TrainingRuntime=Isolated
 try:rt,by,reset=old._runtime('four_way_intersection',profile)
 finally:bd.TrainingRuntime=orig
 state=by[r['state_uid']]
else:rt,reset=intervention_runtime(Path(r['checkpoint_path']))
states64.append(dict(stage='after_runtime_construct',x64=bool(jax.config.x64_enabled)))
if mode in ('old32','new32'):jax.config.update('jax_enable_x64',False)
if mode=='new64':jax.config.update('jax_enable_x64',True)
reset();states64.append(dict(stage='before_rollout',x64=bool(jax.config.x64_enabled)));flow=[];trace=[];original_flow=ScenarioRuntime.flow_world
hexarr=lambda a:dict(dtype=str(np.asarray(a).dtype),shape=list(np.asarray(a).shape),hex=np.asarray(a).tobytes().hex())
def inspect_flow(self,env,key):
 obs=self.observation(env);out=original_flow(self,env,key)
 if not flow:
  config=self.agent.config;leaves=jax.tree_util.tree_leaves(self.agent.network.params);flow.append(dict(x64=bool(jax.config.x64_enabled),observation=hexarr(obs),key=hexarr(key),obs_mean=hexarr(config['obs_mean']),obs_scale=hexarr(config['obs_scale']),act_mean=hexarr(config['act_mean']),act_scale=hexarr(config['act_scale']),normalized_observation=hexarr((np.asarray(obs).reshape(1,-1)-np.asarray(config['obs_mean']))/np.asarray(config['obs_scale'])),parameter_sha256=hashlib.sha256(b''.join(np.asarray(x).tobytes() for x in leaves)).hexdigest(),flow_output=hexarr(out)))
 return out
ScenarioRuntime.flow_world=inspect_flow
make=rt.make_env
def instrument():
 env=make();step=env.step
 def wrapped(a):
  before=dict(position=hexarr(env.positions),velocity=hexarr(env.velocities),action=hexarr(a));ret=step(a);before.update(position_after=hexarr(env.positions),velocity_after=hexarr(env.velocities));trace.append(before);return ret
 env.step=wrapped;return env
rt.make_env=instrument
result=rt.rollout(state,np.asarray(r['original']['eta']),r['original']['future_index'],'orthoflow3');out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'mode':mode,'case_uid':uid,'states_x64':states64,'first_flow':flow,'first_action':trace[0]['action'],'trajectory_sha256':hashlib.sha256(json.dumps(trace,sort_keys=True).encode()).hexdigest(),'steps':len(trace),'outcome':result['outcome'],'result':result,'checkpoint_sha256':r['checkpoint_sha256'],'safety_config':rt.cbf.__dict__}
(HERE/(mode+'.json')).write_text(json.dumps(out,indent=2,default=lambda x:np.asarray(x).tolist())+'\n');(HERE/(mode+'_trace.json')).write_text(json.dumps(trace)+'\n');print(mode,out['outcome'],len(trace),states64,flush=True)
