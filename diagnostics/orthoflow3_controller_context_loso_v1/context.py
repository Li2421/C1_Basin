"""Short physical response descriptors, not success labels or full continuations.

All candidates fixed before source validation or new target predictions. Probe
noise is independent of certification seeds. No scene/checkpoint ID in output.
"""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,csv,hashlib,json,pickle,sys,time,inspect
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
TOY=Path('/home/zhihan/research/02_C1_Toy_GiveWay')
sys.path[:0]=[str(TOY),str(ROOT)]
import jax
import jax.numpy as jnp
# Do not mutate process-wide Flow/RNG precision merely by importing this
# module. Runtime.__init__ selects the native precision explicitly per scene.
OLD=ROOT/'diagnostics/orthoflow3_loso_partial_count_v1'
SWAP=ROOT/'diagnostics/orthoflow3_controller_conditioning_probe_v1'
FOLDS={'toy':'toy_giveway','db':'double_bottleneck','four':'four_way_intersection','ring':'ring_exchange'}
NAMES=['goal_speed_mean','goal_speed_min','lateral_speed_mean','lateral_speed_std',
       'pair_closing_mean','pair_closing_max','speed_mean','safety_removal_mean',
       'correction_execution_mean','action_change_mean']
PROTOCOL={'candidates':{'C0':'original physical h and eta',
 'C1':'eta-independent nominal Flow+safety, 3 steps, summaries of steps 1 and 2',
 'C3':'eta-conditioned Flow+safety+OrthoFlow3, 3 steps, summaries of steps 1 and 2'},
 'steps':3,'seconds':.15,'probe_rng':2026100319,'features':NAMES,
 'noise':'one fixed independent probe stream shared across states/etas/controllers; not Q16 seeds',
 'flow_sampler_precision':{'toy_giveway':'float64 (original controller swap runtime)',
  'double_bottleneck':'float32','four_way_intersection':'float32','ring_exchange':'float32'},
 'normalization':'source TRAIN mean/std only; scale floor 0.05; no target clipping',
 'models':'same entity encoder/trunk widths; context appended; pure observed-count NLL',
 'seeds':[17,23,41],'selection':'source-only scene-balanced validation likelihood',
 'generator':'frozen','new_full_continuations':0,
 'microprobe_semantics':'derived input only, never inserted as rollout success evidence',
 'no_controller_or_scene_id_input':True,'target_labels_for_design':False,
 'historical_target_knowledge':'inherited schema and task motivated by previously reported LOSO failures'}
def load(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Runtime:
 def __init__(self,scene,alternate=False):
  self.scene=scene;self.alternate=alternate
  # random.normal's default dtype changes the actual MACFlow noise sequence.
  # Match the native evaluator, not a global precision preference. Physical
  # states and safety projection remain numpy float64 in every scenario.
  jax.config.update('jax_enable_x64',scene=='toy_giveway')
  from shared_control.basis_families import get_basis_family
  self.basis=get_basis_family('orthoflow3')
  if scene=='toy_giveway':
   from single_integrator.evaluate import load_policy
   from single_integrator.environment import Config,GiveWayEnv
   from single_integrator.cbf import CBFConfig
   p=load(SWAP/'protocol.json');self.cfg=Config(**p['environment']);self.cbf=CBFConfig(**p['cbf'])
   self.path=Path(p['flow_paths'][str(int(alternate))]);policy,_=load_policy(self.path)
   self.sample=jax.jit(lambda o,k:policy.sample_actions(o[None],seed=k)[0]);self.make=lambda:GiveWayEnv(self.cfg)
  elif scene=='double_bottleneck':
   from double_bottleneck.environment import Config,DoubleBottleneckEnv
   from double_bottleneck.flowbc_4a_agent import load_checkpoint
   from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
   self.path=ROOT/'diagnostics/double_bottleneck_recovery_density_final/model/ckpt_selected.pkl'
   raw=pickle.load(self.path.open('rb'));policy,_=load_checkpoint(self.path,raw['config']['environment_fingerprint'])
   self.cfg=Config();self.make=lambda:DoubleBottleneckEnv(self.cfg);self.projector=CertifiedHardSafetyFilter()
   self.sample=jax.jit(lambda o,k:policy.sample_actions(o[None],k)[0])
  else:
   from new_benchmark_common.safety_eta3 import SCENARIOS,ScenarioRuntime
   from new_benchmark_common.macflow import load_checkpoint
   from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
   from shared_control.hard_projection import HardProjectionConfig
   spec=SCENARIOS[scene];man=load(spec['dataset']/'manifest.json');self.path=spec['checkpoint']
   rt=ScenarioRuntime.__new__(ScenarioRuntime);rt.kind=spec['kind'];rt.agent,_=load_checkpoint(self.path,expected_environment_fingerprint=man['environment_fingerprint'])
   rt.cbf=HardProjectionConfig();rt.projector=CertifiedHardSafetyFilter(rt.cbf)
   if scene=='ring_exchange':
    from ring_exchange.environment import LocalFrameConfig,RingExchangeEnv
    self.cfg=LocalFrameConfig(**man['scenario_config']);self.make=lambda:RingExchangeEnv(self.cfg)
   else:
    from four_way_intersection.environment import Config,FourWayIntersectionEnv
    self.cfg=Config(**man['scenario_config']);self.make=lambda:FourWayIntersectionEnv(self.cfg)
   rt.config=self.cfg;self.rt=rt
  self.checkpoint_sha=sha(self.path)
 def reset(self,p):
  env=self.make();pos=np.asarray(p['positions']);vel=np.asarray(p['velocities']);goals=np.asarray(p['goals'])
  if self.scene=='ring_exchange':env.reset(pos,velocities=vel,goals=goals)
  elif self.scene=='four_way_intersection':env.reset(pos,vel);np.testing.assert_allclose(env.goals,goals,atol=1e-7)
  else:
   env.reset(pos);env.velocities[:]=vel
   # DB's h reconstructs float32 goals; these are not authoritative simulator goals.
   np.testing.assert_allclose(env.goals,goals,atol=1e-6)
  env.step_count=int(round((p['max_seconds']-p['remaining_seconds'])/p['dt']))
  return env
 def flow(self,env,key):
  if hasattr(self,'rt'):return self.rt.flow_world(env,key)
  u=np.asarray(self.sample(jnp.asarray(env.observation(),jnp.float32),key),float)
  return u*np.minimum(1.,self.cfg.max_speed/np.maximum(np.linalg.norm(u,axis=-1,keepdims=True),1e-30))
 def project(self,env,u):
  if hasattr(self,'rt'):return self.rt.project(env,u)[0]
  if self.scene=='toy_giveway':
   from single_integrator.cbf import barrier_constraints
   from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
   a,b,_=barrier_constraints(env.snapshot(),self.cbf)
   return project_velocity_with_retry(u,a,b,self.cfg.max_speed,self.cbf)[0]
  return np.asarray(self.projector(env.snapshot(),u).velocity,float)
 def probe(self,p,eta):
  env=self.reset(p);result=[];v=self.cfg.max_speed;old=np.asarray(p['velocities'])
  for step in range(3):
   key=jax.random.fold_in(jax.random.PRNGKey(PROTOCOL['probe_rng']),step)
   flow=np.asarray(p['flow']) if step==0 and p['flow_committed'] else self.flow(env,key)
   safe=self.project(env,flow)
   u=safe if not np.any(eta) else self.project(env,safe+self.basis.compute(env.positions,env.goals,safe,v).correction(eta))
   goal=env.goals-env.positions;goal/=np.maximum(np.linalg.norm(goal,axis=-1,keepdims=True),1e-12)
   g=(u*goal).sum(-1)/v;l=(u[:,1]*goal[:,0]-u[:,0]*goal[:,1])/v
   close=[]
   for i in range(len(u)):
    for j in range(i):
     d=env.positions[i]-env.positions[j];close.append(-np.dot(u[i]-u[j],d)/max(np.linalg.norm(d)*v,1e-12))
   row=[g.mean(),g.min(),l.mean(),l.std(),np.mean(close),max(close),
        np.linalg.norm(u,axis=-1).mean()/v,np.linalg.norm(safe-flow,axis=-1).mean()/v,
        np.linalg.norm(u-safe,axis=-1).mean()/v,np.linalg.norm(u-old,axis=-1).mean()/v]
   if step:result.append(row)
   old=u.copy();env.step(u)
  x=np.mean(result,axis=0);assert np.isfinite(x).all();return x

def cached(rt,state,eta,kind):
 payload={'physical':state['physical'],'checkpoint_sha256':rt.checkpoint_sha,
          'scenario':rt.scene,'eta':np.asarray(eta,float).tolist(),'protocol':PROTOCOL,
          'runtime_sha':hashlib.sha256(inspect.getsource(Runtime).encode()).hexdigest()}
 key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();path=OUT/'response_cache'/f'{key}.json'
 if path.exists():return load(path)
 t=time.perf_counter()
 try:r={'context':rt.probe(state['physical'],np.asarray(eta,float)).tolist(),'valid':True}
 except Exception as e:r={'context':None,'valid':False,'error':f'{type(e).__name__}: {e}'}
 r.update(seconds=time.perf_counter()-t,key=key,state_uid=state['state_uid'],kind=kind,eta=payload['eta'],checkpoint_sha=rt.checkpoint_sha)
 dump(path,r);return r

def swap():
 from single_integrator.environment import bounded_nominal
 pairs=load(SWAP/'pair_manifest.json');rt=[Runtime('toy_giveway',bool(i)) for i in (0,1)]
 h=np.load(ROOT/'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1/cohort_features.npz')['h_raw']
 result=[]
 for row in pairs:
  env=rt[0].make();env.reset(np.asarray(row['initial_positions']))
  flow=rt[0].flow(env,jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(42),row['rollout_id']),0))
  p=dict(next(s['physical'] for s in load(OLD/'states.json') if s['scenario']=='toy_giveway'))
  p.update(positions=env.positions.tolist(),velocities=env.velocities.tolist(),goals=env.goals.tolist())
  p['flow']=flow.tolist();state={'state_uid':row['state_uid'],'physical':p}
  for kind,eta in [('C1',np.zeros(3)),('C3',row['eta'])]:
   a,b=[cached(r,state,eta,kind) for r in rt]
   result.append({**row,'candidate':kind,'baseline_feature_distance':0.,'valid':a['valid'] and b['valid'],
    'contexts':[a['context'],b['context']],'context_distance':float(np.linalg.norm(np.array(a['context'])-b['context'])) if a['valid'] and b['valid'] else None,
    'errors':[a.get('error'),b.get('error')]})
 dump(OUT/'swap_contexts.json',result)
 print(json.dumps({'rows':len(result),'valid':sum(r['valid'] for r in result),'separated':sum((r['context_distance'] or 0)>1e-6 for r in result)}))

def build(scene,shard=0,shards=1):
 import pyarrow.parquet as pq
 states=load(OLD/'states.json');pairs=pq.read_table(OLD/'all_pairs.parquet').to_pylist();rt=Runtime(scene)
 ss=[(i,s) for i,s in enumerate(states) if s['scenario']==scene]
 done=[];start=time.time()
 for num,(i,s) in enumerate(ss):
  if num%shards!=shard:continue
  a=cached(rt,s,np.zeros(3),'C1');done.append({'state_index':i,'candidate':'C1',**a})
  for j,r in enumerate(pairs):
   if r['state_index']==i:done.append({'pair_index':j,'state_index':i,'candidate':'C3',**cached(rt,s,r['eta'],'C3')})
  dump(OUT/'features'/f'{scene}_{shard}.json',done)
  print(json.dumps({'scene':scene,'shard':shard,'states_done':1+num//shards,'rows':len(done),'seconds':time.time()-start}),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['init','swap','build']);ap.add_argument('--scene',choices=list(FOLDS.values()));ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);a=ap.parse_args()
 if a.action=='init':
  dump(OUT/'protocol.json',PROTOCOL);dump(OUT/'working_state.json',{'stage':'counterexample_context','new_full_continuations':0})
 elif a.action=='swap':swap()
 else:build(a.scene,a.shard,a.shards)
