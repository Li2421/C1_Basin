import os,sys,json,copy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from diagnostics.ring_state_selection_transfer_v1.study import read,write,sha,OLD
from diagnostics.controller_q_diversity_v1 import experiment as old
from diagnostics.controller_response_families_v1 import study as parent
from diagnostics.controller_q_diversity_v1.runtime import FIELD
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical
from .runtime import make
HERE=Path(__file__).resolve().parent;EXP='exp_ring_state_selection_controller_family_v1'
NAMES=['rotate_negative','-0.5','isotropic','0.5','rotate_positive','flow'];KAPPA=[-1,-.5,0,.5,1,None]
def prepare():
 assert (HERE.parent/'C0_RESULTS.json').exists()
 source=read(OLD/'protocol.json');prior=read(ROOT/'diagnostics/controller_flow_endpoint_probe_v1/protocol.json');profiles={};states=copy.deepcopy(source['states'][:32]);ctrl=prior['controllers']['ring_exchange']
 for i,s in enumerate(states):s['study_split']='train';s['study_index']=i
 for name,k in zip(NAMES,KAPPA):
  if name=='flow':profiles[name]=source['controller'];continue
  if name in ['isotropic','rotate_positive','rotate_negative']:profiles[name]=ctrl['flow_'+name];continue
  v=copy.deepcopy(ctrl['flow_isotropic']['payload']);v.update(base_controller='continuous_flow_endpoint_affine_ff_v1',response_family='continuous_rotation',kappa=k,controller_source_sha256=sha(HERE/'runtime.py'),execution_callable=str(HERE/'runtime.py'),execution_callable_sha256=sha(HERE/'runtime.py'))
  profiles[name]=dict(controller_uid=uid('ctl',v),payload=v)
 # Archived endpoint programs must still match the code defining their identity.
 for f,h in prior['source_code_hashes'].items():assert sha(f)==h,(f,'changed historical runtime')
 p=dict(schema='ring_controller_selection_continuous_pilot_v1',states=states,controllers={'ring_exchange':profiles},eta=source['eta'],eta_indices=list(range(16)),seeds=list(range(8)),kappa=dict(zip(NAMES,KAPPA)),environments={'ring_exchange':source['environment']},runtime_sha256=sha(HERE/'runtime.py'),FF_operator_sha256=sha(FIELD/'poc/field_controller.py'),source_protocol_sha256=sha(OLD/'protocol.json'),C0_sha256=sha(HERE.parent/'C0_RESULTS.json'),scope='DEVELOPMENT controller family gate on original outcome-blind32 source states; no new independent confirmation',horizon=700,dt=.05)
 write(HERE/'protocol.json',p)
 with connect() as db,transaction(db):
  db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(EXP,'Ring controller family selection pilot',str(HERE),sha(HERE/'protocol.json'),sha(__file__),canonical(dict(cpu_cap=16,rollout_cpu_hours=24))))
  for ctl in profiles.values():
   v=ctl['payload'];previous=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(ctl['controller_uid'],)).fetchone()
   if previous is not None:continue
   db.execute('INSERT INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
 cases=[old.case(p,s,c,j,k) for s in states for c in NAMES for j in range(16) for k in range(8)];missing=[];used=[];unknown=0;groups={}
 with connect(True) as db:
  for r in cases:
   a=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
   if a is None:missing.append(r)
   else:
    assert not a['conflict_quarantined'];assert all(a[k]==r[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']);used.append(r['rollout_uid']);unknown+=int(a['numerical_failure'])
   groups.setdefault((r['state_uid'],r['eta_uid'],r['controller_uid']),[]).append(r['seed_key'])
 write(HERE/'cases.json',cases);write(HERE/'missing_cases.json',missing);write(HERE/'cached_rollout_uids.json',used)
 write(HERE/'planned_rollouts.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=k) for (s,e,c),k in groups.items()]))
 write(HERE/'exact_identity_preflight.json',dict(requested=len(cases),exact_reuse=len(used)-unknown,cached_unknown=unknown,genuinely_missing=len(missing),cases_sha256=sha(HERE/'cases.json')))
 times=[]
 for f in (ROOT/'diagnostics/controller_flow_endpoint_probe_v1/raw').glob('*.json'):
  r=read(f)
  if r['case']['scene']=='ring_exchange':times.append(r['elapsed_seconds'])
 mean=float(np.mean(times));cost=dict(historical_timing_n=len(times),historical_mean_seconds=mean,missing=len(missing),requested=len(cases),reused=len(used),predicted_cpu_hours=len(missing)*mean/3600,predicted_wall_hours_16_workers=len(missing)*mean/3600/16,cap_cpu_hours=24,cap_wall_hours=2,controller_only_change=True)
 cost['within_budget']=cost['predicted_cpu_hours']<24 and cost['predicted_wall_hours_16_workers']<2;write(HERE/'COST.json',cost);print(json.dumps(cost),flush=True)
def worker(index,workers):
 # Delegate exact archived rollout implementation; pure Q, no video requirement.
 parent.HERE=HERE;parent.EXPERIMENT=EXP;parent.EndpointMatchedField=make
 parent.worker(index,workers)
def merge():
 old.HERE=HERE;old.EXPERIMENT=EXP;old.merge()
def audit():
 import jax
 from diagnostics.controller_q_diversity_v1.runtime import native_runtime
 from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
 p=read(HERE/'protocol.json');base=native_runtime('ring_exchange');jax.config.update('jax_enable_x64',False);rows=[]
 for s in p['states'][:2]:
  env=base.make_env(s);key=key_for(s,0,0);expect=base.nominal(env,key)[1]
  # No physical simulation; safe static validation only.
  for name in NAMES[:-1]:
   rt=make(base,name);got=rt.nominal(env,key)[1];np.testing.assert_array_equal(got,expect)
   M=rt.matrix(env);R=rt.unprojected_response();rows.append(dict(state=s['state_uid'],controller=name,nominal_equal=True,response=R.tolist(),matrix=M.tolist(),stability_spectral_norm=float(np.linalg.norm(np.eye(len(M))-M/rt.steps,2))))
 write(HERE/'static_operator_audit.json',dict(all_endpoint_alias_verified=True,code_sha256=p['runtime_sha256'],physical_steps=0,rows=rows));print('static audit passed',flush=True)
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage');ap.add_argument('--index',type=int,default=0);ap.add_argument('--workers',type=int,default=16);a=ap.parse_args()
 if a.stage=='worker':worker(a.index,a.workers)
 else:globals()[a.stage]()
