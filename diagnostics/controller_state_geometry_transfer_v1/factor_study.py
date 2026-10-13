"""Frozen crossed factor manifest, cache registration, workers and journal merge."""
from pathlib import Path
import json,hashlib,datetime,itertools,argparse,time
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity
from shared_rollout_db.src.cache_writer import append_journal
from diagnostics.controller_state_transfer_v1.study import read,write,freeze,sha
HERE=Path(__file__).resolve().parent/'factors';EXP='exp_controller_state_geometry_transfer_v1_factors'
WIDTHS=[.5,.65,.8];TOPO=['single','parallel'];COUNTS=[2,4];C=['isotropic','anisotropic'];E=[0,5,10,15]

def design():
 from .factor_runtime import config,state,Env
 root=HERE.parents[2];pkg=HERE.parent
 deps=[pkg/'factor_runtime.py',root/'bottleneck_family/environment.py',root/'bottleneck_family/scenario.py',root/'shared_control/basis_families.py',root/'shared_control/hard_projection.py',root/'diagnostics/controller_response_families_v1/runtime.py',root/'diagnostics/controller_q_diversity_v1/numpy_backend.py',Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004/poc/field_controller.py')]
 hashes={str(p):sha(p) for p in deps};arch=read(root/'diagnostics/controller_ff_factorial_repair_v1/protocol.json');cases=[];geos=[];states=[];controllers=[];validation=[]
 for gi,(top,n,w) in enumerate(itertools.product(TOPO,COUNTS,WIDTHS)):
  cfg=config(w,n,top);g={'geometry_index':gi,'topology':top,'N':n,'width':w,'config':cfg.to_dict(),'hashes':hashes};g['scenario_uid']=uid('scenario',g);g['env_fingerprint']=uid('env',g);geos.append(g)
  for ci,cn in enumerate(C):
   payload={'scenario_uid':g['scenario_uid'],'environment_sha256':g['env_fingerprint'],'flow_checkpoint_sha256':'NONE_ANALYTIC','controller_kind':'single_agent_visibility_navigation_endpoint_affine_FF','response_family':cn,'initializer_mean':0.,'initializer_scale':.15,'pseudo_steps':10,'horizon':850,'dt':.05,'hashes':hashes,'eta_basis':'orthoflow3','safety':'unchanged FixedSafetySet','success_semantics':'native swept-collision-free all-agent arrival by850','rng_root':2026100403,'jax_enable_x64':True}
   cid=uid('ctl',payload);controllers.append({'controller_uid':cid,'payload':payload})
   for si in range(16):
    ph=state(si,n);alias=f'csg_factor_state_{si:03d}';namespace=int(hashlib.sha256(alias.encode()).hexdigest()[:8],16);sid=uid('state',{'scenario':g['scenario_uid'],'physical':ph});s={'state_uid':sid,'geometry_index':gi,'state_index':si,'scenario_uid':g['scenario_uid'],'physical':ph,'uid':alias,'rng_namespace':namespace}
    if ci==0:
     env=Env(cfg,ph);paths=[env.instance.geometry.path(p,z) for p,z in zip(env.positions,env.goals)];assert all(p is not None for p in paths)
     states.append(s);validation.append({'geometry_index':gi,'state_index':si,'geometry_valid':True,'single_agent_paths':[p.tolist() for p in paths],'joint_feasibility':'not_yet_witnessed'})
    for ei in E:
     eid=eta_identity(arch['eta'][ei])[0]
     for sd in [0,1]:
      sk=canonical({'future_index':sd,'future_root':2026100403,'rng_namespace':namespace});rid=uid('roll',dict(state=sid,eta=eid,controller=cid,seed=sk))
      cases.append({'geometry_index':gi,'state_index':si,'state_uid':sid,'controller_uid':cid,'controller_index':ci,'controller_name':cn,'eta_index':ei,'eta':arch['eta'][ei],'eta_uid':eid,'seed':sd,'seed_key':sk,'rollout_uid':rid,'scenario_uid':g['scenario_uid']})
 p={'experiment_uid':EXP,'preregistration_sha256':sha(pkg/'ADDENDUM_DIAGNOSTICS_AND_FACTORS.md'),'hashes':hashes,'geometries':geos,'states':states,'controllers':controllers,'eta_indices':E,'eta':{str(e):arch['eta'][e] for e in E},'n_cases':len(cases),'train_states':list(range(8)),'test_states':list(range(8,16)),'video_exemption':'pure Q labels and critic study'}
 freeze(HERE/'protocol.json',p);freeze(HERE/'cases.json',cases);freeze(HERE/'geometry_validation.json',validation)
 freeze(HERE/'planned_rollouts.json',{'requests':[{'state_uid':c['state_uid'],'eta_uid':c['eta_uid'],'controller_uid':c['controller_uid'],'seed_keys':[c['seed_key']]} for c in cases]})
 with connect() as db,transaction(db):
  db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(EXP,'Independent width topology N horizon factor panel',str(HERE),sha(HERE/'protocol.json'),sha(pkg/'factor_runtime.py'),canonical({'new_labels':len(cases)})))
  for g in geos:db.execute('INSERT OR IGNORE INTO scenario(scenario_uid,name,code_config_fingerprint,metadata_json) VALUES(?,?,?,?)',(g['scenario_uid'],f"csg_factor_{g['topology']}_N{g['N']}_w{g['width']}",g['env_fingerprint'],canonical(g)))
  for s in states:
   db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',(s['state_uid'],s['scenario_uid'],s['uid'],uid('content',s['physical']),canonical(s['physical']),canonical(geos[s['geometry_index']]),canonical(s),'CONTENT_EXACT'))
  for c in controllers:
   a=c['payload'];db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(c['controller_uid'],a['scenario_uid'],'NONE_ANALYTIC',hashes[str(root/'shared_control/basis_families.py')],uid('safe',hashes),'850','.05',a['success_semantics'],'true-t0; common navigation program',canonical({'root':2026100403}),canonical(a),'EXACT_PROFILE'))
  for e in E:
   eid,v,h=eta_identity(arch['eta'][e]);db.execute('INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex) VALUES(?,?,?,?,?)',(eid,*v,h))
 print(json.dumps({'cases':len(cases),'configurations':len(geos),'validated_starts':len(states)}))

def worker(index,workers):
 assert read(HERE.parent/'phase2_complete.json')['completed'], 'Phase2 must be audited first'
 import jax
 from .factor_runtime import config,Base,Field,rollout
 jax.config.update('jax_enable_x64',True);p=read(HERE/'protocol.json')
 for path,h in p['hashes'].items():assert sha(path)==h,('code drift',path)
 pf=read(HERE/'cache_preflight.json');assert pf['summary']['ambiguous']==0;missing={(r['state_uid'],r['eta_uid'],r['controller_uid'],k) for r in pf['details'] for k in r['missing_seeds']}
 sts={s['state_uid']:s for s in p['states']};rts={};buf=[];started=time.monotonic();n=0
 for c in read(HERE/'cases.json')[index::workers]:
  dest=HERE/'raw'/(c['rollout_uid']+'.json')
  if dest.exists() or (c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key']) not in missing:continue
  key=(c['geometry_index'],c['controller_index']);g=p['geometries'][key[0]]
  if key not in rts:rts[key]=Field(Base(config(g['width'],g['N'],g['topology'])),c['controller_name'])
  trace=HERE/'traces'/(c['rollout_uid']+'.npz');trace.parent.mkdir(parents=True,exist_ok=True)
  r=rollout(rts[key],sts[c['state_uid']],c['eta'],c['seed'],trace);row=dict(schema='csg_factor_seed_v1',experiment_uid=EXP,protocol_sha256=sha(HERE/'protocol.json'),case=c,trace_path=str(trace),**r);write(dest,row);buf.append(row);n+=1
  if len(buf)>=8:append_journal(buf,EXP,f'main_{index}');buf=[]
 if buf:append_journal(buf,EXP,f'main_{index}')
 write(HERE/'workers'/f'{index}.json',{'completed':True,'executed':n,'seconds':time.monotonic()-started})

def merge():
 from . import revision_study as s
 s.HERE=HERE;s.EXPERIMENT=EXP;s.merge()
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['design','worker','merge']);a.add_argument('--index',type=int,default=0);a.add_argument('--workers',type=int,default=16);x=a.parse_args();{'design':design,'merge':merge}.get(x.action,lambda:worker(x.index,x.workers))()
