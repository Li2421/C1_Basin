#!/usr/bin/env python3
"""Provenance-aware offline inventory and deterministic acquisition design."""
import csv, json, hashlib, sys, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from scipy.spatial.distance import cdist
ROOT=Path('/home/zhihan/research/Basin_C1'); D=ROOT/'diagnostics'; HERE=Path(__file__).parent
POINT=D/'orthoflow3_true_t0_point_learning_v1'; OLD=D/'orthoflow3_analytic_basin_margin_learning_v1'
AFF=np.array([.875,0,.375]); SCALE=np.array([.75,1,.75])
SHA='51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38'
STATES={r['state_id']:r for r in json.load(open(POINT/'final_source_split.json'))['states']}
HS=np.unique(np.array(json.load(open(D/'orthoflow3_t0_multiball_basin_learning_v1/geometry_constants.json'))['halfspaces']),axis=0)
def read(p): return list(csv.DictReader(open(p)))
def write(name, rows, fields=None):
 p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True)
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(name,x):
 p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def key(v):return np.asarray(v,dtype='<f8').tobytes().hex()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eta(r):return np.array([float(r['eta'+str(i)]) for i in (1,2,3)])
def inside(z):return np.all(np.atleast_2d(z)@HS[:,:3].T+HS[:,3]<=1e-10,axis=1)
def inventory():
 assert sha(D/'double_bottleneck_eta_basis_redesign/tools/bases.py')==SHA
 frozen_config=json.load(open(D/'gphi_fixed_d_eta_predictor_v1/integrity_audit.json'))
 semantic_files=[D/'double_bottleneck_eta_basis_redesign/tools/bases.py',ROOT/'shared_control/basis_families.py',ROOT/'single_integrator/environment.py',ROOT/'single_integrator/cbf.py',
   D/'success_basin_multimodality/exact_projector.py',D/'gphi_training_dataset_v2/finalize_dataset.py',D/'gphi_training_dataset_startup_complete_v1/startup_feature_builder.py',
   D/'orthoflow3_conservative_basin_ball_pilot6_v1/pilot6.py',OLD/'run_rollout_shard.py',
   Path('/home/zhihan/research/02_C1_Toy_GiveWay/baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')]
 semantics={'scenario':'ToyGiveWay_2A','environment':frozen_config['environment'],'cbf':frozen_config['cbf'],
   'source_sha256':{str(p):sha(p) for p in semantic_files},'basis_sha256':SHA,'basis_scale':3.303687238760696,
   'eta':'fixed true_t0 entire continuation','projections':'exact first and second frozen authority','future_root_seed':2026092811,
   'future_rng':'fold_in(fold_in(fold_in(PRNGKey(root),sha256(state_id)[:8]),future_index),absolute_step)',
   'current_rng':'fold_in(fold_in(PRNGKey(state.flow_seed),state.rng_namespace),0); batch1 first action',
   'monitor_history':'restore_full frozen t0 state archive, not reset midtrajectory','hard_label':'64 distinct future indices0..63; >=63success; numerical errors excluded and stop'}
 semhash=hashlib.sha256(json.dumps(semantics,sort_keys=True).encode()).hexdigest()
 if (HERE/'conditioning_semantics.json').exists():assert json.load(open(HERE/'conditioning_semantics.json'))['controller_semantics_hash']==semhash,'Frozen control stack changed during audit'
 dump('conditioning_semantics.json',dict(semantics,controller_semantics_hash=semhash))
 store={};sources=[];conflicts=[];raw=defaultdict(dict);files=[];excluded_execution=[]
 accepted_h={sid:{s['feature_sha256']} for sid,s in STATES.items()}
 replay_file=HERE/'conditioning_replay_diagnosis.json'
 if replay_file.exists():
  replay=json.load(open(replay_file));assert replay['all_within_frozen_protocol_tolerance'] and replay['physical_steps']==0
  for r in replay['records']:
   assert r['canonical_h']==STATES[r['state_id']]['feature_sha256']
   accepted_h[r['state_id']].update(t['h_sha256'] for t in r['checks'] if t['feature_max_error']<=1e-10)
  dump('conditioning_alias_audit.json',{'status':'RESOLVED_NUMERICAL_BATCH_REPLAY_ALIASES','frozen_tolerance':1e-10,'new_tolerance_introduced':False,
   'max_feature_difference':max(t['feature_max_error'] for r in replay['records'] for t in r['checks']),
   'max_current_action_difference':max(r['action_difference_32_1'] for r in replay['records']),
   'identity':'same physical archive and RNG xi0, frozen numerical replay tolerance; not bit-identical feature representation',
   'accepted_only_exact_reproduced_hashes':{sid:sorted(h) for sid,h in accepted_h.items()},'eta_dedup_remains_exact_float64':True})
 dense=D/'orthoflow3_t0_pact_training_readiness_v1/exact_q64_manifest.csv'
 for r in read(dense):
  if r['state_id'] not in STATES or r['Q64_available'].lower()!='true' or int(r['trials'])!=64:continue
  v=eta(r);k=(r['state_id'],key(v));store[k]=dict(r,eta_key=k[1],sources='dense_archive:'+r['sources'],phases=r['phases'])
 sources.append({'path':str(dense),'sha256':sha(dense),'acceptance':'prior provenance-audited dense exact-Q64 aggregate'})
 files+=list(POINT.glob('state_runs/*/raw/pilot_rollouts.jsonl'))+list(POINT.glob('raw/*/shard*.jsonl'))
 files+=list((D/'orthoflow3_t0_multiball_basin_learning_v1').glob('stage_a/*/raw/pilot_rollouts.jsonl'))
 files+=list((D/'orthoflow3_t0_basin_structure_v1').glob('anchor_runs/*/raw/pilot_rollouts.jsonl'))
 files+=list((D/'orthoflow3_t0_basin_completion_v1').glob('raw/*/raw/pilot_rollouts.jsonl'))
 cont=D/'orthoflow3_t0_eta_continuity_cross_transfer_v1'
 files+=list(cont.glob('raw/shard*.jsonl'))+list(cont.glob('raw/*/shard*.jsonl'))
 files+=list(OLD.glob('raw/*/shard*.jsonl'))+list(HERE.glob('raw/*/shard*.jsonl'))
 for p in sorted(set(files)):
  sources.append({'path':str(p),'sha256':sha(p),'acceptance':'frozen t0 runner; individual h identity checked'})
  for line in open(p):
   if not line.strip():continue
   r=json.loads(line);sid=r.get('state_id');fi=r.get('future_index')
   if sid not in STATES or fi is None or not 0<=int(fi)<64:continue
   s=STATES[sid];assert r['h_conditioning_identifier']==s['h_conditioning_identifier'],(p,sid,'conditioning')
   assert r['source_group']==s['source_group'],(p,sid,'source')
   if r.get('execution_error'):
    excluded_execution.append({'path':str(p),'state_id':sid,'eta':r['eta'],'future_index':fi,'execution_error':r['execution_error']})
    assert HERE not in p.parents,(p,sid,'current-stage execution error')
    continue
   assert r['first_step']['feature_sha256'] in accepted_h[sid],(p,sid,'unexplained h hash')
   k=(sid,key(r['eta']));fi=int(fi)
   if fi in raw[k] and raw[k][fi]['success']!=r['success']:conflicts.append({'state_id':sid,'eta_key':k[1],'future_index':fi,'path':str(p)})
   r['_path']=str(p);raw[k][fi]=r
 for (sid,ek),g in raw.items():
  if len(g)!=64:continue
  rr=[g[i] for i in range(64)];v=rr[0]['eta'];su=sum(x['success'] for x in rr);k=(sid,ek)
  prev=store.get(k)
  if prev and int(prev['successes'])!=su:conflicts.append({'state_id':sid,'eta_key':ek,'cached_success':prev['successes'],'raw_success':su})
  store[k]={'state_id':sid,'eta_key':ek,**{f'eta{i+1}':v[i] for i in range(3)},'successes':su,'trials':64,
   'deadlock':sum(x['outcome'] in ('deadlock','safe_deadlock','strict_deadlock') for x in rr),'timeout':sum(x['outcome']=='timeout' for x in rr),
   'collision':sum(x['outcome']=='collision' for x in rr),'sources':';'.join(sorted({x['_path'] for x in rr} | ({prev['sources']} if prev else set()))),
   'phases':';'.join(sorted({x['phase'] for x in rr})), 'physical_steps':sum(x['continuation_steps'] for x in rr)}
 dump('integrity_conflicts.json',conflicts)
 dump('excluded_execution_records.json',excluded_execution)
 if conflicts:raise RuntimeError('Exact cache conflicts; stop before new execution')
 out=[]
 for (sid,ek),r in sorted(store.items()):
  s=STATES[sid];r.update(scenario='ToyGiveWay_2A',phase='true_t0',source_group=s['source_group'],h=s['feature_sha256'],conditioning=s['h_conditioning_identifier'],
    controller_semantics_hash=semhash,future_root_seed=2026092811,split=s['split'],Q64=int(r['successes'])/64,B63=int(r['successes'])>=63,
    eta_key=ek,geometry_role='heldout' if int(hashlib.sha256((sid+ek+'geometry-v1').encode()).hexdigest()[:8],16)%4==0 else 'fit')
  out.append(r)
 # A second genuinely compatible scenario is kept in a separate conditioning namespace.
 dbgroups=defaultdict(dict);dbhs=defaultdict(set);dbfiles=sorted((HERE/'db_raw').glob('*.jsonl'))
 if dbfiles:
  resolution=HERE/'execution_anomaly_resolution.json'
  quarantined=json.load(open(resolution))['approved_quarantine'] if resolution.exists() else []
  dbstates={s['state_id']:s for s in json.load(open(HERE/'double_bottleneck_state_panel.json'))}
  dbc=json.load(open(HERE/'double_bottleneck_cost_estimate.json'));dbsem={'frozen_hashes':dbc['frozen_hashes'],'basis_sha256':SHA,'basis_scale':3.303687238760696,
   'current_root':2026092906,'future_root':2026092907,'runner_sha256':sha(HERE/'run_db.py'),'projection':'CertifiedHardSafetyFilter plus identical certified retry',
   'eta':'fixed true_t0 all episode','h_identity':'initial observation and current Flow action float64 bytes; u_safe deterministically derived'}
  dbsemhash=hashlib.sha256(json.dumps(dbsem,sort_keys=True).encode()).hexdigest();dump('double_bottleneck_conditioning_semantics.json',dict(dbsem,controller_semantics_hash=dbsemhash))
  for p in dbfiles:
   snapshot=p.read_bytes();tail_bytes=0
   if snapshot and not snapshot.endswith(b'\n'):
    last=snapshot.rfind(b'\n')+1;tail_bytes=len(snapshot)-last;snapshot=snapshot[:last]
   sources.append({'path':str(p),'sha256':hashlib.sha256(snapshot).hexdigest(),'snapshot_bytes':len(snapshot),'incomplete_trailing_bytes_excluded':tail_bytes,'acceptance':'immutable complete-line snapshot; fixed xi0 DB_Q64 namespace; never merged with old varying-xi0 seed trials'})
   for line in snapshot.splitlines():
    r=json.loads(line);sid=r['state_id']
    if not r['scientific_outcome_valid']:
     exact=dict(state_id=sid,eta_key=key(r['eta']),future_index=r['future_index'],outcome=r['outcome'])
     assert exact in quarantined,(p,'unreviewed execution failure')
     excluded_execution.append(dict(exact,path=str(p),execution_error=r['outcome'],physical_steps_before_error=r.get('episode_steps_before_solver_failure')))
     continue
    assert r['future_root_seed']==2026092907 and r['current_root_seed']==2026092906
    dbhs[sid].add(r['h_conditioning_identifier']);k=(sid,key(r['eta']));fi=int(r['future_index'])
    if fi in dbgroups[k]:assert dbgroups[k][fi]['success']==r['success'],(sid,k,'DB exact conflict')
    dbgroups[k][fi]=r
  assert all(len(h)==1 for h in dbhs.values()),'DB current xi0/h changed across futures or eta'
  for (sid,ek),g in sorted(dbgroups.items()):
   if set(g)!=set(range(64)):continue
   rr=[g[i] for i in range(64)];s=dbstates[sid];v=rr[0]['eta'];success=sum(r['success'] for r in rr)
   out.append(dict(state_id=sid,scenario='DoubleBottleneck_4A',phase='true_t0',source_group=s['source_group'],h=next(iter(dbhs[sid])),conditioning=next(iter(dbhs[sid])),
    controller_semantics_hash=dbsemhash,future_root_seed=2026092907,split='geometry_discovery',eta_key=ek,eta1=v[0],eta2=v[1],eta3=v[2],successes=success,trials=64,Q64=success/64,B63=success>=63,
    deadlock=sum(r['outcome']=='strict_deadlock' for r in rr),timeout=sum(r['outcome']=='timeout' for r in rr),collision=sum('collision' in r['outcome'] for r in rr),sources=';'.join(str(p) for p in dbfiles),
    phases=';'.join(sorted({r['phase'] for r in rr})),geometry_role='heldout' if any(r['phase']=='db_independent_interpolation_r3' for r in rr) or int(hashlib.sha256((sid+ek+'geometry-v1').encode()).hexdigest()[:8],16)%4==0 else 'fit',physical_steps=sum(r['episode_steps'] for r in rr)))
 from evidence_roles import validation_memberships
 roles=validation_memberships();annotated=[]
 for r in out:
  r['in_E_bridge']=bool(inside((eta(r)-AFF)/SCALE)[0])
  r['raw_phases']=r['phases'];v=roles.get((r['state_id'],r['eta_key']),set());r['retained_validation_batches']=';'.join(sorted(v))
  if v:
   r['phases']=';'.join(sorted(set(r['phases'].split(';'))|v))
   if r['phases']!=r['raw_phases']:annotated.append(dict(state_id=r['state_id'],eta_key=r['eta_key'],raw_phases=r['raw_phases'],canonical_validation_batches=r['retained_validation_batches']))
 write('acquisition_role_corrections.csv',annotated,['state_id','eta_key','raw_phases','canonical_validation_batches'])
 fields=['state_id','scenario','phase','source_group','h','conditioning','controller_semantics_hash','future_root_seed','split','eta_key','eta1','eta2','eta3','successes','trials','Q64','B63','deadlock','timeout','collision','sources','phases','geometry_role','physical_steps','raw_phases','retained_validation_batches','in_E_bridge']
 write('out_of_domain_historical_evidence.csv',[r for r in out if not r['in_E_bridge']],fields)
 write('exact_q64_inventory.csv',out,fields);dump('cache_provenance.json',sources)
 dump('excluded_execution_records.json',excluded_execution)
 lower=[{'state_id':sid,'eta_key':ek,'trials':len(g),'successes':sum(x['success'] for x in g.values()),'sources':';'.join(sorted({x['_path'] for x in g.values()}))} for (sid,ek),g in raw.items() if len(g)<64]
 write('lower_seed_inventory.csv',lower,['state_id','eta_key','trials','successes','sources'])
 with open(HERE/'partial_continuation_cache.jsonl','w') as f:
  for k,g in raw.items():
   if len(g)<64:
    for r in g.values():f.write(json.dumps(r)+'\n')
 dump('inventory_summary.json',{'exact_state_eta':len(out),'B63':sum(r['B63'] for r in out),'within_E_bridge':sum(r['in_E_bridge'] for r in out),'B63_within_E_bridge':sum(r['B63'] and r['in_E_bridge'] for r in out),'out_of_domain_historical_records':sum(not r['in_E_bridge'] for r in out),'states':len({r['state_id'] for r in out}),'lower_seed_state_eta':len(lower),'source_files':len(sources),'conflicts':0,
  'deadlock_fix':'raw safe_deadlock counted; older summary code compared deadlock and undercounted. Dense archive retains original audited counts.'})
 print('Inventory:',len(out),'exact state-eta;',sum(r['B63'] for r in out),'B63')
 return out
def plan(name,rows,shards=6):
 have={(r['state_id'],r['eta_key']) for r in read(HERE/'exact_q64_inventory.csv')};unique={};manifest=[]
 for r in rows:
  v=np.array(r['eta']);assert inside((v-AFF)/SCALE)[0],r
  k=(r['state_id'],key(v));r=dict(r,eta_key=k[1],cached_q64=k in have)
  manifest.append(r)
  if k not in have:unique.setdefault(k,r)
 write(f'targeted_probe_rounds/{name}/manifest.csv',[dict(r,eta=json.dumps(r['eta'])) for r in manifest])
 out=HERE/'plans'/name;out.mkdir(parents=True,exist_ok=True)
 for sh in range(shards):
  with open(out/f'shard{sh}.jsonl','w') as f:
   for n,r in enumerate(unique.values()):
    if n%shards!=sh:continue
    for seed in range(64):f.write(json.dumps(dict(r,future_index=seed))+'\n')
 estimate={'new_exact_q64':len(unique),'new_continuations':64*len(unique),'shards':shards,'estimated_seconds':len(unique)*9.5+60,'reference':'43Q64 in407s on6shards, prior run',
  'requires_diagnosis_if_wall_exceeds_3x':True,'manifest_sha256':sha(HERE/f'targeted_probe_rounds/{name}/manifest.csv')}
 dump(f'targeted_probe_rounds/{name}/cost_estimate.json',estimate);print(name,estimate)
def common():
 targets=read(POINT/'selected_eta_targets.csv');modes=Counter(tuple(float(r['target_eta'+str(i)]) for i in (1,2,3)) for r in targets)
 panel=[{'mode_id':i,'eta1':v[0],'eta2':v[1],'eta3':v[2],'target_frequency':n} for i,(v,n) in enumerate(sorted(modes.items()))]
 # A fitted RAP common witness is an independently named hypothesis, not evidence of robustness.
 witness=AFF+SCALE*np.array([-.1151046753,.1660461426,.0615081787])
 panel.append(dict(mode_id=9,eta1=witness[0],eta2=witness[1],eta3=witness[2],target_frequency=0))
 write('common_eta_panel.csv',panel)
 rows=[dict(state_id=sid,eta=eta(p).tolist(),mode_id=p['mode_id'],phase='common_core_q64') for sid in sorted(STATES) for p in panel]
 plan('common_core',rows)
if __name__=='__main__':
 inventory()
 if '--common' in sys.argv:common()
