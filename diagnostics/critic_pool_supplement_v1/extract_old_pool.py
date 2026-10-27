"""Read-only, whitelisted historical observations; no validation-scene data selected."""
from common import *
from design import *
from scenes import features
from descriptors_v2 import descriptor
import numpy as np
NAMES={'lit_narrow_passage_flow':[3],'lit_crowded_random_flow':[2,4,6,8,10,12,14],'lit_swap8_moving_swap8_si_user_fix_20261023':[8],'xsc_swap_n4_cross_scene_si_v1':[4],'xsc_swap_n12_cross_scene_si_v1':[12],'xsc_swap_n8_dense_cross_scene_si_v1':[8],'xsc_div_size_r014_cross_scene_si_v1':[6],'xsc_div_time_f20_cross_scene_si_v1':[6]}
EXPECTED={'lit_narrow_passage_flow__N3':3882,**{f'lit_crowded_random_flow__N{n}':v for n,v in zip([2,4,6,8,10,12,14],[96,96,96,3185,96,96,769])},'lit_swap8_moving_swap8_si_user_fix_20261023__N8':7296,'xsc_swap_n4_cross_scene_si_v1__N4':1841,'xsc_swap_n12_cross_scene_si_v1__N12':1800,'xsc_swap_n8_dense_cross_scene_si_v1__N8':1836,'xsc_div_size_r014_cross_scene_si_v1__N6':900,'xsc_div_time_f20_cross_scene_si_v1__N6':900}

def main():
 static={s['name']:s for p in [OLD/'SCENES.json',ROOT/'diagnostics/eta_diverse_scene_test_v1/SCENES.json']for s in read_evidence(p)};groups={};record_hash=hashlib.sha256();records=[]
 with dbm.connect(True)as db:
  scenarios=db.execute('SELECT * FROM scenario WHERE name IN ('+','.join('?'for _ in NAMES)+')',list(NAMES)).fetchall();chosen={}
  for r in scenarios:
   meta=json.loads(r['metadata_json']);cfg=meta['config'];n=cfg['num_agents']
   if n not in NAMES[r['name']]:continue
   chosen[r['scenario_uid']]=dict(r);key=r['name']+'__N'+str(n);base=cfg['scene'];assert not base.startswith(FORBIDDEN)
   if key not in groups:groups[key]=dict(name=key,source_version=r['name'],N=n,config=cfg,static_name=base,states={},cells={},scenario_uids=[])
   g=groups[key];assert canonical(g['config'])==canonical(cfg),('MULTIPLE_CONFIGS_SAME_VERSION',key);g['scenario_uids'].append(r['scenario_uid'])
  state_snapshot=[]
  for row in db.execute('SELECT state_uid,scenario_uid,physical_state_json FROM state WHERE scenario_uid IN ('+','.join('?'for _ in chosen)+')',list(chosen)):
   physical=json.loads(row['physical_state_json']);sc=chosen[row['scenario_uid']];cfg=json.loads(sc['metadata_json'])['config'];g=groups[sc['name']+'__N'+str(cfg['num_agents'])];assert len(physical['positions'])==g['N'];g['states'][uid('physical',physical)]=physical;state_snapshot.append(dict(row))
  write(HERE/'REGISTERED_TRAINING_STATES.receipt.json',dict(states=state_snapshot,all_registered_states=True,labels_not_imputed=True))
  sql="""SELECT r.rollout_uid,s.scenario_uid,r.state_uid,r.controller_uid,r.eta_uid,r.seed_key,r.success,r.numerical_failure,r.conflict_quarantined,s.physical_state_json,c.config_json,e.eta1,e.eta2,e.eta3,e.canonical_hex
  FROM rollout r JOIN state s USING(state_uid) JOIN controller_config c USING(controller_uid) JOIN eta e USING(eta_uid)
  WHERE s.scenario_uid IN ("""+','.join('?'for _ in chosen)+""") AND r.conflict_quarantined=0
  AND json_extract(c.config_json,'$.safety_layer')='distributed'
  AND json_extract(c.config_json,'$.correction_basis')='orthoflow3_local_contact_gated'
  AND json_extract(c.config_json,'$.correction_semantics')='contact_conflict_relative_gate_20261021'
  AND json_extract(c.config_json,'$.nominal.controller') IN (?,?,?) ORDER BY r.rollout_uid"""
  for row in db.execute(sql,[*chosen,*CONTROL_IDS]):
   r=dict(row);sc=chosen[r['scenario_uid']];cfg=json.loads(sc['metadata_json'])['config'];key=sc['name']+'__N'+str(cfg['num_agents']);g=groups[key];physical=json.loads(r['physical_state_json']);assert len(physical['positions'])==g['N'];pid=uid('physical',physical);g['states'].setdefault(pid,physical);ctl=json.loads(r['config_json'])['nominal']['controller'];eta=[r['eta1'],r['eta2'],r['eta3']];assert dbm.eta_identity(eta)[2]==r['canonical_hex'];cell=(pid,ctl,etahex(eta));v=g['cells'].setdefault(cell,dict(state=pid,controller=ctl,eta=eta,success=0,unknown=0,n=0));v['success']+=int(r['success']);v['unknown']+=int(r['numerical_failure']);v['n']+=1
   digest=hashlib.sha256(canonical(r).encode()).hexdigest();record_hash.update((r['rollout_uid']+digest).encode());records.append(dict(rollout_uid=r['rollout_uid'],sha256=digest,scenario_uid=r['scenario_uid']))
 out=[]
 for key,g in groups.items():
  cfg=g['config'];scene=copy.deepcopy(static[g['static_name']]);scene['name']=key;scene['config']=cfg;states=[]
  for i,(pid,physical)in enumerate(sorted(g['states'].items())):states.append(dict(scene=cfg['scene'],index=i,N=g['N'],**physical,physical_id=pid))
  scene['states']=states;scene.setdefault('moving',False);scene['features']=features(scene);cp=descriptor(scene);by={s['physical_id']:i for i,s in enumerate(states)};cells=list(g['cells'].values());cells.sort(key=lambda r:(r['state'],r['controller'],etahex(r['eta'])))
  state_idx=np.array([by[c['state']]for c in cells]);eta=np.asarray([c['eta']for c in cells]);ctl=np.array([CONTROL_IDS.index(c['controller'])for c in cells]);nn=np.array([c['n']for c in cells]);succ=np.array([c['success']for c in cells]);unknown=np.array([c['unknown']for c in cells]);h=np.array([features(dict(scene,states=[s]))[-6:]for s in states])
  path=HERE/'pool_rows'/f'{key}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,state=state_idx,eta=eta,ctl=ctl,y=succ/nn,n=nn,success=succ,unknown=unknown,h=h)
  metadata=dict(name=key,source='official shared_rollout_db, compatible filtered observations',source_version=g['source_version'],scenario_uids=g['scenario_uids'],static_definition=g['static_name'],config=cfg,states=states,state_count=len(states),cells=len(cells),eta_count=len({etahex(v)for v in eta}),c_prime=cp.tolist(),mean_success=float((succ/nn).mean()),new=False,group='old',rows_path=str(path),rows_sha256=sha(path),expected_cells=EXPECTED[key],count_matches_expectation=len(cells)==EXPECTED[key],physical_aliases_merged=True)
  out.append(metadata)
 assert len(out)==14;write(HERE/'OLD_TRAINING_POOL.json',out);write(HERE/'HISTORICAL_ROW_PROVENANCE.json',dict(records=records,selection_sql=sql,record_count=len(records),sha256=record_hash.hexdigest(),validation_scene_rows_selected=0,scope='Whitelisted scenario versions/N, exact3 controllers, distributed corrected SI, all compatible seeds and all exact eta.'))
 write(HERE/'HISTORICAL_POOL_COUNTS.json',{g['name']:dict(cells=g['cells'],expected=g['expected_cells'],equal=g['count_matches_expectation'],states=g['state_count'],eta=g['eta_count'])for g in out})
 print('Extracted14 official training versions;',sum(g['cells']for g in out),'cells. Validation labels not opened.',flush=True)
if __name__=='__main__':main()
