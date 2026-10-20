"""Round20 exact request inventory; never open evaluation outcomes."""
from core import *
import copy
from collections import defaultdict
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity,float_hex
from shared_rollout_db.src.planner import preflight

def create():
    prereg();gate();assert read(HERE/'STATE_DEDUP_AUDIT.json')['passed'];old=read(E8/'protocol.json');geom=read(HERE/'GEOMETRIES.json');states=read(HERE/'FRESH_STATES.json');profiles={};rows={'pilot':[],'probe':[],'main':[]}
    base={sc:{n:old['controllers'][sc][n] for n in NAMES} for sc in SCENES}
    for variant,info in geom['scenarios'].items():
        sc=info['scene'];H=info['config']['max_steps'];t1=round(3*H/7);t2=round(5*H/7);profiles[variant]={}
        for n in NAMES:
            for arm in ARMS:
                if arm.startswith('F'):a=b=int(arm[1:]);t=H
                else:a=int(arm[1:3]);b=10;t=t1 if arm.endswith('t1') else t2
                v=copy.deepcopy(base[sc][n]['payload']);v.update(round20_geometry_variant=variant,round20_scenario_uid=info['scenario_uid'],round20_geometry=geom['geometries'][variant],round20_schedule=dict(eta_a_hex=[float_hex(x) for x in old['eta'][a]],eta_b_hex=[float_hex(x) for x in old['eta'][b]],switch_step=t,action_index_rule='a for t<switch; b otherwise'),round20_run_sha256=sha(HERE/'run.py'))
                ctl=uid('ctl',v);profiles[variant][n+'__'+arm]=dict(controller_uid=ctl,payload=v)
                for s in (q for q in states if q['variant']==variant and (q['study_split']=='evaluation' or q['study_split']=='pilot' and n=='flow_isotropic' and arm in ('F10','F15'))):
                    phase='main' if s['study_split']=='evaluation' else 'pilot'
                    for seed in (range(2) if phase=='main' else (0,)):
                        sk=canonical(dict(future_index=seed,future_root=2026100403,rng_namespace=s['rng_namespace']));e=eta_identity(old['eta'][a])[0];rid=uid('roll',dict(state=s['state_uid'],eta=e,controller=ctl,seed=sk));rows[phase].append(dict(state_uid=s['state_uid'],eta_uid=e,controller_uid=ctl,seed_key=sk,rollout_uid=rid,scene=sc,variant=variant,controller_name=n+'__'+arm,base_controller_name=n,split=s['study_split'],state_index=s['study_index'],eta_index=a,eta=old['eta'][a],eta_a=old['eta'][a],eta_b=old['eta'][b],switch_step=t,seed=seed,arm=arm))
            if n=='flow_isotropic':
                for a in (13,15):
                    v=copy.deepcopy(base[sc][n]['payload']);v.update(round20_geometry_variant=variant,round20_scenario_uid=info['scenario_uid'],round20_geometry=geom['geometries'][variant],round20_probe=dict(eta=a,steps=t1,termination_disabled=True,no_label=True),round20_probe_sha256=sha(HERE/'probe.py'))
                    ctl=uid('ctl',v);profiles[variant][f'probe_{a}']=dict(controller_uid=ctl,payload=v)
                    for s in (q for q in states if q['variant']==variant and q['study_split']=='probe'):
                        sk=canonical(dict(future_index=0,future_root=2026100403,rng_namespace=s['rng_namespace']));e=eta_identity(old['eta'][a])[0];rid=uid('roll',dict(state=s['state_uid'],eta=e,controller=ctl,seed=sk));rows['probe'].append(dict(state_uid=s['state_uid'],eta_uid=e,controller_uid=ctl,seed_key=sk,rollout_uid=rid,scene=sc,variant=variant,controller_name=n,base_controller_name=n,split='probe',state_index=s['study_index'],eta_index=a,eta=old['eta'][a],seed=0,probe_steps=t1))
    assert len(rows['pilot'])==256 and len(rows['probe'])==384 and len(rows['main'])==13440
    p=dict(schema='round20_cross_scene_probe_selection_v1',experiment_uid=EXP,states=states,controllers=profiles,base_profiles=base,environments=old['environments'],eta=old['eta'],geometries=geom['geometries'],source_hashes=read(HERE/'HISTORICAL_ASSET_INDEX.json')['assets'],run_sha256=sha(HERE/'run.py'),probe_sha256=sha(HERE/'probe.py'),core_sha256=sha(HERE/'core.py'),prereg_sha256=sha(HERE/'PREREGISTRATION.md'),gate_sha256=sha(HERE/'CONSISTENCY_GATE.json'),dedup_sha256=sha(HERE/'STATE_DEDUP_AUDIT.json'))
    freeze(HERE/'protocol.json',p)
    with connect() as db,transaction(db):
        for variant,info in geom['scenarios'].items():
            sid=info['scenario_uid'];db.execute('INSERT OR IGNORE INTO scenario(scenario_uid,name,code_config_fingerprint,metadata_json) VALUES(?,?,?,?)',(sid,info['scene']+'_'+variant,uid('fingerprint',info['config']),canonical(dict(parent=old['environments'][info['scene']]['scenario_uid'],config=info['config']))))
        for s in states:
            db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',(s['state_uid'],s['scenario_uid'],s['family'],s['content_hash'],canonical(s['physical']),canonical(geom['scenarios'][s['variant']]['config']),canonical(dict(alias=s['uid'],split=s['study_split'],metadata=s['metadata'])),'CONTENT_EXACT'))
        for variant,pr in profiles.items():
            for c in pr.values():
                v=c['payload'];db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(c['controller_uid'],geom['scenarios'][variant]['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
        for phase in ('pilot','main'):
            db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(EXP+'__'+phase,'Round20 '+phase,str(HERE/phase),sha(HERE/'protocol.json'),sha(HERE/'run.py'),canonical(dict(phase=phase))))
    for phase in ('pilot','probe','main'):
        case=rows[phase];freeze(HERE/f'{phase}_CASES.json',case);groups=defaultdict(list)
        for c in case:groups[c['state_uid'],c['eta_uid'],c['controller_uid']].append(c['seed_key'])
        freeze(HERE/f'{phase}_PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=keys) for (s,e,c),keys in groups.items()]));pre=preflight(HERE/f'{phase}_PLANNED_ROLLOUTS.json');write(HERE/f'{phase}_CACHE_PREFLIGHT.json',pre);assert pre['summary']['ambiguous']==0
        with connect(True) as db:
            missing=[c for c in case if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=? AND conflict_quarantined=0',(c['rollout_uid'],)).fetchone() is None]
        write(HERE/f'{phase}_MISSING_CASES.json',missing);write(HERE/f'{phase}_PREFLIGHT.receipt.json',dict(utc=utc(),planned=len(case),missing=len(missing),preflight_sha256=sha(HERE/f'{phase}_CACHE_PREFLIGHT.json'),manifest_sha256=sha(HERE/f'{phase}_PLANNED_ROLLOUTS.json'),protocol_sha256=sha(HERE/'protocol.json')))
        print(phase,len(case),'missing',len(missing))
if __name__=='__main__':create()
