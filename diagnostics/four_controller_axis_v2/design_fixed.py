"""Design/register Round19b fixed-arm census and run official preflight."""
from __future__ import annotations
import copy,datetime,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT)]
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity
from shared_rollout_db.src.planner import preflight
EXP='exp_round19b_four_controller_axis_fixed_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def main():
    assert read(HERE/'R5_FREEZE.receipt.json')['features_sha256']==sha(HERE/'R5_FROZEN.npz')
    assert read(HERE/'STATE_FREEZE.receipt.json')['fresh_states_sha256']==sha(HERE/'FRESH_STATES.json')
    old=read(ROOT/'diagnostics/controller_ff_factorial_repair_v1/protocol.json')
    states=[s for s in read(HERE/'FRESH_STATES.json') if s['study_split']=='census'];assert len(states)==32
    g2=read(HERE/'G2_REPORT.json')['checks'];assert len(g2)==25
    base=old['controllers']['four_way_intersection']['flow_isotropic']['payload'];profiles={};cases=[]
    for item in g2:
        name=item['name']+'__'+item['flavor'];p=copy.deepcopy(base)
        p.update(flow_checkpoint_path=item['checkpoint_path'],flow_checkpoint_sha256=item['checkpoint_sha256'],
                 round19b_adapter_sha256=sha(HERE/'adapter.py'),
                 conditioning=base['conditioning']+'+round19b_checkpoint_native_flow_v1',
                 execution_semantics='round19b_four_flow_isotropic_fixed_v1')
        cid=uid('ctl',p);profiles[name]=dict(controller_uid=cid,payload=p,run=item['name'],flavor=item['flavor'])
        for s in states:
            for k in (10,13,15):
                eta=old['eta'][k];e=eta_identity(eta)[0]
                for seed in (0,1):
                    sk=canonical(dict(future_index=seed,future_root=2026100403,rng_namespace=s['rng_namespace']))
                    rid=uid('roll',dict(state=s['state_uid'],eta=e,controller=cid,seed=sk))
                    cases.append(dict(state_uid=s['state_uid'],eta_uid=e,controller_uid=cid,seed_key=sk,rollout_uid=rid,
                                      scene='four_way_intersection',controller_name=name,base_controller_name=name,
                                      state_index=s['study_index'],eta_index=k,eta=eta,eta_a=eta,eta_b=eta,
                                      switch_step=1100,seed=seed,arm='F'+str(k)))
    assert len(cases)==25*32*3*2 and len({x['rollout_uid'] for x in cases})==len(cases)
    protocol=dict(schema='round19b_four_fixed_v1',experiment_uid=EXP,
                  preregistration_sha256=sha(HERE/'PREREGISTRATION.md'),r5_freeze_sha256=sha(HERE/'R5_FREEZE.receipt.json'),
                  states=states,profiles=profiles,controller_order=list(profiles),eta=old['eta'],
                  environments=old['environments'],runner_sha256=sha(HERE/'run_full.py'),adapter_sha256=sha(HERE/'adapter.py'),
                  arms=['F10','F13','F15'],horizon=1100,jax_enable_x64=False)
    write(HERE/'FIXED_PROTOCOL.json',protocol);write(HERE/'FIXED_CASES.json',cases)
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
                   (EXP,'Round19b Four fixed-arm census',str(HERE/'stage2'),sha(HERE/'FIXED_PROTOCOL.json'),sha(HERE/'run_full.py'),
                    canonical(dict(requests=len(cases),probe_freeze=sha(HERE/'R5_FREEZE.receipt.json')))))
        for s in states:
            sid=s['scenario_uid'];db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',
                (s['state_uid'],sid,s['family'],s['content_hash'],canonical(s['physical']),canonical(old['environments']['four_way_intersection']['config']),canonical(dict(alias=s['uid'],metadata=s['metadata'])),'CONTENT_EXACT'))
            db.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',(sid,s['uid'],s['state_uid'],EXP))
        for name,ctl in profiles.items():
            v=ctl['payload'];db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],
                 uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),
                 v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
            stored=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(ctl['controller_uid'],)).fetchone()
            assert stored[0]==canonical(v)
        for k in (10,13,15):
            e,v,h=eta_identity(old['eta'][k]);db.execute('INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex) VALUES(?,?,?,?,?)',(e,*v,h))
    groups=defaultdict(list)
    for x in cases:groups[x['state_uid'],x['eta_uid'],x['controller_uid']].append(x['seed_key'])
    manifest=dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=keys) for (s,e,c),keys in groups.items()])
    write(HERE/'FIXED_PLANNED_ROLLOUTS.json',manifest)
    result=preflight(HERE/'FIXED_PLANNED_ROLLOUTS.json');write(HERE/'FIXED_CACHE_PREFLIGHT.json',result)
    assert result['summary']['ambiguous']==0
    missing=[];existing=[]
    with connect(True) as db:
        for x in cases:
            row=db.execute('SELECT numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(x['rollout_uid'],)).fetchone()
            if row:
                assert not row['conflict_quarantined'];existing.append(dict(uid=x['rollout_uid'],unknown=bool(row['numerical_failure'])))
            else:missing.append(x)
    assert len(missing)+len(existing)==len(cases)
    write(HERE/'FIXED_MISSING_CASES.json',missing)
    write(HERE/'FIXED_CACHE_INVENTORY.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          planned=len(cases),existing=len(existing),genuinely_missing=len(missing),unknown_existing=sum(x['unknown'] for x in existing),
          official_preflight_summary=result['summary'],no_unknown_rerun=True))
    print(json.dumps(dict(stage='fixed_design',planned=len(cases),missing=len(missing),preflight=result['summary'])),flush=True)
if __name__=='__main__':main()
