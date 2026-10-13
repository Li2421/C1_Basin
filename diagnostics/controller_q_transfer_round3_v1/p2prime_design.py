"""Conditional P2′ full design, only executable after P1 gate failure."""
import copy
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
from diagnostics.controller_q_diversity_v1 import experiment as old
from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,v):
    path=HERE/name
    if path.exists():assert read(path)==v,('immutable',name)
    else:path.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    gate=read(HERE/'P1_RESULTS.json')['gate']
    assert not gate['p2_qualified'],'P2′ only if P1 gate failed'
    native=read(ROOT/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/protocol.json')
    placement=read(ROOT/'diagnostics/orthoflow3_tt_ff_matched_learnability_v1/protocol.json')
    kappas=read(ROOT/'diagnostics/ring_state_selection_transfer_v1/c1/protocol.json')
    states=generate('train',256,2026110000,native)+generate('validation',32,2026120000,native)+generate('test',64,2026130000,native)
    old_uids={s['state_uid'] for s in native['states']}|{s['state_uid'] for s in read(ROOT/'diagnostics/state_nuisance_rootcause_v1/heldout_adaptive_selection_v1/protocol.json')['states']}
    assert len({s['state_uid'] for s in states})==352 and not old_uids.intersection(s['state_uid'] for s in states)
    sources=['FF','TT','rotate_negative','isotropic','rotate_positive'];held=['TF','FT']
    profiles={'FF':copy.deepcopy(native['controller']),'TT':copy.deepcopy(placement['controllers']['ring_exchange']['TT'])}
    for name in sources[2:]:profiles[name]=copy.deepcopy(kappas['controllers']['ring_exchange'][name])
    for name in held:
        v=copy.deepcopy(native['controller']['payload'])
        v.update(base_controller='native_v10_ring_factorial_placement_round3_v1',placement=name,
                 controller_source_sha256=sha(HERE/'placement_runtime.py'),
                 execution_callable=str(HERE/'placement_runtime.py'),execution_callable_sha256=sha(HERE/'placement_runtime.py'),
                 factorial_code_sha256=sha(Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004/validation_stage2/factorial_controller.py')),
                 conditioning='free true-t0 native Flow v10; frozen TF/FT factorial; same Ring plant, FF safety and η basis')
        v.pop('legacy_controller_uid_verified_by_original_function',None)
        profiles[name]=dict(controller_uid=uid('ctl',v),payload=v)
    for name in sources+held:
        assert profiles[name]['payload']['jax_enable_x64'] is False,name
        assert profiles[name]['payload']['horizon']==700
    protocol=dict(schema='controller_q_transfer_round3_p2prime_v1',experiment_uid='exp_controller_q_transfer_round3_p2prime_v1',
                  states=states,controllers={'ring_exchange':profiles},source_controllers=sources,held_controllers=held,
                  eta=native['eta'],seeds=list(range(8)),horizon=700,dt=.05,
                  environments={'ring_exchange':native['environment']},
                  prereg_sha256=sha(HERE/'PREREGISTRATION_P2PRIME.md'),p1_gate_sha256=sha(HERE/'P1_RESULTS.json'),
                  placement_runtime_sha256=sha(HERE/'placement_runtime.py'),
                  ff_operator_sha256=sha(Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004/poc/field_controller.py')),
                  factorial_code_sha256=sha(Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004/validation_stage2/factorial_controller.py')),
                  jax_enable_x64=False,split_seeds=dict(train=2026110000,validation=2026120000,test=2026130000),
                  scope='Real Flow checkpoint fixed v10, held-out placement TF/FT; distinct from held-out real Flow P2')
    dump('P2PRIME_PROTOCOL.json',protocol)
    rng=np.random.default_rng(2026101207);train_masks={s['state_uid']:sorted(map(int,rng.permutation(16)[:8])) for s in states[:256]}
    dump('P2PRIME_TRAIN_ETA_MASK.json',train_masks)
    cases=[]
    for name in sources:
        for s in states[:288]:
            indices=train_masks[s['state_uid']] if s['study_split']=='train' else range(16)
            for j in indices:
                for k in range(4):cases.append(old.case(protocol,s,name,j,k))
    for name in held:
        for s in states[288:]:
            for j in range(16):
                for k in range(8):cases.append(old.case(protocol,s,name,j,k))
    assert len(cases)==67584
    cases.sort(key=lambda r:(r['controller_name'],r['split'],r['state_index'],r['eta_index'],r['seed']))
    dump('P2PRIME_CASES.json',cases)
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
                   (protocol['experiment_uid'],'Round3 held placement TF FT',str(HERE),sha(HERE/'P2PRIME_PROTOCOL.json'),sha(__file__),canonical(dict(max_new_rollouts=70000,max_cpu_workers=24))))
        for s in states:
            db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',
                       (s['state_uid'],s['scenario_uid'],s['family'],s['content_hash'],canonical(s['physical']),canonical(native['environment']['config']),canonical(dict(source_alias=s['uid'],metadata=s['metadata'])),'CONTENT_EXACT'))
            db.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',(s['scenario_uid'],s['uid'],s['state_uid'],protocol['experiment_uid']))
        for eta in native['eta']:
            eid,v,hx=eta_identity(eta)
            db.execute('INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex) VALUES(?,?,?,?,?)',(eid,*v,hx))
        for ctl in profiles.values():
            v=ctl['payload'];oldc=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(ctl['controller_uid'],)).fetchone()
            assert oldc is None or oldc['config_json']==canonical(v)
            db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                       (ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
    from collections import Counter
    with connect(True) as db:
        counts=Counter();missing=[];groups=defaultdict(list);used=[];unknown=[];quarantine=[]
        for r in cases:
            oldrow=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if oldrow is None:missing.append(r)
            elif oldrow['conflict_quarantined']:quarantine.append(r['rollout_uid'])
            elif oldrow['numerical_failure']:unknown.append(r['rollout_uid'])
            else:used.append(r['rollout_uid'])
            groups[(r['state_uid'],r['eta_uid'],r['controller_uid'])].append(r['seed_key'])
    assert not quarantine and len(missing)<=70000
    dump('P2PRIME_PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=k) for (s,e,c),k in groups.items()]))
    dump('P2PRIME_MISSING_CASES.json',missing)
    dump('P2PRIME_CACHED_UIDS.json',dict(valid=used,numerical_unknown=unknown,quarantined=quarantine))
    dump('P2PRIME_COST.json',dict(total=len(cases),new=len(missing),cached_valid=len(used),cached_unknown=len(unknown),
                                  estimated_cpu_hours=len(missing)*4/3600,estimated_wall_hours_at24=len(missing)*4/(3600*24),
                                  per_episode_assumed_seconds=4,max_new=70000))
    print(json.dumps(dict(gross=len(cases),new=len(missing),cached=len(used)+len(unknown))))


if __name__=='__main__':main()
