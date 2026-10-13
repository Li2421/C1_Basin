"""Freeze four-scene FF LOSO Q4/Q16 requests and exact cache identities."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from shared_rollout_db.src.rollout_db import canonical, connect, eta_identity, transaction, uid
from diagnostics.orthoflow3_generator_critic_frozen_test_v1 import run_frozen_test as frozen
from diagnostics.orthoflow3_mode_free_generator_critic_hard_cohort_v1.prepare_cache_plan import SCENARIO as TOY_SCENARIO, CORRECTION_CONFIG as TOY_CONFIG

HERE=Path(__file__).resolve().parent
EXPERIMENT='exp_formal_physical_state_critic_loso_20261008_v1'
SCENES=('toy_give_way','double_bottleneck','four_way_intersection','ring_exchange')
SPLITS=('train','validation','test')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): assert read(path)==value, f'Frozen mismatch: {path}'
    else: path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def controllers():
    result={}
    for scene in SCENES:
        if scene=='toy_give_way':
            row=dict(scenario_uid=TOY_SCENARIO,
                     intervention_uid=uid('ctl',TOY_CONFIG),
                     safety_uid=uid('ctl',TOY_CONFIG),
                     intervention_payload=TOY_CONFIG,safety_payload=TOY_CONFIG,
                     runtime='historical Toy WIDE-IC FF runner')
        elif scene=='double_bottleneck':
            rt=frozen.DoubleFrozenRuntime()
            rng=dict(name='formal_new_global_state_current_latched_v1',
                     current_root=948610000, state_token_rule='state_token(state_uid)',
                     future_root=frozen.DOUBLE_FUTURE_ROOT,
                     future_index_fold_in=True, physical_step_fold_in=True)
            p=dict(rt.controllers['orthoflow3']['payload']);p['rng']=rng
            b=dict(rt.controllers['hard_safety_q16']['payload']);b['rng']=rng
            row=dict(scenario_uid=rt.scenario_uid,intervention_uid=uid('ctl',p),
                     safety_uid=uid('ctl',b),intervention_payload=p,safety_payload=b,
                     intervention_source_uid=rt.controllers['orthoflow3']['uid'],
                     safety_source_uid=rt.controllers['hard_safety_q16']['uid'],
                     runtime='historical frozen DB FF runner on independent accepted global states')
        else:
            rt=frozen.FrozenNewRuntime(scene)
            row=dict(scenario_uid=rt.scenario_uid,
                     intervention_uid=rt.controllers['orthoflow3']['uid'],
                     safety_uid=rt.controllers['hard_safety_q16']['uid'],
                     intervention_payload=rt.controllers['orthoflow3']['payload'],
                     safety_payload=rt.controllers['hard_safety_q16']['payload'],
                     runtime='historical FrozenNewRuntime FF runner')
        result[scene]=row
    return result


def main():
    ctl=controllers()
    proposal_hashes={}
    panel_rows={}
    for scene in SCENES:
        source=read(HERE/f'states_{scene}.json')
        by_alias={s['alias']:s for s in source['states']}
        assert len(by_alias)==672
        for split in SPLITS:
            file=HERE/'loso'/scene/split/'proposals.json';pp=read(file)
            assert pp['no_test_labels_used'] if 'no_test_labels_used' in pp else pp['test_labels_used'] is False
            assert pp['states_sha256']==sha(HERE/f'states_{scene}.json')
            rows=pp['states'];assert len(rows)=={'train':512,'validation':32,'test':128}[split]
            assert all(r['state_alias'] in by_alias and by_alias[r['state_alias']]['split']==split for r in rows)
            assert len({r['state_uid'] for r in rows})==len(rows)
            assert all(len(r['eta'])==16 for r in rows)
            proposal_hashes[f'{scene}/{split}']=sha(file)
            panel_rows[(scene,split)]=(rows,by_alias)
    protocol=dict(schema='formal_q_four_scene_loso_ff_v1',experiment_uid=EXPERIMENT,
                  scenes=list(SCENES),splits={'train':512,'validation':32,'test':128},
                  train_seeds=[0,1,2,3],validation_test_seeds=list(range(16)),
                  candidate_count=16,candidate_policy='frozen historical stochastic K16 generator per state; mean excluded',
                  safety_policy='same physical state and future seed, frozen safety controller, eta zero',
                  controller_or_flow_changed=False,flow_or_controller_context_in_critic=False,
                  frozen_proposal_hashes=proposal_hashes,controllers=ctl,
                  state_files={s:sha(HERE/f'states_{s}.json') for s in SCENES},
                  test_labels_forbidden_until_models_frozen=True,
                  train_scene_splits={s:[x for x in SCENES if x!=s] for s in SCENES})
    freeze(HERE/'loso_protocol.json',protocol)
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
                   (EXPERIMENT,'Formal physical-state Q FF LOSO',str(HERE/'loso'),
                    sha(HERE/'loso_protocol.json'),sha(__file__),canonical(dict(pure_Q=True))))
        for scene in SCENES:
            ctrl=ctl[scene]
            for kind in ('intervention','safety'):
                c=ctrl[f'{kind}_uid'];p=ctrl[f'{kind}_payload']
                old=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(c,)).fetchone()
                if old:
                    assert old[0]==canonical(p)
                else:
                    assert scene=='double_bottleneck'
                    base=db.execute('SELECT * FROM controller_config WHERE controller_uid=?',
                                    (ctrl[f'{kind}_source_uid'],)).fetchone()
                    assert base
                    db.execute('INSERT INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                               (c,ctrl['scenario_uid'],base['flow_checkpoint_sha256'],base['orthoflow3_sha256'],
                                base['safety_config_hash'],base['horizon'],base['dt'],base['success_semantics_version'],
                                base['conditioning_version'],canonical(p['rng']),canonical(p),'EXACT_PROFILE'))
            for split in SPLITS:
                rows,by_alias=panel_rows[(scene,split)]
                for r in rows:
                    s=by_alias[r['state_alias']]
                    physical=({'initial_positions':s['physical']['positions']} if scene=='toy_give_way'
                              else s['physical'])
                    content=(hashlib.sha256(canonical(physical).encode()).hexdigest() if scene=='toy_give_way'
                             else uid('content',dict(physical=s['physical'],reset='empty monitor true-t0')))
                    assert r['state_uid']==uid('state',dict(scenario=ctrl['scenario_uid'],content=content))
                    old=db.execute('SELECT * FROM state WHERE state_uid=?',(r['state_uid'],)).fetchone()
                    if old:
                        assert old['physical_state_json']==canonical(physical)
                    else:
                        db.execute('INSERT INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',
                                   (r['state_uid'],ctrl['scenario_uid'],f'formal_q_loso:{scene}:{split}',content,
                                    canonical(physical),canonical({}),canonical(dict(alias=s['alias'],metadata=s['metadata'])),'CONTENT_EXACT'))
                    db.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',
                               (ctrl['scenario_uid'],s['alias'],r['state_uid'],EXPERIMENT))
                    for eta in r['eta']+([[0.,0.,0.]] if split!='train' else []):
                        eid,v,hx=eta_identity(eta)
                        db.execute('INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex) VALUES(?,?,?,?,?)',
                                   (eid,*v,hx))
    counts={}
    for scene in SCENES:
        ctrl=ctl[scene]
        for split in SPLITS:
            rows,_=panel_rows[(scene,split)]
            trials=4 if split=='train' else 16
            cases=[];requests=[]
            for i,r in enumerate(rows):
                bank=list(enumerate(r['eta']))
                if split!='train':bank.append((16,[0.,0.,0.]))
                for j,eta in bank:
                    controller=ctrl['intervention_uid'] if j<16 else ctrl['safety_uid']
                    eid=eta_identity(eta)[0]
                    # Toy's frozen Flow folds rollout_id into both current and future
                    # action keys. Record it in the cache seed identity so equal
                    # physical positions with different rollout_id never cross-reuse.
                    seed_suffix=({'rollout_id':r['rollout_id']} if scene=='toy_give_way' else {})
                    seeds=[canonical(dict(future_index=k,**seed_suffix)) for k in range(trials)]
                    requests.append(dict(state_uid=r['state_uid'],eta_uid=eid,
                                         controller_uid=controller,seed_keys=seeds))
                    for k,sk in enumerate(seeds):
                        identity=dict(state=r['state_uid'],eta=eid,controller=controller,seed=sk)
                        cases.append(dict(scene=scene,split=split,state_index=i,
                                          state_uid=r['state_uid'],state_alias=r['state_alias'],
                                          eta_index=j,eta=eta,eta_uid=eid,controller_uid=controller,
                                          future_index=k,seed_key=sk,rollout_uid=uid('roll',identity)))
            panel=HERE/'loso'/scene/split
            freeze(panel/'cases.json',cases)
            freeze(panel/'planned_rollouts.json',dict(requests=requests))
            counts[f'{scene}/{split}']=dict(states=len(rows),requests=len(requests),slots=len(cases))
    freeze(HERE/'loso_counts.json',counts)
    print(json.dumps(counts))


def preflight(scene,split):
    assert scene in SCENES and split in SPLITS
    panel=HERE/'loso'/scene/split
    subprocess.run([sys.executable,'-m','shared_rollout_db.plan',
                    '--manifest',str(panel/'planned_rollouts.json'),
                    '--output',str(panel/'cache_preflight.json')],check=True,
                   stdout=(panel/'preflight_stdout.json').open('w'))
    cases=read(panel/'cases.json');missing=[];valid=unknown=0
    with connect(True) as db:
        for c in cases:
            row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            if row is None:missing.append(c)
            else:
                assert row['state_uid']==c['state_uid'] and row['eta_uid']==c['eta_uid']
                assert row['controller_uid']==c['controller_uid'] and row['seed_key']==c['seed_key']
                assert row['compatibility_quality']=='EXACT_REUSE' and not row['conflict_quarantined']
                valid+=int(not row['numerical_failure']);unknown+=int(row['numerical_failure'])
    freeze(panel/'missing_cases.json',missing)
    audit=dict(scene=scene,split=split,slots=len(cases),exact_valid=valid,
               exact_numerical_unknown=unknown,missing=len(missing))
    freeze(panel/'preflight_exact_audit.json',audit)
    print(json.dumps(audit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','preflight'))
    p.add_argument('--scene',choices=SCENES);p.add_argument('--split',choices=SPLITS)
    a=p.parse_args()
    if a.action=='prepare':main()
    else:
        assert a.scene and a.split
        preflight(a.scene,a.split)
