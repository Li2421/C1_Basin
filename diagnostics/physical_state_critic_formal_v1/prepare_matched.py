"""Freeze expanded Toy/Ring matched TT/FF K16/Q4/Q16 rollout identities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'diagnostics/orthoflow3_tt_ff_matched_learnability_v1'
sys.path.insert(0, str(ROOT))
from shared_rollout_db.src.rollout_db import canonical, uid, eta_identity, connect, transaction

OLD = json.loads((SOURCE / 'protocol.json').read_text())
FUTURE_ROOT = 2026100403
EXPERIMENT = 'exp_formal_physical_state_critic_matched_20261008_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def freeze(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert read(path) == obj, f'frozen artifact differs: {path}'
    else:
        path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def key(s, seed):
    return canonical(dict(future_index=seed, future_root=FUTURE_ROOT,
                          rng_namespace=s['rng_namespace']))


def states():
    output = []
    for scene in ('toy_give_way', 'ring_exchange'):
        env = OLD['environments'][scene]
        source = read(HERE / f'states_{scene}.json')
        for s in source['states']:
            physical = s['physical']
            content = uid('content', dict(physical=physical, reset='empty monitor true-t0'))
            sid = uid('state', dict(scenario=env['scenario_uid'], content=content))
            if 'prior_state_uid' in s['metadata']:
                assert sid == s['metadata']['prior_state_uid']
            output.append(dict(uid=s['alias'], state_uid=sid, scenario=scene,
                               scenario_uid=env['scenario_uid'], content_hash=content,
                               physical=physical, rng_namespace=s['rng_namespace'],
                               family=f"formal_q:{scene}:{s['alias']}",
                               metadata=dict(split=s['split'],draw_seed=s['draw_seed'],
                                             origin=s['origin'],physical_hash=s['physical_hash'])))
    assert len(output) == 1344 and len({s['state_uid'] for s in output}) == 1344
    return output


def main():
    eta = OLD['eta']
    assert len(eta) == 16
    state_rows = states()
    protocol = dict(schema='formal_matched_tt_ff_v1',experiment_uid=EXPERIMENT,
                    states=state_rows, eta=eta, controllers=OLD['controllers'],
                    environments=OLD['environments'], future_root=FUTURE_ROOT,
                    source_protocol_sha256=sha(SOURCE/'protocol.json'),
                    train_states_per_scene=512,validation_states_per_scene=32,
                    fresh_test_states_per_scene=128,
                    train_seeds=list(range(4)), validation_test_seeds=list(range(16)),
                    placements=['TT','FF'],
                    candidate_policy='identical frozen public Sobol K16 across TT/FF and states',
                    safety_policy='eta=(0,0,0) applied by identical frozen controller and seed; no learned correction',
                    controller_or_flow_changed=False,
                    test_labels_forbidden_until_models_frozen=True)
    freeze(HERE/'matched_protocol.json',protocol)
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
                   (EXPERIMENT,'Formal physical-state critic matched TT/FF',str(HERE),sha(HERE/'matched_protocol.json'),sha(__file__),canonical(dict(pure_Q=True))))
        for s in state_rows:
            db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',
                       (s['state_uid'],s['scenario_uid'],s['family'],s['content_hash'],
                        canonical(s['physical']),canonical(OLD['environments'][s['scenario']]['config']),
                        canonical(dict(source_alias=s['uid'],metadata=s['metadata'])),'CONTENT_EXACT'))
            db.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',
                       (s['scenario_uid'],s['uid'],s['state_uid'],EXPERIMENT))
        for e in eta+[[0.,0.,0.]]:
            eid,v,hx=eta_identity(e)
            db.execute('INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex) VALUES(?,?,?,?,?)',
                       (eid,*v,hx))
    counts = {}
    for scene in ('toy_give_way','ring_exchange'):
        for chain in ('TT','FF'):
            ctl = OLD['controllers'][scene][chain]['controller_uid']
            for split in ('train','validation','test'):
                requests = []
                cases = []
                n = 4 if split == 'train' else 16
                subset = [s for s in state_rows if s['scenario']==scene and s['metadata']['split']==split]
                for i,s in enumerate(subset):
                    bank = list(enumerate(eta)) + ([(16,[0.,0.,0.])] if split!='train' else [])
                    for j,e in bank:
                        eid=eta_identity(e)[0]
                        seed_keys=[key(s,k) for k in range(n)]
                        requests.append(dict(state_uid=s['state_uid'],eta_uid=eid,
                                             controller_uid=ctl,seed_keys=seed_keys))
                        for k,sk in enumerate(seed_keys):
                            ident=dict(state=s['state_uid'],eta=eid,controller=ctl,seed=sk)
                            cases.append(dict(state_index=i,state_uid=s['state_uid'],source_alias=s['uid'],
                                              scene=scene,chain=chain,split=split,
                                              eta_index=j,eta=e,eta_uid=eid,controller_uid=ctl,
                                              seed=k,seed_key=sk,rollout_uid=uid('roll',ident)))
                panel=HERE/'matched'/scene/chain/split
                freeze(panel/'cases.json',cases)
                freeze(panel/'planned_rollouts.json',dict(requests=requests))
                counts[f'{scene}/{chain}/{split}']=dict(states=len(subset),candidates=16,
                    safety_candidates=1 if split!='train' else 0,
                    state_eta_pairs=len(requests),rollout_slots=len(cases))
    freeze(HERE/'matched_counts.json',counts)
    print(json.dumps(counts))


if __name__=='__main__':
    main()
