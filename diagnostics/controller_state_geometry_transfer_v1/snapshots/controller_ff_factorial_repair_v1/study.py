"""Combine identified input and crossed-supervision repairs, with controls.

This remains a modest DEVELOPMENT experiment. Full original K16 / fresh-state
confirmation is conditional on this stage passing state-use falsification.
"""
import argparse,copy,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from diagnostics.controller_flow_endpoint_probe_v1 import study as pilot
from diagnostics.controller_four_scene_mechanism_v1.study import HERE as FOUR,read,write,freeze,sha
from diagnostics.controller_q_diversity_v1 import experiment as old
from diagnostics.controller_response_families_v1 import study as parent
from shared_rollout_db.src.rollout_db import connect,transaction,canonical

HERE=Path(__file__).resolve().parent
EXPERIMENT='exp_controller_ff_factorial_root_repair_v1'


def prepare():
    previous=read(pilot.HERE/'protocol.json');source=read(FOUR/'protocol.json')
    p=copy.deepcopy(previous);p.update(schema='ff_factorial_root_repair_v1',states=copy.deepcopy(source['states']),
        purpose='Test input representation and useful state/eta supervision jointly, while controlling alternative fixed/controller eta priors.',
        scope='DEVELOPMENT:32TRAIN+8VAL natural states/scene; five goal-reference and five Flow-reference affine programs; four pre-frozen eta. No final/B15/K16 claim.',
        expected_new_outcomes=6400,expected_reuse_outcomes=8960,
        cache='Reuse all7680goal-reference outcomes and1280Flow-reference pilot outcomes, including numerical unknown. Generate only missing6400Flow-program outcomes.',
        train_seeds=[0,1],validation_seeds=[0,1,2,3],
        source_state_coverage='32 independent physical TRAIN states/scene. This diagnostic does not claim that32states suffice for final training; inspect within-source validation before interpreting transfer failures.',
        primary_label_controls='Real crossed labels vs Flow-reference source-TRAIN targets replaced by per-scene/controller/exact-eta mean. The latter retains controller/eta priors but removes Flow within-scene state associations.',
        primary_input_controls='Same83channel fused architecture: unconstrained field64 only, versus field64+current action5+encounter14. Withheld channels are zero, not a smaller model.',
        baselines=['eta_only_same_all_labels','field_same_all_labels','encounter_same_all_labels','encounter_flow_eta_prior','encounter_goal_only'],
        checkpoint='Source validation selection regret first; strong eta-pair ranking then centered state-eta Q error, probability/controller error and NLL. Rule fixed before these new VAL labels. No held-scene fitting/normalization/selection.',
        training='Same6000steps,120cells/step,AdamW8e-4,weight_decay1e-4,three seeds17/23/41,per-cell binomialNLL. Labels and source scene weighting identical for primary learned baselines.',
        next_gate='Finish all four source-only folds and joint diagnostic. Check state mismatch, C mismatch, state-effect/eta contrasts and all4fixed-eta controls. Expand physical coverage only if source-generalization diagnostics warrant it; finalK16 fresh confirmation only after the method is frozen.')
    freeze(HERE/'protocol.json',p)
    if not (HERE/'runtime.py').exists():(HERE/'runtime.py').symlink_to(pilot.HERE/'runtime.py')
    if not (HERE/'static_operator_audit.json').exists():(HERE/'static_operator_audit.json').symlink_to(pilot.HERE/'static_operator_audit.json')
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXPERIMENT,'Crossed FF input/supervision root repair',str(HERE),sha(HERE/'protocol.json'),sha(__file__),canonical(dict(max_idle_cpu=24))))
    cases=[old.case(p,s,c,j,k) for s in p['states'] for c in p['controllers'][s['scenario']] for j in p['eta_indices']
           for k in (p['train_seeds'] if s['study_split']=='train' else p['validation_seeds'])]
    cases.sort(key=lambda r:(r['split'],r['state_index'],r['controller_name'],r['eta_index'],r['seed'],r['scene']))
    freeze(HERE/'cases.json',cases);missing=[];counts=Counter();groups=defaultdict(list)
    with connect(True) as db:
        for r in cases:
            a=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if a is None:missing.append(r)
            else:
                assert not a['conflict_quarantined'] and all(a[k]==r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
                counts['cached_unknown' if a['numerical_failure'] else 'exact_reuse']+=1
            groups[r['state_uid'],r['eta_uid'],r['controller_uid']].append(r['seed_key'])
    assert all(r['controller_name'].startswith('flow_') for r in missing)
    counts['genuinely_missing']=len(missing);write(HERE/'missing_cases.json',missing)
    write(HERE/'exact_identity_preflight.json',dict(counts,cases_sha256=sha(HERE/'cases.json')))
    freeze(HERE/'planned_rollouts.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=k) for (s,e,c),k in groups.items()]))
    print(json.dumps(dict(counts)),flush=True)


def worker(index,workers):
    from diagnostics.controller_flow_endpoint_probe_v1.runtime import FlowEndpointMatchedField
    p=read(HERE/'protocol.json');assert all(sha(Path(f))==h for f,h in p['source_code_hashes'].items())
    parent.HERE=HERE;parent.EXPERIMENT=EXPERIMENT
    parent.EndpointMatchedField=lambda base,name:FlowEndpointMatchedField(base,name.removeprefix('flow_'))
    parent.worker(index,workers)


def merge():
    parent.HERE=HERE;parent.EXPERIMENT=EXPERIMENT;parent.merge()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','merge']);p.add_argument('--index',type=int,default=0);p.add_argument('--workers',type=int,default=96);a=p.parse_args()
    if a.stage=='worker':worker(a.index,a.workers)
    else:globals()[a.stage]()
