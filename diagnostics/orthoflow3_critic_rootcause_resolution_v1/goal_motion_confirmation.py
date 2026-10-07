"""New outcome-blind families, immutable motion controller, paired Q16.

Only dataset/request paths and worker partitioning change. The program runtime,
parent native rollout runtime and both controller IDs remain frozen.
"""
import argparse
import copy
import json
import numpy as np
from . import goal_motion_alias as motion
from . import seed_replication as engine
from .function_support import read, write, sha
from .state_support import validate_plan
from ring_exchange.environment import Config, sample_instance
from new_benchmark_common.basin_dataset_v1 import content_hash
from shared_rollout_db.src.rollout_db import connect, transaction, canonical, uid
OUT = motion.OUT.parent/'goal_motion_confirmation'
EXP = 'exp_c1_goal_motion_independent_family_confirmation_v1'


def prepare():
    assert not (OUT/'protocol.json').exists(), 'Frozen independent confirmation exists'
    proto = copy.deepcopy(read(motion.OUT/'protocol.json'))
    oldstates = read(motion.OUT/'states.json')
    eta_rows = read(motion.OUT/'pairs.json')[:2]
    cfg = Config(**read(OUT.parent.parent/'ring_exchange_stage1/base_u_v10_local_dataset/manifest.json')['scenario_config'])
    states, pairs = [], []
    with connect() as db, transaction(db):
        original = db.execute('SELECT * FROM state WHERE state_uid=?', (oldstates[0]['uid'],)).fetchone()
        for i in range(8):
            seed = 940043000+i
            inst = sample_instance(seed,'development',cfg)
            physical = {k:getattr(inst,k).tolist() for k in ('positions','velocities','goals')}
            physical['timestep'] = 0
            ch = content_hash(physical)
            su = uid('state',{'scenario':original['scenario_uid'],'content':ch})
            assert db.execute('SELECT 1 FROM state WHERE state_uid=?',(su,)).fetchone() is None
            group = f'goal_motion_independent:development:{seed}'
            s = dict(alias=f'MOTION_CONFIRM_{seed}',uid=su,index=i,repair_index=i,content_hash=ch,
                physical=physical,source_group=group,rollout_id=group,split='test',source_dataset_split='development',
                provenance=dict(generator='ring_exchange.environment.sample_instance',initial_seed=seed,native_split='development',
                    outcome_blind=True,task='goal_motion_independent_confirmation',no_model_selection=True))
            geometry=json.loads(original['goals_geometry_json']);geometry['goals']=physical['goals']
            db.execute('INSERT INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,h0_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?,?)',
                (su,original['scenario_uid'],group,ch,canonical(physical),None,canonical(geometry),canonical(s['provenance']),'CONTENT_EXACT'))
            db.execute('INSERT INTO state_alias VALUES(?,?,?,?)',(original['scenario_uid'],s['alias'],su,EXP))
            states.append(s)
            for old in eta_rows:
                pair=copy.deepcopy(old)
                pair.update(state_uid=su,state_index=i,split='test',target_seeds=16)
                pairs.append(pair)
        minimum=min(float(np.linalg.norm(np.array(s['physical']['goals'])-s['physical']['positions'],axis=1).min()) for s in states)
        assert minimum-80*.05*.52>1.
        proto.update(experiment_uid=EXP,experiment='independent_goal_motion_alias_confirmation',
            profiles=[proto['profiles'][0],proto['parent_profile']], requested_seed_count=512,
            source_rule='Exactly native sample_instance seeds940043000..940043007;development stream;all familiesincluded;freeze before outcomes',
            matched_parent_evidence_reused=0,new_labels_enter_training=False,
            analysis_scope='Independent outcome-blind family confirmation of a frozen stationary motion-gated controller intervention. Two seen eta and known Ring geometry; not unseen-controller learning or cross-scene generalization.',
            discovery_protocol_sha256=sha(motion.OUT/'protocol.json'),
            analysis='Paired16seeds;all16cells;Holm correction. Repeat result or preserve negative; no familyreplacement.')
        write(OUT/'protocol.json',proto);write(OUT/'states.json',states);write(OUT/'pairs.json',pairs)
        write(OUT/'controller_program.json',read(motion.OUT/'controller_program.json'))
        assert sha(OUT/'controller_program.json')==sha(motion.OUT/'controller_program.json')
        proof=read(motion.OUT/'input_identity_proof.json')
        proof.update(min_initial_agent_goal_distance=minimum,remaining_distance_lower_bound=minimum-80*.05*.52)
        write(OUT/'input_identity_proof.json',proof)
        db.execute('INSERT INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXP,'independent_goal_motion_alias_confirmation',str(OUT),sha(OUT/'protocol.json'),sha(__file__),canonical({'source_family_independent':True,'no_model_selection':True})))
    req=[]
    for c in proto['profiles']:
        part=[dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=c['controller_uid'],seed_keys=[canonical({'future_index':k}) for k in range(16)]) for p in pairs]
        req.extend(part);write(OUT/f'planned_{c["name"]}.json',{'requests':part})
    write(OUT/'planned_rollouts.json',{'requests':req})
    write(OUT/'preexisting_numerical_exclusions.json',dict(count=0,records=[]))


def validate():
    from . import state_support
    from shared_rollout_db.src.planner import preflight
    state_support.OUT=OUT;state_support.validate_plan()
    for c in read(OUT/'protocol.json')['profiles']:
        doc=preflight(OUT/f'planned_{c["name"]}.json')
        write(OUT/f'cache_preflight_{c["name"]}.json',doc)
        write(OUT/f'execution_preflight_{c["name"]}.json',doc)


def run(index):
    if index<6:
        motion.OUT,motion.EXP=OUT,EXP
        original=motion.read
        motion.read=lambda p:original(OUT/'cache_preflight_goal_motion.json') if p==OUT/'cache_preflight.json' else original(p)
        motion.run(index)
    else:
        engine.OUT,engine.EXP=OUT,EXP
        engine.run('alt',index-6)


def merge():
    from . import postprocess,goal_motion_analysis
    engine.OUT,engine.EXP=OUT,EXP
    postprocess.run('goal_motion_confirmation')
    goal_motion_analysis.OUT,goal_motion_analysis.EXP=OUT,EXP
    goal_motion_analysis.main()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','validate','run','merge'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='run':run(a.index)
    else:globals()[a.action]()
