"""Outcome-level native-controller slot intervention; source diagnostic only.

All physical agent/goal pairs stay the same. Cyclically relabel the ordered
controller slots. This is NOT a new independent source family. Seed namespaces
are content-identity dependent, so comparisons are independent samples, not
misrepresented as matched random numbers.
"""
import argparse,copy,json
from pathlib import Path
from . import seed_replication as engine
from shared_rollout_db.src.rollout_db import connect,transaction,canonical,uid
from new_benchmark_common.basin_dataset_v1 import content_hash

OUT=Path(__file__).resolve().parent/'permutation_outcomes'
EXP='exp_orthoflow3_rootcause_role_permutation_v1'
read=engine.read;write=engine.write

def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Permutation protocol frozen')
    source=read(engine.OUT/'states.json')[:8];oldpairs=read(engine.OUT/'pairs.json')
    p=copy.deepcopy(read(engine.OUT/'protocol.json'));states=[];pairs=[]
    perm=[1,2,3,0]
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,start_time,metadata_json) VALUES(?,?,?,CURRENT_TIMESTAMP,?)',
            (EXP,'native_controller_slot_outcome_intervention',str(OUT),canonical({'source_only':True})))
        for i,parent in enumerate(source):
            s=copy.deepcopy(parent);original=db.execute('SELECT * FROM state WHERE state_uid=?',(s['uid'],)).fetchone()
            for k in ('positions','velocities','goals'):s['physical'][k]=[s['physical'][k][j] for j in perm]
            s['content_hash']=content_hash(s['physical'])
            s['uid']=uid('state',{'scenario':original['scenario_uid'],'content':s['content_hash']})
            s['alias']+='__controller_slots_cycle1';s['repair_index']=i
            s['provenance'].update(parent_state_uid=parent['uid'],agent_permutation=perm,
                intervention='ordered_controller_slot_relabeling_same_physical_agent_goal_set',
                source_family_preserved=True,RNG_comparison='independent_content_namespaces_not_matched_latents')
            assert s['uid']!=parent['uid'] and s['source_group']==parent['source_group']
            geometry=json.loads(original['goals_geometry_json']);geometry['goals']=s['physical']['goals']
            db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,h0_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?,?)',
                (s['uid'],original['scenario_uid'],s['source_group'],s['content_hash'],canonical(s['physical']),None,canonical(geometry),canonical(s['provenance']),'CONTENT_EXACT'))
            db.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',(original['scenario_uid'],s['alias'],s['uid'],EXP))
            states.append(s)
            for op in oldpairs:
                if op['state_uid']!=parent['uid']:continue
                pair=copy.deepcopy(op);pair.update(state_uid=s['uid'],state_index=i,target_seeds=32,parent_state_uid=parent['uid'])
                pairs.append(pair)
    p.update(experiment_uid=EXP,experiment='native_controller_slot_outcome_intervention',
        source_train_families=8,requested_seed_count=1024,rollout_budget_new_upper_bound=1024,
        source_rule='first8 of outcome-blind replication hash order',agent_permutation=perm,
        statistical_comparison='Independent seed namespaces; same physical multiset, same frozen controller, same eta and safety',
        primary_question='Does a native controller slot relabeling change task-success probability?',
        heldout_generalization_claim=False,analysis='Compare new Q32 to original fresh48 on same physical set; do not infer full h+C alias from action symmetry alone')
    write(OUT/'states.json',states);write(OUT/'pairs.json',pairs);write(OUT/'protocol.json',p)
    requests=[]
    for c in p['profiles']:
        part=[dict(state_uid=s['state_uid'],eta_uid=s['eta_uid'],controller_uid=c['controller_uid'],seed_keys=[canonical({'future_index':k}) for k in range(32)]) for s in pairs]
        write(OUT/f'planned_{c["name"]}.json',{'requests':part});requests.extend(part)
    write(OUT/'planned_rollouts.json',{'requests':requests})
    with connect() as db,transaction(db):
        db.execute('UPDATE experiment SET protocol_hash=?,code_hash=? WHERE experiment_uid=?',(engine.sha(OUT/'protocol.json'),engine.sha(__file__),EXP))
    print(dict(requested=1024,states=8,eta=2,controllers=2))

def configure():engine.OUT=OUT;engine.EXP=EXP

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','filter','run','merge'))
    p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        configure()
        if a.action=='filter':engine.filter_numerical()
        elif a.action=='merge':engine.merge()
        else:engine.run(engine.CONTROLLERS[a.index//6],a.index%6)
