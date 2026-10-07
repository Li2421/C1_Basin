"""Independent continuation-block replication of existing source Q contrasts.

Select source families by a pre-outcome hash; retain the exact controller,
state, eta and seed semantics. This experiment does not create training labels
from frozen LOSO or independent held-controller confirmation states.
"""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from diagnostics.orthoflow3_source_contrast_interaction_v1 import pipeline as parent
from diagnostics.orthoflow3_controller_training_repair_v1 import data as native
from shared_rollout_db.src.rollout_db import connect,transaction,canonical

OUT=Path(__file__).resolve().parent/'seed_replication'
SRC=parent.OUT
EXP='exp_orthoflow3_rootcause_seed_replication_v1'
CONTROLLERS=('alt','second')
SHARDS=6
read=parent.read;write=parent.write;sha=parent.sha

def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Replication protocol frozen')
    source=[s for s in read(SRC/'states.json') if s['split']=='train']
    source.sort(key=lambda s:hashlib.sha256(('c1_rootcause_seed_replication_32_v1\0'+s['source_group']).encode()).hexdigest())
    states=copy.deepcopy(source[:32])
    for i,s in enumerate(states):s['repair_index']=i
    source_pairs={}
    for p in read(SRC/'pairs.json'):source_pairs.setdefault(p['state_uid'],[]).append(p)
    pairs=[]
    for i,s in enumerate(states):
        for p in source_pairs[s['uid']]:
            p=copy.deepcopy(p);p.update(state_index=i,target_seeds=64)
            pairs.append(p)
    protocol=copy.deepcopy(read(SRC/'protocol.json'))
    protocol.update(experiment_uid=EXP,experiment='independent_seed_block_replication',
        selection_rule='SHA256(source_group), salt c1_rootcause_seed_replication_32_v1; first 32 source TRAIN families',
        source_train_families=32,independent_source_validation_families=0,
        historical_standard_seeds=list(range(16)),new_replication_seeds=list(range(16,64)),
        requested_seed_count=32*2*2*64,rollout_budget_new_upper_bound=32*2*2*48,
        target_labels_used=False,model_changes=False,
        primary_question='Do source Q rankings and robust preference changes reproduce on independent continuation blocks?',
        analysis='Compare old Q16 with fresh Q48; original B15 and three fresh B15 blocks reported separately, no replacement of historical labels')
    write(OUT/'states.json',states);write(OUT/'pairs.json',pairs);write(OUT/'protocol.json',protocol)
    requests=[]
    for profile in protocol['profiles']:
        part=[dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=profile['controller_uid'],
                   seed_keys=[canonical({'future_index':k}) for k in range(64)]) for p in pairs]
        write(OUT/f'planned_{profile["name"]}.json',{'requests':part});requests.extend(part)
    write(OUT/'planned_rollouts.json',{'requests':requests})
    write(OUT/'working_state.json',dict(phase='prepared',new_rollouts=0,target_labels_used=False))
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXP,'rootcause_independent_seed_replication',str(OUT),sha(OUT/'protocol.json'),sha(__file__),canonical({'source_only':True,'generator_modified':False})))
    print({'requested':len(requests)*64,'source_families':32})

def configure():
    parent.OUT=OUT;parent.EXP=EXP
    parent.configure()

def run(controller,shard):
    configure();parent.run(controller,shard,SHARDS)

def filter_numerical():
    configure();parent.filter_numerical()

def merge():
    configure()
    # The validated merger accepts arbitrary future_index below target_seeds;
    # there is no fake Q16 conversion or seed renaming.
    from diagnostics.orthoflow3_controller_training_repair_v1 import alignment_audit
    original=alignment_audit.main
    try:
        alignment_audit.main=lambda:None;native.merge()
    finally:alignment_audit.main=original
    protocol=read(OUT/'protocol.json');stats=dict(requested=0,valid=0,numerical=0,missing=0,collision=0,conflict=0)
    with connect(True) as db:
        for request in read(OUT/'planned_rollouts.json')['requests']:
            records={r['seed_key']:r for r in db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',
                (request['state_uid'],request['eta_uid'],request['controller_uid']))}
            for sk in request['seed_keys']:
                stats['requested']+=1;r=records.get(sk)
                if r is None:stats['missing']+=1;continue
                assert r['compatibility_quality']=='EXACT_REUSE'
                stats['conflict']+=r['conflict_quarantined'];stats['collision']+=r['collision']
                stats['numerical' if r['numerical_failure'] else 'valid']+=1
    assert stats['missing']==stats['conflict']==0
    write(OUT/'alignment_audit.json',stats);write(OUT/'working_state.json',dict(phase='merged',**stats))
    print(stats)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','filter','run','merge'))
    p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='filter':filter_numerical()
    elif a.action=='merge':merge()
    else:run(CONTROLLERS[a.index//SHARDS],a.index%SHARDS)
