"""Freeze source-selected eta contrast and a new, outcome-blind Ring source panel.

Native simulation, seed semantics, journals and atomic SQLite merge are reused.
No held-controller/LOSO target label participates in this experiment.
"""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
from diagnostics.orthoflow3_controller_training_repair_v1 import data as native
from diagnostics.orthoflow3_controller_state_residual_factorial_v1 import pipeline as prior
from diagnostics.orthoflow3_controller_information_probe_v1 import probe
from new_benchmark_common import basin_dataset_v1 as bd
from shared_rollout_db.src.rollout_db import ROOT as DBROOT,canonical,connect,transaction,uid,eta_identity

OUT=Path(__file__).resolve().parent
SRC=prior.OUT
EXP='exp_orthoflow3_source_contrast_interaction_v1'
ETA_INDEX=(10,15)
CONTROLLERS=('alt','second')
SHARDS=6
SALT='source_contrast_interaction_independent_validation_v1'
read=prior.read;write=prior.write

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('frozen protocol already exists')
    old_states=read(SRC/'states.json');old_pairs=read(SRC/'pairs.json');old_proto=read(SRC/'protocol.json')
    dataset=np.load(SRC/'variants/large_20/dataset.npz')
    train_indices=sorted(int(x) for x in set(dataset['state_index'][dataset['split']=='train']))
    assert len(train_indices)==46 and all(old_states[i]['split']=='train' for i in train_indices)
    held=read(probe.OUT/'held_controller_true_t0_v1/states.json')
    used={s['source_group'] for s in old_states+held}
    available=[s for s in bd.parent_state_rows('ring_exchange') if s['split']=='train' and s['source_group'] not in used]
    available.sort(key=lambda s:hashlib.sha256((SALT+'\0'+s['source_group']).encode()).hexdigest())
    assert len(available)>=16
    validation=copy.deepcopy(available[:16])
    for s in validation:s['split']='validation'
    train=copy.deepcopy([old_states[i] for i in train_indices])
    states=train+validation
    assert len(states)==62 and len({s['source_group'] for s in states})==62
    loso_truth=read(OUT.parent/'orthoflow3_cross_scene_zero_shot_v1'/'target_truth.json')
    loso_uids={r['state_uid'] for r in loso_truth}
    assert not ({s['uid'] for s in states}&loso_uids)
    with connect(True) as db:
        loso_groups={r[0] for u in loso_uids for r in db.execute(
            'SELECT source_group FROM state WHERE state_uid=?',(u,)) if r[0]}
    assert not ({s['source_group'] for s in states}&loso_groups)
    assert all(s['physical']['timestep']==0 for s in states)
    for i,s in enumerate(states):s['repair_index']=i
    etas=[(old_pairs[e]['eta_uid'],old_pairs[e]['eta']) for e in ETA_INDEX]
    assert all(eta_identity(e)[0]==u for u,e in etas)
    pairs=[]
    for si,s in enumerate(states):
        for ei,(eu,e) in zip(ETA_INDEX,etas):
            pairs.append({'state_uid':s['uid'],'state_index':si,'split':s['split'],
                'eta_uid':eu,'eta':e,'eta_index':ei,'target_seeds':16})
    profiles=[p for p in old_proto['profiles'] if p['name'] in CONTROLLERS]
    assert [p['name'] for p in profiles]==list(CONTROLLERS)
    requests=[]
    for profile in profiles:
        part=[{'state_uid':p['state_uid'],'eta_uid':p['eta_uid'],'controller_uid':profile['controller_uid'],
               'seed_keys':[canonical({'future_index':k}) for k in range(16)]} for p in pairs]
        write(OUT/f'planned_{profile["name"]}.json',{'requests':part})
        requests.extend(part)
    proto=dict(old_proto)
    proto.update(experiment_uid=EXP,selected_eta_indices=list(ETA_INDEX),profiles=profiles,
        selected_controller_names=list(CONTROLLERS),semantic_execution_revision=native.REVISION,
        source_train_families=46,independent_source_validation_families=16,
        validation_rule='unused parent TRAIN source families, SHA256 salt; no outcome consulted',
        eta_choice_rule='posthoc from OLD source-TRAIN Q4 only: eta10 vs eta15 has both preference directions under second controller and controller-dependent preference change under alt vs second',
        source_train_indices_prior=train_indices,
        source_overlap_held_controller=0,source_overlap_LOSO_target=0,
        audited_LOSO_target_state_uids=len(loso_uids),audited_LOSO_target_source_groups=len(loso_groups),
        requested_seed_count=len(requests)*16,
        input_features='unchanged H20 eta-conditioned physical response, 24 columns',
        rollout_budget_new_upper_bound=len(requests)*16,
        original_files_sha256={k:sha(SRC/k) for k in ('states.json','pairs.json','protocol.json')},
        original_dataset_sha256=sha(SRC/'variants/large_20/dataset.npz'),
        success_semantics_changed=False,generator_modified=False,target_labels_used=False)
    write(OUT/'states.json',states);write(OUT/'pairs.json',pairs)
    write(OUT/'planned_rollouts.json',{'requests':requests})
    write(OUT/'protocol.json',proto)
    write(OUT/'working_state.json',{'phase':'prepared','target_labels_used':False,'new_rollouts':0,
        'source_train_states':46,'new_source_validation_states':16,'generator_modified':False})
    print({'train_states':46,'independent_validation_states':16,'eta_indices':ETA_INDEX,
           'controllers':CONTROLLERS,'requests':len(requests)*16})

def register():
    pre=read(OUT/'cache_preflight.json')['summary']
    assert pre['total_requested']==read(OUT/'protocol.json')['requested_seed_count']
    with connect() as db,transaction(db):
        db.execute('''INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json)
            VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)''',
            (EXP,'source_contrast_interaction_v1',str(OUT),sha(OUT/'protocol.json'),sha(__file__),
             canonical({'source_only':True,'target_labels':0,'generator_modified':False})))

def configure():
    p=read(OUT/'protocol.json')
    assert p['runtime_sha256']==native.old.digest(native.old.__file__)
    assert p['semantic_execution_revision']==native.REVISION
    native.OUT=OUT;native.EXPERIMENT=EXP

def filter_numerical():
    excluded=[]
    with connect(True) as db:
        for name in CONTROLLERS:
            doc=read(OUT/f'cache_preflight_{name}.json')
            for item in doc['details']:
                keep=[]
                for sk in item['missing_seeds']:
                    rr=db.execute('''SELECT numerical_failure,conflict_quarantined,compatibility_quality
                        FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?''',
                        (item['state_uid'],item['eta_uid'],item['controller_uid'],sk)).fetchone()
                    if rr and rr['numerical_failure']:
                        assert not rr['conflict_quarantined'] and rr['compatibility_quality']=='EXACT_REUSE'
                        excluded.append({'state_uid':item['state_uid'],'eta_uid':item['eta_uid'],
                                         'controller_uid':item['controller_uid'],'seed_key':sk})
                    else:keep.append(sk)
                item['missing_seeds']=keep
            doc['summary']['genuinely_missing']-=sum(x['controller_uid']==next(
                p['controller_uid'] for p in read(OUT/'protocol.json')['profiles'] if p['name']==name) for x in excluded)
            write(OUT/f'execution_preflight_{name}.json',doc)
    write(OUT/'preexisting_numerical_exclusions.json',{'count':len(excluded),'records':excluded,
        'policy':'compatible numerical attempts are not rerun and never imputed'})
    print({'preexisting_numerical_excluded':len(excluded)})

def run(controller,shard,shards):
    assert controller in CONTROLLERS and shards==SHARDS and 0<=shard<shards
    configure()
    source_read=native.read
    def intercept(path):
        if Path(path)==OUT/f'cache_preflight_{controller}.json':
            return source_read(OUT/f'execution_preflight_{controller}.json')
        return source_read(path)
    try:
        native.read=intercept;native.run(controller,shard,shards)
    finally:native.read=source_read

def merge():
    configure()
    from diagnostics.orthoflow3_controller_training_repair_v1 import alignment_audit
    old=alignment_audit.main
    try:
        alignment_audit.main=lambda:None;native.merge()
    finally:alignment_audit.main=old
    audit()

def audit():
    configure();p=read(OUT/'protocol.json')
    states={r['uid']:r for r in read(OUT/'states.json')}
    pairs={(r['state_uid'],r['eta_uid']):r for r in read(OUT/'pairs.json')}
    stats={'requested':p['requested_seed_count'],'journal_records':0,'journal_numerical':0,
           'valid_DB_records':0,'numerical_DB_records':0,'unattempted':0,'ambiguous':0,'collision':0}
    with connect(True) as db:
        configs={r['controller_uid']:json.loads(db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',
            (r['controller_uid'],)).fetchone()[0]) for r in p['profiles']}
        for path in sorted((DBROOT/'journals'/EXP).glob(f'{native.REVISION}_*.jsonl')):
            source_uid=uid('src',{'path':str(path.resolve())})
            for number,line in enumerate(path.read_text().splitlines(),1):
                env=json.loads(line);assert env['schema']=='controller_training_repair_v1'
                rec=env['record'];native.validate_record(rec,p,states,pairs,configs)
                rid=uid('roll',{'state':rec['state_uid'],'eta':eta_identity(rec['eta'])[0],
                    'controller':rec['controller_uid'],'seed':canonical({'future_index':rec['future_index']})})
                row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone()
                assert row and row['raw_record_hash']==native.digest(rec)
                assert row['compatibility_quality']=='EXACT_REUSE' and not row['conflict_quarantined']
                assert db.execute('SELECT 1 FROM rollout_source WHERE rollout_uid=? AND source_uid=? AND source_line=?',
                    (rid,source_uid,number)).fetchone()
                stats['journal_records']+=1;stats['journal_numerical']+=int(rec['numerical_failure'])
        for req in read(OUT/'planned_rollouts.json')['requests']:
            for sk in req['seed_keys']:
                rr=db.execute('''SELECT numerical_failure,collision,conflict_quarantined,compatibility_quality
                    FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?''',
                    (req['state_uid'],req['eta_uid'],req['controller_uid'],sk)).fetchone()
                if rr is None:stats['unattempted']+=1
                elif rr['conflict_quarantined'] or rr['compatibility_quality']!='EXACT_REUSE':stats['ambiguous']+=1
                elif rr['numerical_failure']:stats['numerical_DB_records']+=1
                else:stats['valid_DB_records']+=1
                if rr:stats['collision']+=int(rr['collision'])
    write(OUT/'alignment_audit.json',stats)
    assert stats['ambiguous']==0
    return stats

def postflight():
    from shared_rollout_db.src.planner import preflight
    doc=preflight(OUT/'planned_rollouts.json');write(OUT/'cache_postflight.json',doc)
    stats=audit()
    assert stats['unattempted']==0 and stats['valid_DB_records']+stats['numerical_DB_records']==stats['requested']
    assert doc['summary']['genuinely_missing']==stats['numerical_DB_records']
    state=read(OUT/'working_state.json');state.update(phase='postflight_complete',new_rollouts=stats['journal_records'],
        journal_merged=True,valid_DB_records=stats['valid_DB_records'],
        numerical_separate=stats['numerical_DB_records'])
    write(OUT/'working_state.json',state)
    pre=read(OUT/'cache_preflight.json')['summary']
    with connect() as db,transaction(db):
        db.execute('''UPDATE experiment SET end_time=CURRENT_TIMESTAMP,
            reused_rollout_count=?,new_rollout_count=? WHERE experiment_uid=?''',
            (pre['exact_reusable']+pre['partial_reusable'],stats['journal_records'],EXP))
    print({'cache':doc['summary'],'alignment':stats})

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','register','filter','run','merge','audit','postflight'))
    p.add_argument('--controller',choices=CONTROLLERS,default='alt')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=SHARDS)
    p.add_argument('--index',type=int,default=None,help='Slurm task 0..11: controller then shard')
    a=p.parse_args()
    if a.index is not None:
        assert a.action=='run' and 0<=a.index<len(CONTROLLERS)*SHARDS
        a.controller=CONTROLLERS[a.index//SHARDS];a.shard=a.index%SHARDS
    {'prepare':prepare,'register':register,'filter':filter_numerical,
     'run':lambda:run(a.controller,a.shard,a.shards),'merge':merge,'audit':audit,'postflight':postflight}[a.action]()
if __name__=='__main__':main()
