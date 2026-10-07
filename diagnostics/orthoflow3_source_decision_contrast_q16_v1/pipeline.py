"""Only source TRAIN, Ring's third known controller and two exact eta.

The native executor, journal writer, semantic validator and atomic merger are
reused. This file only narrows the frozen plan and changes provenance UID.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from diagnostics.orthoflow3_controller_training_repair_v1 import data as native
from diagnostics.orthoflow3_controller_state_residual_factorial_v1 import pipeline as source
from shared_rollout_db.src.rollout_db import ROOT as DBROOT, canonical, connect, transaction, uid, eta_identity

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SRC=source.OUT
EXPERIMENT='exp_orthoflow3_source_decision_contrast_q16_v1'
REVISION=native.REVISION
CONTROLLER='second'
ETA_INDEX=(1,3)
SHARDS=6

def read(path):return json.loads(Path(path).read_text())
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Frozen protocol exists; will not reselect states/eta')
    sd=dict(np.load(SRC/'variants/large_20/dataset.npz'))
    states=read(SRC/'states.json');all_pairs=read(SRC/'pairs.json');base=read(SRC/'protocol.json')
    selected=[]
    for pair in all_pairs:
        ix=pair['state_index']*16+pair['eta_index']
        assert all_pairs[ix]==pair
        if sd['split'][ix]=='train' and pair['eta_index'] in ETA_INDEX:
            assert pair['split']=='train' and pair['target_seeds']==4
            p=dict(pair);p['target_seeds']=16;selected.append(p)
    assert len(selected)==46*2
    assert len({p['state_uid'] for p in selected})==46
    assert all(states[p['state_index']]['uid']==p['state_uid'] for p in selected)
    assert all(p['state_uid'] not in {states[j]['uid'] for j in range(len(states)) if sd['split'][j*16]=='validation'} for p in selected)
    profile=next(c for c in base['profiles'] if c['name']==CONTROLLER)
    assert len({p['eta_uid'] for p in selected})==2
    assert all(eta_identity(p['eta'])[0]==p['eta_uid'] for p in selected)
    requests=[{'state_uid':p['state_uid'],'eta_uid':p['eta_uid'],'controller_uid':profile['controller_uid'],
               'seed_keys':[canonical({'future_index':k}) for k in range(16)]} for p in selected]
    protocol=dict(base)
    protocol.update(experiment_uid=EXPERIMENT,
        purpose='source-TRAIN only decision contrast label precision; no held-controller/LOSO target access',
        frozen_parent_sha256={name:sha(SRC/name) for name in ('protocol.json','states.json','pairs.json')},
        frozen_source_dataset_sha256=sha(SRC/'variants/large_20/dataset.npz'),
        source_state_indices=sorted({p['state_index'] for p in selected}),
        source_eta_indices=list(ETA_INDEX),selected_controller_uid=profile['controller_uid'],
        source_pair_count=len(selected),requested_seed_count=len(selected)*16,
        known_existing_standard_seeds=list(range(4)),planned_seeds=list(range(16)),
        semantic_execution_revision=REVISION,
        excluded_split=['validation','held_controller','LOSO_target'],
        max_additional_continuations_before_preflight=46*2*12,
        no_hyperparameter_or_model_selection_from_target=True,
        generator_modified=False,success_semantics_modified=False,
        training_comparison='old Q4 vs new Q16 normalized to weight4 vs old Q4 weight16; matched seeds/steps/splits',
        frozen_training_hyperparameters={'steps':1500,'batch_pairs':32,'learning_rate':0.0008,'weight_decay':0.0001,
                                          'seeds':[17,23,41],'checkpoints':'inner TRAIN-family validation only'})
    write(OUT/'states.json',states)
    write(OUT/'pairs.json',selected)
    write(OUT/'protocol.json',protocol)
    write(OUT/'planned_rollouts.json',{'requests':requests})
    write(OUT/'working_state.json',{'phase':'prepared','requested':len(selected)*16,'new_rollouts':0,
        'target_labels_opened':False,'generator_modified':False,'journal_merged':False})
    print({'source_TRAIN_families':46,'pairs':len(selected),'requested':len(selected)*16,
           'expected_old_standard_seeds':len(selected)*4,'new_upper_bound':len(selected)*12})

def register():
    assert (OUT/'cache_preflight.json').exists()
    proto=read(OUT/'protocol.json')
    assert read(OUT/'cache_preflight.json')['summary']['total_requested']==proto['requested_seed_count']
    with connect() as db,transaction(db):
        db.execute('''INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json)
           VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)''',
            (EXPERIMENT,'source_decision_contrast_q16_v1',str(OUT),sha(OUT/'protocol.json'),sha(__file__),
             canonical({'source_only':True,'target_labels':0,'native_executor':REVISION,'generator_modified':False})))

def configure():
    assert read(OUT/'protocol.json')['runtime_sha256']==native.old.digest(native.old.__file__)
    assert read(OUT/'protocol.json')['semantic_execution_revision']==native.REVISION
    native.OUT=OUT;native.EXPERIMENT=EXPERIMENT

def filter_numerical():
    original=read(OUT/'cache_preflight.json')
    profile=read(OUT/'protocol.json')['selected_controller_uid']
    excluded=[]
    with connect(True) as db:
        for row in original['details']:
            assert row['controller_uid']==profile
            keep=[]
            for seed in row['missing_seeds']:
                rr=db.execute('''SELECT numerical_failure,conflict_quarantined,compatibility_quality FROM rollout
                    WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?''',
                    (row['state_uid'],row['eta_uid'],profile,seed)).fetchone()
                if rr and rr['numerical_failure']:
                    assert rr['compatibility_quality']=='EXACT_REUSE' and not rr['conflict_quarantined']
                    excluded.append({'state_uid':row['state_uid'],'eta_uid':row['eta_uid'],'seed_key':seed})
                else:keep.append(seed)
            row['missing_seeds']=keep
    original['summary']['genuinely_missing']-=len(excluded)
    write(OUT/'execution_preflight_second.json',original)
    write(OUT/'preexisting_numerical_exclusions.json',{'count':len(excluded),'records':excluded,
        'policy':'do not repeat a compatible numerical outcome or impute success'})
    print({'planned_genuine_missing':read(OUT/'cache_preflight.json')['summary']['genuinely_missing'],
           'preexisting_numerical_excluded':len(excluded),'to_execute':original['summary']['genuinely_missing']})

def run(shard,shards):
    assert shards==SHARDS and 0<=shard<shards
    configure()
    assert (OUT/'execution_preflight_second.json').exists()
    original=native.read
    def intercept(path):
        if Path(path)==OUT/'cache_preflight_second.json':return original(OUT/'execution_preflight_second.json')
        return original(path)
    try:
        native.read=intercept
        native.run(CONTROLLER,shard,shards)
    finally:native.read=original

def merge():
    configure()
    from diagnostics.orthoflow3_controller_training_repair_v1 import alignment_audit
    original=alignment_audit.main
    try:
        alignment_audit.main=lambda:None
        native.merge()
    finally:alignment_audit.main=original
    audit()

def audit():
    configure()
    pairs=read(OUT/'pairs.json');states={s['uid']:s for s in read(OUT/'states.json')}
    profile=read(OUT/'protocol.json')['selected_controller_uid']
    bykey={(p['state_uid'],p['eta_uid']):p for p in pairs}
    stats={'requested':len(pairs)*16,'journal_records':0,'journal_numerical':0,'DB_records':0,
           'DB_numerical':0,'missing_requests':0,'ambiguous_requests':0,'conflicts':0,'existing_reused':0,
           'new_experiment_records':0}
    with connect(True) as db:
        configs={profile:json.loads(db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(profile,)).fetchone()[0])}
        journalpaths=sorted((DBROOT/'journals'/EXPERIMENT).glob(f'{REVISION}_{CONTROLLER}_shard*_*.jsonl'))
        for path in journalpaths:
            source_uid=uid('src',{'path':str(path.resolve())})
            for number,line in enumerate(path.read_text().splitlines(),1):
                env=json.loads(line);assert env['schema']=='controller_training_repair_v1'
                rec=env['record']
                native.validate_record(rec,read(OUT/'protocol.json'),states,bykey,configs)
                eu=eta_identity(rec['eta'])[0];sk=canonical({'future_index':rec['future_index']})
                ruid=uid('roll',{'state':rec['state_uid'],'eta':eu,'controller':profile,'seed':sk})
                row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(ruid,)).fetchone()
                assert row and row['raw_record_hash']==native.digest(rec)
                assert not row['conflict_quarantined'] and row['compatibility_quality']=='EXACT_REUSE'
                assert all(row[k]==rec[k] for k in ('success','deadlock','timeout','collision','numerical_failure','episode_length'))
                assert db.execute('SELECT 1 FROM rollout_source WHERE rollout_uid=? AND source_uid=? AND source_line=?',
                                  (ruid,source_uid,number)).fetchone()
                stats['journal_records']+=1;stats['journal_numerical']+=int(rec['numerical_failure'])
        for req in read(OUT/'planned_rollouts.json')['requests']:
            for seed in req['seed_keys']:
                row=db.execute('''SELECT numerical_failure,conflict_quarantined,compatibility_quality,experiment_uid
                    FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?''',
                    (req['state_uid'],req['eta_uid'],profile,seed)).fetchone()
                if row is None:stats['missing_requests']+=1
                elif row['conflict_quarantined'] or row['compatibility_quality']!='EXACT_REUSE':stats['ambiguous_requests']+=1
                elif row['numerical_failure']:stats['DB_numerical']+=1
                else:stats['DB_records']+=1
                if row and row['experiment_uid']==EXPERIMENT:stats['new_experiment_records']+=1
                elif row:stats['existing_reused']+=1
    assert stats['journal_records']+stats['existing_reused']==stats['requested'] or stats['missing_requests']>0
    write(OUT/'alignment_audit.json',stats)
    if stats['ambiguous_requests']:raise RuntimeError('Incompatible/conflicted DB records; stop')
    return stats

def postflight():
    from shared_rollout_db.src.planner import preflight
    result=preflight(OUT/'planned_rollouts.json')
    write(OUT/'cache_postflight.json',result)
    stats=audit();expected=stats['DB_numerical']
    assert stats['missing_requests']==0 and stats['ambiguous_requests']==0
    assert result['summary']['genuinely_missing']==expected
    assert stats['DB_records']+stats['DB_numerical']==stats['requested']
    status=read(OUT/'working_state.json')
    status.update(phase='postflight_complete',new_rollouts=stats['journal_records'],
                  journal_merged=True,cache_exact_reusable=stats['DB_records'],numerical_separate=expected)
    write(OUT/'working_state.json',status)
    with connect() as db,transaction(db):
        db.execute('UPDATE experiment SET end_time=CURRENT_TIMESTAMP,new_rollout_count=? WHERE experiment_uid=?',
                   (stats['journal_records'],EXPERIMENT))
    print({'cache':result['summary'],'alignment':stats})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('prepare','register','filter','run','merge','audit','postflight'))
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=SHARDS)
    a=ap.parse_args()
    f={'prepare':prepare,'register':register,'filter':filter_numerical,
       'run':lambda:run(a.shard,a.shards),'merge':merge,'audit':audit,'postflight':postflight}[a.action]
    f()
if __name__=='__main__':main()
