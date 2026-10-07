"""Frozen source-only H20/H80 by family coverage experiment.

Reuses the validated Ring source executor; new success records use identical
controller/state/eta/seed semantics. H80 is a derived input, not a task label.
"""
from __future__ import annotations
import argparse,copy,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import jax
from diagnostics.orthoflow3_controller_training_repair_v1 import data as old
from diagnostics.orthoflow3_controller_information_probe_v1 import probe
from new_benchmark_common import basin_dataset_v1 as bd
from shared_rollout_db.src.rollout_db import canonical,connect,transaction

OUT=Path(__file__).resolve().parent
OLD=old.OUT
EXP='exp_orthoflow3_controller_state_residual_factorial_v1'
SALT='source_state_residual_factorial_v1_frozen_before_outcomes'
CONTROLLERS=old.CONTROLLERS

def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def hashfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Frozen; refusing to redo selection')
    original=read(OLD/'protocol.json');existing=read(OLD/'states.json')
    held=read(probe.OUT/'held_controller_true_t0_v1/states.json')
    used={s['source_group'] for s in existing+held}
    pool=bd.parent_state_rows('ring_exchange')
    states=[copy.deepcopy(s) for s in existing if s['split']=='train']
    for split,count in (('train',24),('validation',8)):
        candidates=[s for s in pool if s['split']==split and s['source_group'] not in used]
        candidates.sort(key=lambda s:hashlib.sha256((SALT+'\0'+s['source_group']).encode()).hexdigest())
        assert len(candidates)>=count
        states.extend(copy.deepcopy(candidates[:count]))
    assert len(states)==56 and len({s['source_group'] for s in states})==56
    assert len({s['uid'] for s in states}&{s['uid'] for s in held})==0
    assert all(s['physical']['timestep']==0 for s in states)
    for i,s in enumerate(states):s['repair_index']=i
    source=read(OLD/'pairs.json')[:16]
    etas=[(p['eta_uid'],p['eta']) for p in source]
    assert len({r[0] for r in etas})==16
    pairs=[{'state_uid':s['uid'],'state_index':s['repair_index'],'split':s['split'],
            'eta_uid':eu,'eta':eta,'eta_index':j,'target_seeds':4 if s['split']=='train' else 16}
           for s in states for j,(eu,eta) in enumerate(etas)]
    req=[]
    for p in original['profiles']:
        part=[{'state_uid':r['state_uid'],'eta_uid':r['eta_uid'],'controller_uid':p['controller_uid'],
               'seed_keys':[canonical({'future_index':k}) for k in range(r['target_seeds'])]}
              for r in pairs]
        write(OUT/f'planned_{p["name"]}.json',{'requests':part})
        req.extend(part)
    write(OUT/'states.json',states);write(OUT/'pairs.json',pairs)
    write(OUT/'planned_rollouts.json',{'requests':req})
    train_old={s['source_group'] for s in existing if s['split']=='train'}
    write(OUT/'protocol.json',{
        'experiment_uid':EXP,'schema':'source_state_residual_factorial_v1',
        'profiles':original['profiles'],'base_controller_uid':original['base_controller_uid'],
        'base_flow_sha256':original['base_flow_sha256'],
        'environment_fingerprint':original['environment_fingerprint'],
        'runtime_sha256':original['runtime_sha256'],'source_initial_train_families':24,
        'new_train_families':24,'new_validation_families':8,'source_eta_count':16,
        'source_families_all_distinct':True,'previous_source_VAL_excluded':6,
        'held_controller_families_excluded':len(held),
        'state_selection_rule':SALT+' sorted SHA256 of source_group, separately within TRAIN and validation',
        'pair_design':'same 16 exact eta for all states under three frozen controllers',
        'train_seeds':[0,1,2,3],'validation_seeds':list(range(16)),
        'H20_seconds':1.,'H80_seconds':4.,'full_task_seconds':35.,
        'H80_probe_roots':[2026100417,2026100499],
        'controller_context_modeling':'identical pure observed-count NLL and entity encoder, change only input context horizon',
        'frozen_source_validation':'8 previously unopened parent validation families',
        'held_target_labels_opened':False,'generator_modified':False,
        'new_success_semantics':False,'maximum_new_continuations':10752,
        'batch_cap':4096,'preflight_required':True,
        'numeric_failures':'record and exclude; never impute Q16 or repeat failed same seed',
        'source_state_uid_overlap':0,'source_group_overlap':0,
        'archive_hashes':{'old_protocol':hashfile(OLD/'protocol.json'),
                          'old_source_states':hashfile(OLD/'states.json'),
                          'held_controller_states':hashfile(probe.OUT/'held_controller_true_t0_v1/states.json')},
        'runtime_code_hash':hashfile(__file__)})
    with connect() as db,transaction(db):
        db.execute('''INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json)
            VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)''',
            (EXP,'source_state_residual_factorial_v1',str(OUT),hashfile(OUT/'protocol.json'),
             hashfile(__file__),canonical({'source_only':True,'target_labels':0,'generator_modified':False})))
    print({'families':len(states),'TRAIN':48,'VAL':8,'pairs':len(pairs),'requested':len(req)*4+3*8*16*12})

def configure():
    old.OUT=OUT;old.EXPERIMENT=EXP
    assert old.REVISION=='native_float32_v2'
    assert read(OUT/'protocol.json')['runtime_sha256']==old.old.digest(old.old.__file__)

def physical():
    configure();old.physical()

def filter_preflight():
    """Planner counts existing numerical attempts as missing: never resubmit them."""
    record=[]
    with connect(True) as db:
        for controller in CONTROLLERS:
            source=read(OUT/f'cache_preflight_{controller}.json')
            filtered=copy.deepcopy(source)
            dropped=0
            for part in filtered['details']:
                keep=[]
                for sk in part['missing_seeds']:
                    rr=db.execute('''SELECT numerical_failure,conflict_quarantined,compatibility_quality,experiment_uid
                        FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?''',
                        (part['state_uid'],part['eta_uid'],part['controller_uid'],sk)).fetchone()
                    if rr and rr['numerical_failure']:
                        assert rr['compatibility_quality']=='EXACT_REUSE' and not rr['conflict_quarantined']
                        dropped+=1
                        record.append({'controller':controller,'state_uid':part['state_uid'],
                                       'eta_uid':part['eta_uid'],'seed_key':sk,
                                       'source_experiment_uid':rr['experiment_uid'],
                                       'reason':'compatible numerical failure already executed, never re-run or impute'})
                    else:keep.append(sk)
                part['missing_seeds']=keep
            filtered['summary']['genuinely_missing']-=dropped
            write(OUT/f'execution_preflight_{controller}.json',filtered)
    write(OUT/'preexisting_numerical_exclusions.json',{
        'original_global_preflight_immutable':True,'already_executed_numerical_count':len(record),
        'excluded_from_new_jobs':record})
    print({'preexisting_numerical':len(record),'actual_to_execute':read(OUT/'cache_preflight.json')['summary']['genuinely_missing']-len(record)})

def run(controller,shard,shards):
    configure()
    original=old.read
    execute=OUT/f'execution_preflight_{controller}.json'
    assert execute.exists(), 'Filter pre-existing numerical attempts before submission'
    def source(path):
        if Path(path)==OUT/f'cache_preflight_{controller}.json':return original(execute)
        return original(path)
    try:
        old.read=source
        old.run(controller,shard,shards)
    finally:old.read=original

def merge():
    configure()
    # The original merger audits its own old experiment after committing.
    # Reuse its exact validated atomic insert; run our independent audit next.
    from diagnostics.orthoflow3_controller_training_repair_v1 import alignment_audit as aa
    original=aa.main
    try:
        aa.main=lambda:None
        old.merge()
    finally:aa.main=original
    audit()

def audit():
    configure();protocol=read(OUT/'protocol.json')
    states={s['uid']:s for s in read(OUT/'states.json')}
    pairs={(p['state_uid'],p['eta_uid']):p for p in read(OUT/'pairs.json')}
    stats=Counter();from shared_rollout_db.src.rollout_db import ROOT as DBROOT,uid,eta_identity
    with connect(True) as db:
        payload={p['controller_uid']:json.loads(db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(p['controller_uid'],)).fetchone()[0]) for p in protocol['profiles']}
        for path in sorted((DBROOT/'journals'/EXP).glob(f'{old.REVISION}_*.jsonl')):
            source=uid('src',{'path':str(path.resolve())})
            for line_number,line in enumerate(path.read_text().splitlines(),1):
                env=json.loads(line);assert env['schema']=='controller_training_repair_v1'
                r=env['record'];old.validate_record(r,protocol,states,pairs,payload)
                eu=eta_identity(r['eta'])[0];sk=canonical({'future_index':r['future_index']})
                rid=uid('roll',{'state':r['state_uid'],'eta':eu,'controller':r['controller_uid'],'seed':sk})
                row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone()
                if row is None:stats['awaiting_merger']+=1;continue
                assert row['raw_record_hash']==old.digest(r)
                assert row['conflict_quarantined']==0 and row['compatibility_quality']=='EXACT_REUSE'
                for k in ('success','deadlock','timeout','collision','numerical_failure','episode_length'):assert row[k]==r[k]
                assert db.execute('SELECT 1 FROM rollout_source WHERE rollout_uid=? AND source_uid=? AND source_line=?',(rid,source,line_number)).fetchone()
                stats['journal_DB_aligned']+=1
                stats['numerical_separate']+=int(r['numerical_failure'])
                stats['collision']+=int(r['collision'])
        for request in read(OUT/'planned_rollouts.json')['requests']:
            for sk in request['seed_keys']:
                rr=db.execute('SELECT numerical_failure,conflict_quarantined,compatibility_quality FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?',
                    (request['state_uid'],request['eta_uid'],request['controller_uid'],sk)).fetchone()
                if rr is None:stats['not_attempted']+=1
                elif rr['conflict_quarantined'] or rr['compatibility_quality']!='EXACT_REUSE':stats['invalid']+=1
                elif rr['numerical_failure']:stats['numerical_not_imputed']+=1
                else:stats['exact_reusable']+=1
    write(OUT/'alignment_audit.json',dict(stats))
    if stats['invalid']:raise RuntimeError('Invalid canonical records; stop')
    return stats

def postflight():
    from shared_rollout_db.src.planner import preflight
    configure()
    results=preflight(OUT/'planned_rollouts.json')
    write(OUT/'cache_postflight.json',results)
    stats=audit()
    assert stats['not_attempted']==0
    assert results['summary']['genuinely_missing']==stats['numerical_not_imputed']
    print({'summary':results['summary'],'alignment':dict(stats)})

def main():
    pa=argparse.ArgumentParser();pa.add_argument('action',choices=['prepare','physical','filter_preflight','run','merge','audit','postflight'])
    pa.add_argument('--controller',choices=CONTROLLERS,default='base')
    pa.add_argument('--shard',type=int,default=0);pa.add_argument('--shards',type=int,default=6)
    a=pa.parse_args();{'prepare':prepare,'physical':physical,'filter_preflight':filter_preflight,'run':lambda:run(a.controller,a.shard,a.shards),
                         'merge':merge,'audit':audit,'postflight':postflight}[a.action]()
if __name__=='__main__':main()
