"""Exact historical import, preflight and single-merger database alignment."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from .design import ROOT,MAIN,FIELD,FP,EXPERIMENT,read,write,freeze,sha,canonical,eta_identity,connect,transaction,uid
from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.rollout_db import ROOT as DBROOT

JOURNALS=DBROOT/'journals'/EXPERIMENT
FLAGS=('success','deadlock','timeout','collision','numerical_failure')


def protocol():
    p=read(ROOT/'protocol.json')
    assert sha(ROOT/'execution.py')==p['execution_sha256']
    if (ROOT/'runtime_dependency_audit.json').exists():
        dep=read(ROOT/'runtime_dependency_audit.json')
        assert sha(FIELD/'source_snapshot.json')==dep['snapshot_sha256']
        for r in dep['files']:assert sha(r['path'])==r['sha256'],r['path']
        for r in dep['main_vs_copied_import_equivalence']:assert sha(r['main_path'])==r['sha256'],r['main_path']
    for scene,profiles in p['controllers'].items():
        for chain,c in profiles.items():
            cc=c['payload']
            assert sha(cc['flow_checkpoint_path'])==cc['flow_checkpoint_sha256']
            for rel,h in cc['action_code_hashes'].items():assert sha(FIELD/rel)==h,rel
            if chain=='FF':assert sha(cc['execution_callable'])==cc['execution_callable_sha256']
    return p


def raw_record(raw,state,chain,origin,source=None):
    p=read(ROOT/'protocol.json');scene=state['scenario'];profile=p['controllers'][scene][chain]
    assert raw['state_uid']==state['uid'] and raw['scenario']==scene
    assert 0<=raw['seed']<16 and int(raw['seed'])==raw['seed']
    assert eta_identity(raw['eta'])[0] in {eta_identity(e)[0] for e in p['eta']}
    if chain=='FF':assert raw['controller_uid']==profile['controller_uid']
    numeric=raw['error'] is not None
    assert not(raw['success'] and (numeric or raw['collision']))
    assert (raw['termination']=='numerical_failure')==numeric
    assert 0<=raw['steps']<=profile['payload']['horizon']
    sk=canonical(dict(future_index=raw['seed'],future_root=2026100403,rng_namespace=state['rng_namespace']))
    return dict(schema='tt_ff_matched_seed_v1',experiment_uid=EXPERIMENT,origin=origin,
        state_uid=state['state_uid'],source_alias=state['uid'],physical_content_hash=state['content_hash'],
        eta_uid=eta_identity(raw['eta'])[0],eta=raw['eta'],controller_uid=profile['controller_uid'],
        controller_payload_sha256=hashlib.sha256(canonical(profile['payload']).encode()).hexdigest(),
        seed_key=sk,chain=chain,scene=scene,split=state['metadata']['split'],
        success=int(raw['success']),deadlock=int(raw['termination']=='deadlock'),
        timeout=int(raw['termination']=='timeout'),collision=int(raw['collision']),numerical_failure=int(numeric),
        episode_length=raw['steps'],outcome=raw['termination'],protocol_sha256=sha(ROOT/'protocol.json'),
        source=source,raw=raw)


def import_history():
    p=protocol();by={s['uid']:s for s in p['states']}
    dest=ROOT/'historical_import.json'
    if dest.exists():
        print(json.dumps({k:v for k,v in read(dest).items() if k!='journals'}));return
    source=FP/'field_rollout.sqlite';con=sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
    phases=('toy_source_screen','toy_source_full','ring_t0_screen','ring_t0_full')
    stats=collections.Counter();records=[];keys=set()
    for row in con.execute('SELECT * FROM continuation WHERE phase IN (?,?,?,?) ORDER BY phase,id',phases):
        assert row['state_uid'] in by,('Source manifest mismatch',row['state_uid'])
        state=by[row['state_uid']];assert state['metadata']['split']!='test'
        raw=json.loads(row['result_json']);assert raw['id']==row['id'] and raw['controller_uid']==row['controller_uid']
        assert raw['state_uid']==row['state_uid'] and raw['seed']==row['seed']
        assert eta_identity(raw['eta'])[0]==eta_identity(json.loads(row['eta_json']))[0]
        assert int(raw['success'])==row['success'] and int(raw['error'] is not None)==row['numerical_unresolved']
        r=raw_record(raw,state,'FF','historical',dict(database=str(source),id=row['id'],phase=row['phase'],raw_sha256=hashlib.sha256(row['result_json'].encode()).hexdigest()))
        records.append(r);stats['source_records']+=1;stats['source_numerical_rows']+=r['numerical_failure']
        keys.add((r['state_uid'],r['eta_uid'],r['controller_uid'],r['seed_key']))
    con.close();assert records
    journal=append_journal(records,EXPERIMENT,'historical_source_exact')
    freeze(dest,dict(**stats,unique_seed_keys=len(keys),journals=[str(journal)],new_rollouts=0,
        source_protocol_hashes=p['source_screens'],all_original_screen_states_preserved=True,
        adaptive_full_extra_seeds_retained_but_not_extra_primary_training_weight=True))
    print(json.dumps(dict(**stats,unique_seed_keys=len(keys),new_rollouts=0)))


def validate(r,p,states):
    assert r['schema']=='tt_ff_matched_seed_v1' and r['experiment_uid']==EXPERIMENT
    assert r['protocol_sha256']==sha(ROOT/'protocol.json') and r['origin'] in ('historical','new')
    s=states[r['state_uid']];profile=p['controllers'][s['scenario']][r['chain']]
    assert r['controller_uid']==profile['controller_uid'] and r['source_alias']==s['uid']
    assert r['physical_content_hash']==s['content_hash'] and r['split']==s['metadata']['split']
    assert r['controller_payload_sha256']==hashlib.sha256(canonical(profile['payload']).encode()).hexdigest()
    rr=raw_record(r['raw'],s,r['chain'],r['origin'],r['source']);assert rr==r
    if r['split']=='test':assert (ROOT/'models_frozen.json').exists(),'TEST execution/import before model freeze'


def merge():
    p=protocol();states={s['state_uid']:s for s in p['states']};stats=collections.Counter()
    with connect() as db:
        for path in sorted(JOURNALS.glob('*.jsonl')):
            source=uid('source',dict(path=str(path),sha256=sha(path)))
            if db.execute('SELECT 1 FROM source_file WHERE source_uid=?',(source,)).fetchone():continue
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            with transaction(db):
                counts=collections.Counter()
                db.execute('INSERT INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen) VALUES(?,?,?,?,?,?,?)',
                    (source,EXPERIMENT,str(path),sha(path),'.jsonl','SEED_EXACT',len(rows)))
                for line,r in enumerate(rows,1):
                    validate(r,p,states);key=(r['state_uid'],r['eta_uid'],r['controller_uid'],r['seed_key'])
                    old=db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?',key).fetchone()
                    rid=old['rollout_uid'] if old else uid('roll',dict(state=key[0],eta=key[1],controller=key[2],seed=key[3]))
                    core=tuple(r[k] for k in FLAGS)
                    if old:
                        if core!=tuple(old[k] for k in FLAGS) or old['episode_length']!=r['episode_length']:
                            db.execute('UPDATE rollout SET conflict_quarantined=1 WHERE rollout_uid=?',(rid,))
                            db.execute('INSERT OR IGNORE INTO conflict(conflict_uid,entity_type,identity_key,existing_json,incoming_json,source_file) VALUES(?,?,?,?,?,?)',
                                (uid('conflict',dict(rollout_uid=rid,incoming=r)),'rollout',rid,canonical(dict(old)),canonical(r),str(path)))
                            counts['ambiguous']+=1;stats['conflict']+=1;continue
                        counts['duplicate']+=1;stats['duplicate']+=1
                    else:
                        db.execute('''INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,
                            success,deadlock,timeout,collision,numerical_failure,episode_length,outcome,experiment_uid,original_source_file,
                            compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                            (rid,*key,key[3],*core,r['episode_length'],r['outcome'],EXPERIMENT,str(path),'EXACT_REUSE',uid('record',r)))
                        counts['imported']+=1;stats[r['origin']+'_inserted']+=1;stats['numerical_inserted']+=r['numerical_failure']
                    db.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)',(rid,source,line))
                db.execute('UPDATE source_file SET rows_imported=?,rows_duplicate=?,rows_ambiguous=? WHERE source_uid=?',
                    (counts['imported'],counts['duplicate'],counts['ambiguous'],source))
            stats['journals']+=1
        newcount=db.execute("SELECT count(*) FROM rollout WHERE experiment_uid=? AND original_source_file NOT LIKE '%historical_source_exact%'",(EXPERIMENT,)).fetchone()[0]
        histcount=db.execute("SELECT count(*) FROM rollout WHERE experiment_uid=? AND original_source_file LIKE '%historical_source_exact%'",(EXPERIMENT,)).fetchone()[0]
        db.execute('UPDATE experiment SET new_rollout_count=?,reused_rollout_count=? WHERE experiment_uid=?',(newcount,histcount,EXPERIMENT));db.commit()
    path=ROOT/'merge_audits'/('merge_'+str(__import__('time').time_ns())+'.json');write(path,dict(stats))
    print(json.dumps(dict(**stats,total_new_attempts=newcount,historical_reused=histcount)))
    assert not stats['conflict'],'Conflict quarantined: stop execution and audit'


def request_manifest(rows):
    g=collections.defaultdict(list)
    for r in rows:g[r['state_uid'],r['eta_uid'],r['controller_uid']].append(r['seed_key'])
    return dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=ss) for (s,e,c),ss in g.items()])


def audit(rows):
    p=protocol();counts=collections.Counter();missing=[];groups=collections.defaultdict(list)
    for r in rows:groups[r['state_uid'],r['eta_uid'],r['controller_uid']].append(r)
    with connect(True) as db:
        for (s,e,c),req in groups.items():
            st=db.execute('SELECT * FROM state WHERE state_uid=?',(s,)).fetchone();ct=db.execute('SELECT * FROM controller_config WHERE controller_uid=?',(c,)).fetchone()
            assert st and ct and st['scenario_uid']==ct['scenario_uid'] and st['identity_quality']=='CONTENT_EXACT' and ct['compatibility_quality']=='EXACT_PROFILE'
            profile=p['controllers'][req[0]['scene']][req[0]['chain']];assert ct['config_json']==canonical(profile['payload'])
            found={r['seed_key']:r for r in db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(s,e,c))}
            all_present=all(r['seed_key'] in found and not found[r['seed_key']]['numerical_failure'] for r in req)
            counts['pairs']+=1;counts['complete_pairs']+=int(all_present)
            for r in req:
                counts['requested']+=1;old=found.get(r['seed_key'])
                if old:
                    assert not old['conflict_quarantined'] and old['compatibility_quality']=='EXACT_REUSE'
                    if old['numerical_failure']:counts['numerical_unknown_not_rerun']+=1
                    else:counts['exact_reuse' if all_present else 'partial_reuse']+=1
                else:counts['truly_missing']+=1;missing.append(r)
        counts['aggregate_reuse']=0
    return dict(counts),missing


def preflight(stage):
    rows=read(ROOT/'cases.json')
    if stage=='source':rows=[r for r in rows if r['split']!='test']
    elif stage=='test':
        assert (ROOT/'models_frozen.json').exists();rows=[r for r in rows if r['split']=='test']
    else:assert stage=='all'
    folder=ROOT/('preflight_'+stage);folder.mkdir(exist_ok=True)
    freeze(folder/'planned_rollouts.json',request_manifest(rows))
    with (folder/'cli_stdout.json').open('w') as stdout:
        subprocess.run([sys.executable,'-m','shared_rollout_db.plan','--manifest',str(folder/'planned_rollouts.json'),
            '--output',str(folder/'cache_preflight.json')],cwd=MAIN,check=True,stdout=stdout)
    counts,missing=audit(rows)
    write(folder/'exact_identity_audit.json',dict(summary=counts,planner_empty_cache_status_note='Legacy planner calls no compatible cached record INCOMPATIBLE. Explicit canonical metadata are validated here; no signature relaxed.',protocol_sha256=sha(ROOT/'protocol.json')))
    write(folder/'missing_cases.json',missing)
    print(json.dumps(dict(stage=stage,**counts)))


def batches(stage):
    assert stage in ('source','test');base=ROOT/('preflight_'+stage)
    missing=read(base/'missing_cases.json')
    missing.sort(key=lambda r:(r['scene'],r['chain'],r['state_uid'],r['eta_index'],r['seed']))
    result=[]
    for i,start in enumerate(range(0,len(missing),4096)):
        dest=ROOT/'batches'/f'{stage}{i}';dest.mkdir(parents=True,exist_ok=True)
        rr=missing[start:start+4096];freeze(dest/'cases.json',rr);freeze(dest/'planned_rollouts.json',request_manifest(rr))
        with (dest/'cli_preflight_stdout.json').open('w') as f:
            subprocess.run([sys.executable,'-m','shared_rollout_db.plan','--manifest',str(dest/'planned_rollouts.json'),'--output',str(dest/'cache_preflight.json')],cwd=MAIN,stdout=f,check=True)
        cc,mm=audit(rr);assert len(mm)==len(rr),'Recompute missing plan before scheduling'
        freeze(dest/'identity_preflight.json',dict(summary=cc,protocol_sha256=sha(ROOT/'protocol.json'),cases_sha256=sha(dest/'cases.json')))
        result.append(dict(batch=f'{stage}{i}',**cc))
    freeze(base/'batches.json',result);print(json.dumps(result))


def postflight(batch):
    base=ROOT/'batches'/batch;counts,missing=audit(read(base/'cases.json'))
    with (base/'cli_postflight_stdout.json').open('w') as f:
        subprocess.run([sys.executable,'-m','shared_rollout_db.plan','--manifest',str(base/'planned_rollouts.json'),'--output',str(base/'cache_postflight.json')],cwd=MAIN,stdout=f,check=True)
    write(base/'postflight_audit.json',dict(summary=counts,all_attempted_records_in_global_db=not missing,
        numerical_retained_as_unknown=True,missing=missing))
    print(json.dumps(dict(batch=batch,**counts)));assert not missing


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=('import','merge','preflight','batches','postflight'));a.add_argument('--stage',default='source');a.add_argument('--batch');q=a.parse_args()
    if q.action=='import':import_history()
    elif q.action=='merge':merge()
    elif q.action=='preflight':preflight(q.stage)
    elif q.action=='batches':batches(q.stage)
    else:postflight(q.batch)
