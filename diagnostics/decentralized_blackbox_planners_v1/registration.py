"""Append-only design metadata, with complete preservation checks."""
from core import *
from shared_rollout_db.src.rollout_db import connect,transaction,canonical,uid,eta_identity
class IdentityConflict(Exception):pass
TABLES={'scenario':'scenario_uid','experiment':'experiment_uid','state':'state_uid','controller_config':'controller_uid','eta':'eta_uid'}
def snapshot(db):
    return {t:{r[k]:hashlib.sha256(canonical(dict(r)).encode()).hexdigest() for r in db.execute('SELECT * FROM '+t)} for t,k in TABLES.items()}
def counts(db):return {r[0]:r[1] for r in db.execute('SELECT experiment_uid,count(*) FROM rollout GROUP BY experiment_uid')}
def register(label,rows):
    dest=HERE/'registration'/label;assert not dest.exists();dest.mkdir(parents=True)
    with connect() as db,transaction(db):
        before=snapshot(db);beforecounts=counts(db);inserted=[];existing=[]
        write(dest/'BEFORE.json',dict(utc=utc(),row_hashes=before,rollout_counts=beforecounts))
        for table,payload in rows:
            assert table in TABLES;key=TABLES[table];old=db.execute(f'SELECT * FROM {table} WHERE {key}=?',(payload[key],)).fetchone()
            if old:
                if not all(old[k]==v for k,v in payload.items()):raise IdentityConflict(f'Identity collision {table} {payload[key]}')
                existing.append(dict(table=table,uid=payload[key]));continue
            keys=list(payload);sql=f"INSERT OR IGNORE INTO {table} ({','.join(keys)}) VALUES ({','.join('?' for k in keys)})";db.execute(sql,[payload[k] for k in keys]);found=db.execute(f'SELECT * FROM {table} WHERE {key}=?',(payload[key],)).fetchone();assert found is not None
            assert all(found[k]==v for k,v in payload.items());inserted.append(dict(table=table,row=dict(found)))
        after=snapshot(db);aftercounts=counts(db)
        assert all(all(after[t].get(k)==v for k,v in old.items()) for t,old in before.items())
        assert beforecounts==aftercounts,'Existing experiment rollout counts changed during registration'
        write(dest/'INSERTED_ROWS.json',inserted);write(dest/'AFTER.json',dict(utc=utc(),row_hashes=after,rollout_counts=aftercounts));write(dest/'RECEIPT.json',dict(utc=utc(),passed=True,inserted=len(inserted),already_present=len(existing),inserted_rows_sha256=sha(dest/'INSERTED_ROWS.json'),existing_rows_unchanged=True,rollout_counts_unchanged=True))
    return inserted

def normalized_state(s):
    # Official exact-journal schema compares precisely these three physical fields.
    physical={k:s['physical'][k] for k in ['positions','velocities','goals']}
    assert s['physical'].get('timestep',0)==0
    content=uid('content',dict(physical=physical,reset='empty monitor true-t0'))
    return dict(s,original_state_uid=s['state_uid'],physical=physical,content_hash=content,state_uid=uid('state',dict(scenario=s['scenario_uid'],content=content)))
def state_row(s):return ('state',dict(state_uid=s['state_uid'],scenario_uid=s['scenario_uid'],source_group=s['family'],content_hash=s['content_hash'],physical_state_json=canonical(s['physical']),provenance_json=canonical(dict(**s['metadata'],original_state_uid=s['original_state_uid'],reset='empty monitor true-t0',representation='physical3_without_redundant_timestep0')),identity_quality='CONTENT_EXACT'))
def exp_row(phase):
    return ('experiment',dict(experiment_uid='exp_teacher_student_macflow_v1d_'+phase,name='Round30 '+phase,path=str(HERE/phase),protocol_hash=sha(HERE/'PREREGISTRATION.md'),code_hash=sha(__file__),metadata_json=canonical(dict(preregistered=True,phase=phase))))
