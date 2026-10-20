"""Only official append-only design registration and cache planning."""
from common import *
from shared_rollout_db.src.rollout_db import connect, transaction, canonical, uid

def register(stage, rows, identities):
    prereg(); inserted=[]; tables=['experiment','scenario','state','controller_config','eta']
    exp_path=HERE/'EXPERIMENT_ROW.json'
    if not exp_path.exists():
        write(exp_path,dict(experiment_uid=EXP,name='Multiscene navigation competence pilot',
            path=str(HERE),protocol_hash=sha(HERE/'PREREGISTRATION.md'),code_hash=sha(__file__),
            metadata_json=canonical(dict(round=40))))
    identities=[('experiment',read(exp_path))]+identities
    with connect() as db,transaction(db):
        before={t:[dict(r) for r in db.execute('SELECT * FROM '+t+' ORDER BY 1')] for t in tables}
        keys={t:db.execute('PRAGMA table_info('+t+')').fetchone()['name'] for t in tables}
        for table,value in identities:
            assert table in tables
            key=keys[table]; old=db.execute('SELECT * FROM '+table+' WHERE '+key+'=?',(value[key],)).fetchone()
            if old:
                assert all(old[k]==v for k,v in value.items()),('UID_CONTENT_CONFLICT',table,value[key])
                continue
            db.execute('INSERT OR IGNORE INTO '+table+'('+','.join(value)+') VALUES('+','.join('?' for _ in value)+')',list(value.values()))
            inserted.append(dict(table=table,row=dict(db.execute('SELECT * FROM '+table+' WHERE '+key+'=?',(value[key],)).fetchone())))
        hashes={}
        for table,records in before.items():
            a=hashlib.sha256();b=hashlib.sha256()
            for row in records:
                a.update(canonical(row).encode());b.update(canonical(dict(db.execute('SELECT * FROM '+table+' WHERE '+keys[table]+'=?',(row[keys[table]],)).fetchone())).encode())
            assert a.digest()==b.digest()
            hashes[table]=dict(rows=len(records),before=a.hexdigest(),after=b.hexdigest())
    write(HERE/f'{stage}_INSERTED_ROWS.json',inserted)
    seal(f'{stage}_REGISTRATION.receipt.json',[HERE/f'{stage}_INSERTED_ROWS.json',exp_path],preexisting_row_hashes=hashes)
    write(HERE/f'{stage}_CASES.json',rows)
    manifest=HERE/f'{stage}_PLANNED_ROLLOUTS.json'
    write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in rows]))
    pf=planner.preflight(manifest);write(HERE/f'{stage}_CACHE_PREFLIGHT.json',pf)
    assert not pf['summary']['ambiguous']
    with connect(True) as db:
        missing=[c for c in rows if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() is None]
    # Existing numerical failures remain unknown even when the default planner hides them.
    requested_missing={(v['state_uid'],v['eta_uid'],v['controller_uid'],s) for v in pf['details'] for s in v['missing_seeds']}
    assert all(tuple(c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']) in requested_missing for c in missing)
    write(HERE/f'{stage}_MISSING_CASES.json',missing)
    seal(f'{stage}_PREFLIGHT.receipt.json',[manifest,HERE/f'{stage}_CACHE_PREFLIGHT.json',HERE/f'{stage}_MISSING_CASES.json'],before_submission=True,planned=len(rows),missing=len(missing))
    return missing

def postflight(stage):
    path=HERE/f'{stage}_PLANNED_ROLLOUTS.json'; pf=planner.preflight(path)
    rows=read(HERE/f'{stage}_CASES.json')
    with connect(True) as db:
        absent=[c['rollout_uid'] for c in rows if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() is None]
        unknown=[c['rollout_uid'] for c in rows if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=? AND numerical_failure=1',(c['rollout_uid'],)).fetchone()]
    write(HERE/f'{stage}_POSTFLIGHT.json',dict(planner=pf,absent=absent,unknown_preserved=unknown))
    assert not absent
