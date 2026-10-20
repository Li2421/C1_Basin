"""Official append-only identity registration and sealed exact preflight."""
from common import *
from shared_rollout_db.src.rollout_db import uid,canonical,eta_identity,transaction


def register(stage,p,rows,profiles):
    prereg();inserted=[];tables=('experiment','state','controller_config','eta','scenario')
    with rollout_db.connect() as db,transaction(db):
        keys={t:db.execute('PRAGMA table_info('+t+')').fetchone()['name'] for t in tables}
        def digest_table(t,skip=()):
            ignored=set(skip);digest=hashlib.sha256();count=0
            for r in db.execute('SELECT * FROM '+t+' ORDER BY 1'):
                if r[keys[t]] in ignored:continue
                digest.update(canonical(dict(r)).encode());count+=1
            return dict(rows=count,sha256=digest.hexdigest())
        before={t:digest_table(t) for t in tables};added={t:set() for t in tables}
        def ins(t,v):
            key=keys[t];old=db.execute('SELECT * FROM '+t+' WHERE '+key+'=?',(v[key],)).fetchone()
            if old is not None:
                assert all(old[k]==x for k,x in v.items()),('UID_CONTENT_CONFLICT',t,v[key]);return
            db.execute('INSERT OR IGNORE INTO '+t+'('+','.join(v)+') VALUES('+','.join('?' for _ in v)+')',list(v.values()))
            added[t].add(v[key]);inserted.append(dict(table=t,row=dict(db.execute('SELECT * FROM '+t+' WHERE '+key+'=?',(v[key],)).fetchone())))
        identity=HERE/'EXPERIMENT_REGISTERED_IDENTITY.json'
        if not identity.exists():write(identity,dict(experiment_uid=EXP,name='Cross-type 6 shared correction and controller information',path=str(HERE),
            protocol_hash=sha(HERE/'PREREGISTRATION.md'),code_hash=sha(__file__),metadata_json=canonical(dict(series='crosstype',round='6'))))
        ins('experiment',read(identity))
        for s in p['states']:
            ins('state',dict(state_uid=s['state_uid'],scenario_uid=s['scenario_uid'],source_group=s.get('family',s['uid']),content_hash=s['content_hash'],
                physical_state_json=canonical(s['transport_physical']),goals_geometry_json=canonical(p['native_manifest']['scenario_config']),
                provenance_json=canonical(dict(original_state_uid=s['original_state_uid'],original_physical=s['original_physical'],original_uid=s['uid'],rng_namespace=s['rng_namespace'],future_root=2026100403)),identity_quality='CONTENT_EXACT'))
        for cid,record in profiles.items():
            v=record['payload'];assert cid==record['controller_uid']
            ins('controller_config',dict(controller_uid=cid,scenario_uid=v['scenario_uid'],flow_checkpoint_sha256=v.get('flow_checkpoint_sha256'),
                orthoflow3_sha256=v['orthoflow3_sha256'],safety_config_hash=uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),
                horizon=str(v['horizon']),dt=str(v['dt']),success_semantics_version=v['success_semantics'],conditioning_version=v['conditioning'],
                rng_semantics_version=canonical(v['rng']),config_json=canonical(v),compatibility_quality='EXACT_PROFILE'))
        etas={eta_identity(e)[0]:e for c in rows for e in (c['eta_a'],c['eta_b'])}
        for e in etas.values():
            eid,values,hx=eta_identity(e);ins('eta',dict(eta_uid=eid,eta1=values[0],eta2=values[1],eta3=values[2],canonical_hex=hx))
        after={t:digest_table(t,added[t]) for t in tables};assert before==after
    write(HERE/f'{stage}_INSERTED_ROWS.json',inserted)
    seal(f'{stage}_REGISTRATION.receipt.json',[HERE/f'{stage}_INSERTED_ROWS.json',identity],preexisting_row_hashes=before,after_preexisting_row_hashes=after)
    write(HERE/f'{stage}_protocol.json',p);write(HERE/f'{stage}_CASES.json',rows)
    manifest=HERE/f'{stage}_PLANNED_ROLLOUTS.json'
    write(manifest,dict(requests=[{k:c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key')} for c in rows]))
    preflight=planner.preflight(manifest);assert preflight['summary']['ambiguous']==0
    write(HERE/f'{stage}_CACHE_PREFLIGHT.json',preflight)
    missing_keys={(r['state_uid'],r['eta_uid'],r['controller_uid'],s) for r in preflight['details'] for s in r['missing_seeds']}
    with rollout_db.connect(True) as db:
        missing=[c for c in rows if (c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key']) in missing_keys and
            db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() is None]
    missing=list({c['rollout_uid']:c for c in missing}.values())
    write(HERE/f'{stage}_MISSING_CASES.json',missing)
    seal(f'{stage}_PREFLIGHT.receipt.json',[manifest,HERE/f'{stage}_CACHE_PREFLIGHT.json',HERE/f'{stage}_MISSING_CASES.json',HERE/f'{stage}_protocol.json'],
        before_submission=True,planned=len(rows),genuinely_missing=len(missing),existing_unknown_not_rerun=True)
    print(stage,len(rows),'planned;',len(missing),'missing',flush=True)
