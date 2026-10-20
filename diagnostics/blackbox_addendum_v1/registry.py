from common import *
from shared_rollout_db.src.rollout_db import connect,transaction,canonical,uid,eta_identity
TABLES={'scenario':'scenario_uid','experiment':'experiment_uid','state':'state_uid','controller_config':'controller_uid','eta':'eta_uid'}
class IdentityConflict(Exception):pass

def register(stage,p,cases,extra_rows=()):
    prereg();assert not (HERE/f'{stage}_PREFLIGHT.receipt.json').exists();inserted=[];checks={}
    with connect() as db,transaction(db):
        before={t:{r[k]:hashlib.sha256(canonical(dict(r)).encode()).hexdigest() for r in db.execute('SELECT * FROM '+t)} for t,k in TABLES.items()}
        def ins(t,v):
            key=TABLES[t];old=db.execute(f'SELECT * FROM {t} WHERE {key}=?',(v[key],)).fetchone()
            if old:
                if not all(old[k]==z for k,z in v.items()):raise IdentityConflict(f'{t} {v[key]}')
                return
            keys=list(v);db.execute(f"INSERT OR IGNORE INTO {t} ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})",[v[k] for k in keys]);row=dict(db.execute(f'SELECT * FROM {t} WHERE {key}=?',(v[key],)).fetchone());inserted.append(dict(table=t,row=row))
        ins('experiment',dict(experiment_uid=EXP,name='Black-box controller addendum',path=str(HERE),protocol_hash=sha(HERE/'PREREGISTRATION.md'),code_hash=sha(__file__),metadata_json=canonical(dict(round=37))))
        for t,v in extra_rows:ins(t,v)
        for sid,v in p.get('scenarios',{}).items():
            old=db.execute('SELECT * FROM scenario WHERE scenario_uid=?',(sid,)).fetchone()
            if old:
                assert old['code_config_fingerprint']==v['code_config_fingerprint']
            else:ins('scenario',v)
        for s in p['states']:
            physical={k:s['physical'][k] for k in ['positions','velocities','goals']};old=db.execute('SELECT * FROM state WHERE state_uid=?',(s['state_uid'],)).fetchone()
            if old:
                if json.loads(old['physical_state_json'])!=physical or old['scenario_uid']!=s['scenario_uid']:raise IdentityConflict('state physical '+s['state_uid'])
                continue
            ins('state',dict(state_uid=s['state_uid'],scenario_uid=s['scenario_uid'],source_group=s.get('family',s['uid']),content_hash=s['content_hash'],physical_state_json=canonical(physical),goals_geometry_json=canonical(s.get('evaluation_config',{})),provenance_json=canonical(dict(original_uid=s['uid'],original_state_uid=s.get('original_state_uid'),rng_namespace=s.get('rng_namespace'),reset='true t0',round37=True)),identity_quality='CONTENT_EXACT'))
        seen=set()
        for vv in p['profiles'].values():
            cid=vv['controller_uid'];v=vv['payload']
            if cid in seen:continue
            seen.add(cid);ins('controller_config',dict(controller_uid=cid,scenario_uid=v['scenario_uid'],flow_checkpoint_sha256=v.get('flow_checkpoint_sha256'),orthoflow3_sha256=v.get('orthoflow3_sha256'),safety_config_hash=uid('safe',dict(source=v.get('safety_projection_sha256'),config=v.get('cbf'),engine=v.get('round37_engine'))),horizon=str(v['horizon']),dt=str(v['dt']),success_semantics_version=v['success_semantics'],conditioning_version=canonical(v['conditioning']) if isinstance(v['conditioning'],dict) else v['conditioning'],rng_semantics_version=canonical(v['rng']),config_json=canonical(v),compatibility_quality='EXACT_PROFILE'))
        for eta in [c['eta_a'] for c in cases]:
            eid,v,hx=eta_identity(eta);ins('eta',dict(eta_uid=eid,eta1=v[0],eta2=v[1],eta3=v[2],canonical_hex=hx))
        for t,records in before.items():
            after={r[TABLES[t]]:hashlib.sha256(canonical(dict(r)).encode()).hexdigest() for r in db.execute('SELECT * FROM '+t)};assert all(after.get(k)==v for k,v in records.items()),'DB prior row mutation';checks[t]=dict(rows=len(records),before_sha256=hashlib.sha256(canonical(records).encode()).hexdigest(),after_same_rows_sha256=hashlib.sha256(canonical({k:after[k] for k in records}).encode()).hexdigest(),unchanged=True)
    write(HERE/f'{stage}_INSERTED_ROWS.json',inserted);seal(f'{stage}_REGISTRATION.receipt.json',[HERE/f'{stage}_INSERTED_ROWS.json'],existing_rows=checks)
    write(HERE/f'{stage}_protocol.json',p);write(HERE/f'{stage}_CASES.json',cases);manifest=HERE/f'{stage}_PLANNED_ROLLOUTS.json';unique={c['rollout_uid']:c for c in cases};write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in unique.values()]));pf=planner.preflight(manifest);assert not pf['summary']['ambiguous'];write(HERE/f'{stage}_CACHE_PREFLIGHT.json',pf);missing=[];reuse=[]
    with connect(True) as db:
        for c in unique.values():
            r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            if r:
                assert r['compatibility_quality']=='EXACT_REUSE' and not r['conflict_quarantined'];reuse.append(dict(case=c,numerical_failure=r['numerical_failure']))
            else:missing.append(c)
    write(HERE/f'{stage}_MISSING_CASES.json',missing);write(HERE/f'{stage}_REUSE_CASES.json',reuse);seal(f'{stage}_PREFLIGHT.receipt.json',[manifest,HERE/f'{stage}_CACHE_PREFLIGHT.json',HERE/f'{stage}_MISSING_CASES.json',HERE/f'{stage}_protocol.json',HERE/f'{stage}_CASES.json'],before_submission=True,missing=len(missing),reused=len(reuse));print(stage,'planned',len(cases),'missing',len(missing),flush=True)
