"""Append-only official registration and atomic evidence; adapted from v1 registry."""
from experiment import *
import shared_rollout_db
from shared_rollout_db.src import rollout_db as dbm, planner, cache_writer
assert Path(shared_rollout_db.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db')
assert (cache_writer.ROOT/'journals').resolve()==ROOT/'shared_rollout_db/journals'
EXP='exp_baseline_literature_scenes_supplement_eta_swap8_v1'

def verify_receipt(name):
    proof=read(HERE/name)
    for path,expected in proof['files'].items():assert sha(path)==expected,path
    return proof

def register(stage,cases):
    prereg();verify_receipt('IMPLEMENTATION_FROZEN.receipt.json')
    assert not (HERE/f'{stage}_REGISTRATION.receipt.json').exists(),stage
    tables=('experiment','scenario','state','controller_config','eta')
    added={t:set() for t in tables};inserted=[]
    with dbm.connect() as db,dbm.transaction(db):
        keys={t:db.execute('PRAGMA table_info('+t+')').fetchone()['name'] for t in tables}
        def digest(t,skip=()):
            hh=hashlib.sha256();count=0
            for r in db.execute('SELECT * FROM '+t+' ORDER BY 1'):
                if r[keys[t]] in skip:continue
                hh.update(dbm.canonical(dict(r)).encode());count+=1
            return dict(rows=count,sha256=hh.hexdigest())
        before={t:digest(t) for t in tables};seen={t:set() for t in tables}
        def ins(t,v):
            key=v[keys[t]]
            if key in seen[t]:return
            seen[t].add(key)
            old=db.execute('SELECT * FROM '+t+' WHERE '+keys[t]+'=?',(key,)).fetchone()
            if old:
                assert all(old[k]==x for k,x in v.items()),('UID_CONTENT_CONFLICT',t,key)
            else:
                db.execute('INSERT INTO '+t+'('+','.join(v)+') VALUES('+','.join('?' for _ in v)+')',list(v.values()))
                added[t].add(key);inserted.append(dict(table=t,row=v))
        ins('experiment',dict(experiment_uid=EXP,name='Eta screen and swap8 supplement v1',path=str(HERE),protocol_hash=sha(HERE/'PREREGISTRATION.md'),code_hash=sha(HERE/'experiment.py'),metadata_json=canonical(dict(evidence='DEVELOPMENT'))))
        for c in cases:
            s=c['state'];physical={k:s[k] for k in ('positions','velocities','goals')}
            ins('scenario',dict(scenario_uid=c['scenario_uid'],name=s['scene']+'_'+c['version'],code_config_fingerprint=c['scenario_uid'],metadata_json=canonical(c['scenario_payload'])))
            ins('state',dict(state_uid=c['state_uid'],scenario_uid=c['scenario_uid'],source_group=f"N{s['N']}_state{s['index']}",content_hash=uid('physical',physical),physical_state_json=canonical(physical),goals_geometry_json=canonical(c['scenario_payload']),provenance_json=canonical(c['state_provenance']),identity_quality='CONTENT_EXACT'))
            ins('controller_config',dict(controller_uid=c['controller_uid'],scenario_uid=c['scenario_uid'],config_json=canonical(c['controller_payload']),compatibility_quality='EXACT_PROFILE'))
            eid,v,hx=dbm.eta_identity(c['eta_a']);assert eid==c['eta_uid']
            ins('eta',dict(eta_uid=eid,eta1=v[0],eta2=v[1],eta3=v[2],canonical_hex=hx))
        after={t:digest(t,added[t]) for t in tables};assert before==after
    write(HERE/f'{stage}_INSERTED_ROWS.json',inserted)
    seal(f'{stage}_REGISTRATION.receipt.json',[HERE/f'{stage}_INSERTED_ROWS.json'],preexisting_rows_before=before,preexisting_rows_after=after)
    write(HERE/f'{stage}_CASES.json',cases)
    manifest=HERE/f'{stage}_PLANNED_ROLLOUTS.json'
    write(manifest,dict(requests=[{k:c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key')} for c in cases]))
    pf=planner.preflight(manifest);write(HERE/f'{stage}_CACHE_PREFLIGHT.json',pf)
    assert not pf['summary']['ambiguous'],pf['summary']
    missing={(r['state_uid'],r['eta_uid'],r['controller_uid'],s) for r in pf['details'] for s in r['missing_seeds']}
    with dbm.connect(True) as db:
        todo=[];reuse=[]
        for c in cases:
            key=(c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key'])
            existing=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            if key in missing and existing is None:todo.append(c)
            else:
                assert existing is not None,('missing exact evidence row',c['rollout_uid'])
                reuse.append(dict(rollout_uid=c['rollout_uid'],cached_row=dict(existing)))
    write(HERE/f'{stage}_REUSED_ROWS.json',reuse)
    write(HERE/f'{stage}_MISSING_CASES.json',todo)
    seal(f'{stage}_PREFLIGHT.receipt.json',[manifest,HERE/f'{stage}_CACHE_PREFLIGHT.json',HERE/f'{stage}_MISSING_CASES.json',HERE/f'{stage}_CASES.json',HERE/f'{stage}_REUSED_ROWS.json'],before_submission=True,unknowns_not_retried=True)
    print(stage,pf['summary'],'executions',len(todo),flush=True)

def case(s,version,method,scenario_payload,controller_payload,seed=0,eta_a=None,eta_b=None,theta=None,provenance=None):
    eta_a=[0.,0.,0.] if eta_a is None else eta_a
    eta_b=eta_a if eta_b is None else eta_b
    sc=uid('scenario',scenario_payload);physical={k:s[k] for k in ('positions','velocities','goals')}
    sid=uid('state',dict(scenario=sc,physical=physical));payload=dict(controller_payload,method=method,eta_b=eta_b,theta=theta)
    cid=uid('ctl',dict(scenario=sc,payload=payload));eid=dbm.eta_identity(eta_a)[0];sk=canonical(dict(future_index=seed))
    rid=uid('roll',dict(state=sid,eta=eid,controller=cid,seed=sk))
    return dict(state=s,version=version,method=method,scenario_payload=scenario_payload,controller_payload=payload,scenario_uid=sc,state_uid=sid,controller_uid=cid,eta_uid=eid,seed_key=sk,rollout_uid=rid,seed=seed,eta_a=eta_a,eta_b=eta_b,theta=theta,state_provenance=provenance)

def record(c,result):
    return dict(result,**{k:c[k] for k in ('scenario_uid','state_uid','controller_uid','eta_uid','seed_key','rollout_uid')},case=c,eta=c['eta_a'],cache_identity_schema='registered_exact_uid_v1',initial_positions=c['state']['positions'],initial_velocities=c['state']['velocities'],goals=c['state']['goals'],future_index=c['seed'],deadlock=0,timestamp=utc())

def save(stage,c,result,arrays):
    import numpy as np
    folder=HERE/stage;trace=folder/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True)
    temp=trace.with_suffix('.partial.npz');np.savez_compressed(temp,**arrays)
    with temp.open('rb') as stream:os.fsync(stream.fileno())
    temp.replace(trace)
    raw=folder/'raw'/f"{c['rollout_uid']}.json"
    result=clean(record(c,dict(result,trace_path=str(trace),trace_sha256=sha(trace))))
    write(raw,result)
    journal=cache_writer.append_journal([result],EXP,worker_id=f'{stage}_{os.getpid()}')
    write(folder/'journal_receipts'/f"{c['rollout_uid']}.json",dict(journal=str(journal),raw_sha256=sha(raw)))
    return result

def save_alias(stage,c,source_path):
    old=read(source_path);assert sha(old['trace_path'])==old['trace_sha256']
    omitted=('case','rollout_uid','state_uid','controller_uid','scenario_uid','eta_uid','seed_key','timestamp','eta','future_index','initial_positions','initial_velocities','goals','cache_identity_schema')
    result={k:v for k,v in old.items() if k not in omitted}
    result.update(deterministic_source=dict(path=str(source_path),sha256=sha(source_path),reason='Frozen deterministic controller ignores PRNG; identical state, layer, action schedule and numerical inputs'),elapsed_seconds=0.,new_physical_steps=0)
    result=record(c,result);write(HERE/stage/'raw'/f"{c['rollout_uid']}.json",result)
    journal=cache_writer.append_journal([result],EXP,worker_id=f'{stage}_alias_{os.getpid()}')
    write(HERE/stage/'journal_receipts'/f"{c['rollout_uid']}.json",dict(journal=str(journal),raw_sha256=sha(HERE/stage/'raw'/f"{c['rollout_uid']}.json")))
    return result

def repair_journal(stage,c):
    raw=HERE/stage/'raw'/f"{c['rollout_uid']}.json"
    marker=HERE/stage/'journal_receipts'/f"{c['rollout_uid']}.json"
    if raw.exists() and not marker.exists():
        result=read(raw);assert sha(result['trace_path'])==result['trace_sha256']
        journal=cache_writer.append_journal([result],EXP,worker_id=f'{stage}_recovery_{os.getpid()}')
        write(marker,dict(journal=str(journal),raw_sha256=sha(raw),recovered_existing_raw_without_new_steps=True))
