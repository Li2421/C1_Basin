"""Append-only official DB registration, exact preflight, journals."""
from experiment import *
import shared_rollout_db
from shared_rollout_db.src import rollout_db as dbm, planner, cache_writer
assert Path(shared_rollout_db.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db')
assert (cache_writer.ROOT/'journals').resolve()==ROOT/'shared_rollout_db/journals'
EXP='exp_baseline2_literature_scenes_v1'

def register(stage,cases):
    prereg();tables=('experiment','scenario','state','controller_config','eta');added={t:set() for t in tables};inserted=[]
    with dbm.connect() as db,dbm.transaction(db):
        keys={t:db.execute('PRAGMA table_info('+t+')').fetchone()['name'] for t in tables}
        def digest(t,skip=()):
            hh=hashlib.sha256();count=0
            for r in db.execute('SELECT * FROM '+t+' ORDER BY 1'):
                if r[keys[t]] in skip:continue
                hh.update(dbm.canonical(dict(r)).encode());count+=1
            return dict(rows=count,sha256=hh.hexdigest())
        before={t:digest(t) for t in tables}
        def ins(t,v):
            old=db.execute('SELECT * FROM '+t+' WHERE '+keys[t]+'=?',(v[keys[t]],)).fetchone()
            if old:
                assert all(old[k]==x for k,x in v.items()),('UID_CONTENT_CONFLICT',t,v[keys[t]])
            else:
                db.execute('INSERT OR IGNORE INTO '+t+'('+','.join(v)+') VALUES('+','.join('?' for _ in v)+')',list(v.values()))
                added[t].add(v[keys[t]]);inserted.append(dict(table=t,row=v))
        ins('experiment',dict(experiment_uid=EXP,name='Four literature scenes pipeline migration',path=str(HERE),protocol_hash=sha(HERE/'PREREGISTRATION.md'),code_hash=sha(HERE/'experiment.py'),metadata_json=canonical(dict(evidence='DEVELOPMENT'))))
        for c in cases:
            s=c['state'];physical={k:s[k] for k in ('positions','velocities','goals')}
            ins('scenario',dict(scenario_uid=c['scenario_uid'],name=s['scene']+'_'+c['version'],code_config_fingerprint=c['scenario_uid'],metadata_json=canonical(c['scenario_payload'])))
            ins('state',dict(state_uid=c['state_uid'],scenario_uid=c['scenario_uid'],source_group=f"N{s['N']}_state{s['index']}",content_hash=uid('physical',physical),physical_state_json=canonical(physical),goals_geometry_json=canonical(c['scenario_payload']),provenance_json=canonical(dict(sampler=sha(HERE/'states.py'),state_index=s['index'])),identity_quality='CONTENT_EXACT'))
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
    missing={(r['state_uid'],r['eta_uid'],r['controller_uid'],s) for r in pf['details'] for s in r['missing_seeds']}
    with dbm.connect(True) as db:
        todo=[c for c in cases if (c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key']) in missing and db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() is None]
    write(HERE/f'{stage}_MISSING_CASES.json',todo)
    seal(f'{stage}_PREFLIGHT.receipt.json',[manifest,HERE/f'{stage}_CACHE_PREFLIGHT.json',HERE/f'{stage}_MISSING_CASES.json',HERE/f'{stage}_CASES.json'],before_submission=True,unknowns_not_retried=True)
    print(stage,pf['summary'],'executions',len(todo),flush=True)

def case(s,version,method,scenario_payload,controller_payload,seed=0,eta_a=None,eta_b=None,theta=None):
    eta_a=eta_a or [0.,0.,0.];eta_b=eta_b or eta_a
    sc=uid('scenario',scenario_payload);physical={k:s[k] for k in ('positions','velocities','goals')}
    sid=uid('state',dict(scenario=sc,physical=physical));payload=dict(controller_payload,method=method,eta_b=eta_b,theta=theta)
    cid=uid('ctl',dict(scenario=sc,payload=payload));eid=dbm.eta_identity(eta_a)[0];sk=canonical(dict(future_index=seed))
    rid=uid('roll',dict(state=sid,eta=eid,controller=cid,seed=sk))
    return dict(state=s,version=version,method=method,scenario_payload=scenario_payload,controller_payload=payload,scenario_uid=sc,state_uid=sid,controller_uid=cid,eta_uid=eid,seed_key=sk,rollout_uid=rid,seed=seed,eta_a=eta_a,eta_b=eta_b,theta=theta)

def record(c,result):
    return dict(result,**{k:c[k] for k in ('scenario_uid','state_uid','controller_uid','eta_uid','seed_key','rollout_uid')},
                case=c,eta=c['eta_a'],cache_identity_schema='registered_exact_uid_v1',initial_positions=c['state']['positions'],initial_velocities=c['state']['velocities'],goals=c['state']['goals'],future_index=c['seed'],deadlock=0,timestamp=utc())

def save(stage,c,result,arrays):
    import numpy as np
    folder=HERE/stage;trace=folder/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(trace,**arrays)
    raw=folder/'raw'/f"{c['rollout_uid']}.json"
    result=clean(record(c,dict(result,trace_path=str(trace),trace_sha256=sha(trace))))
    write(raw,result)
    cache_writer.append_journal([result],EXP,worker_id=f'{stage}_{os.getpid()}')
    return result

def save_alias(stage,c,source_path):
    old=read(source_path);assert sha(old['trace_path'])==old['trace_sha256']
    result={k:v for k,v in old.items() if k not in ('case','rollout_uid','state_uid','controller_uid','scenario_uid','eta_uid','seed_key','timestamp','eta','future_index','initial_positions','initial_velocities','goals','cache_identity_schema')}
    result.update(deterministic_source=dict(path=str(source_path),sha256=sha(source_path),reason='Controller action ignores PRNG key; exact same state/action schedule, physics and numerical inputs'),elapsed_seconds=0.,new_physical_steps=0)
    result=record(c,result);write(HERE/stage/'raw'/f"{c['rollout_uid']}.json",result);cache_writer.append_journal([result],EXP,worker_id=f'{stage}_alias_{os.getpid()}')
    return result
