"""New experiment: immutable identities and the official rollout cache protocol."""
import os, sys, json, hashlib, math, time
from pathlib import Path
D = Path(__file__).resolve().parent
ROOT = D.parents[1]
sys.path.insert(0, str(ROOT))
from shared_rollout_db.src import rollout_db as db, planner, cache_writer
EXP = 'exp_external_methods_native_frequency_20261104'

def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [clean(v) for v in x]
    if hasattr(x, 'tolist'): return clean(x.tolist())
    if isinstance(x, float) and not math.isfinite(x): return None
    return x

def dump(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())

def make_case(name, scenario, state, payload):
    sid0=db.uid('scenario',scenario); sid=db.uid('state',dict(scenario=sid0,physical=state))
    eta_values=payload.get('c1_request',{}).get('spec',{}).get('eta',[0,0,0])
    cid=db.uid('ctl',dict(scenario=sid0,payload=payload));eid=db.eta_identity(eta_values)[0]
    sk=db.canonical(dict(seed=0))
    return dict(name=name,scenario=scenario,state=state,payload=payload,scenario_uid=sid0,state_uid=sid,
                controller_uid=cid,eta_uid=eid,seed_key=sk,
                rollout_uid=db.uid('roll',dict(state=sid,eta=eid,controller=cid,seed=sk)),eta_values=eta_values)

def register(cases, label):
    assert cache_writer.ROOT.resolve()==ROOT/'shared_rollout_db'
    for folder in ['raw','traces','logs','artifacts']:
        assert (D/folder).resolve().is_relative_to(Path('/mnt/ssd/zhihan/Basin_C1'))
    with db.connect() as con, db.transaction(con):
        con.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,metadata_json) VALUES(?,?,?,?)',
                    (EXP,EXP,str(D),db.canonical(dict(training=False))))
        for c in cases:
            rows={'scenario':dict(scenario_uid=c['scenario_uid'],name=c['name'],code_config_fingerprint=c['scenario_uid'],metadata_json=db.canonical(c['scenario'])),
                  'state':dict(state_uid=c['state_uid'],scenario_uid=c['scenario_uid'],content_hash=hashlib.sha256(db.canonical(c['state']).encode()).hexdigest(),physical_state_json=db.canonical(c['state']),goals_geometry_json=db.canonical(c['scenario']),provenance_json=db.canonical(dict(experiment=EXP)),identity_quality='CONTENT_EXACT'),
                  'controller_config':dict(controller_uid=c['controller_uid'],scenario_uid=c['scenario_uid'],config_json=db.canonical(c['payload']),compatibility_quality='EXACT_PROFILE')}
            for table, vals in rows.items():
                con.execute(f'INSERT OR IGNORE INTO {table} ('+','.join(vals)+') VALUES('+','.join('?' for _ in vals)+')',list(vals.values()))
            eid,e,h=db.eta_identity(c.get('eta_values',[0,0,0]));con.execute('INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex) VALUES(?,?,?,?,?)',(eid,*e,h))
    path=D/'tasks'/(label+'_planned.json');dump(path,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in cases],exact_configs=cases))
    pf=planner.preflight(path);dump(D/'tasks'/(label+'_preflight.json'),pf)
    assert pf['summary']['ambiguous']==0
    for c in cases: dump(D/'tasks'/(c['rollout_uid']+'.json'),c)
    return pf

def record(c, result, arrays):
    import numpy as np
    dest=D/'traces'/(c['rollout_uid']+'.npz');tmp=dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,**arrays);tmp.replace(dest)
    r=dict(result,**{k:c[k] for k in ['scenario_uid','state_uid','controller_uid','eta_uid','seed_key','rollout_uid']},
           case=c,trace=str(dest),trace_sha256=sha(dest),cache_identity_schema='registered_exact_uid_v1',
           eta=c.get('eta_values',[0,0,0]),seed=0,deadlock=0,source='new_rollout',worker_cpu=sorted(os.sched_getaffinity(0)),
           initial_positions=c['state']['positions'],goals=c['state']['goals'],
           **({'initial_velocities':c['state']['velocities']} if 'velocities' in c['state'] else {}))
    cache_writer.append_journal([clean(r)],EXP);dump(D/'raw'/(c['rollout_uid']+'.json'),r)
    dump(D/(c['name']+'_RESULT.json'),r)
    return r
