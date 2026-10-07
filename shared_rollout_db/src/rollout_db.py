from __future__ import annotations
import contextlib, hashlib, json, sqlite3, struct, time
from pathlib import Path
from typing import Iterable

ROOT=Path(__file__).resolve().parents[1]
DB_PATH=ROOT/'rollout.sqlite'

def canonical(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def uid(prefix,obj): return prefix+'_'+hashlib.sha256(canonical(obj).encode()).hexdigest()
def float_hex(x): return struct.pack('>d',float(x)).hex()
def eta_identity(eta):
    v=[float(x) for x in eta]
    if len(v)!=3: raise ValueError('eta must have exactly 3 scalars')
    hx=':'.join(float_hex(x) for x in v)
    return uid('eta',{'float64_be':hx}),v,hx

def connect(readonly=False):
    uri=f'file:{DB_PATH}?mode=ro' if readonly else str(DB_PATH)
    con=sqlite3.connect(uri,uri=readonly,timeout=60)
    con.row_factory=sqlite3.Row
    con.execute('PRAGMA foreign_keys=ON'); con.execute('PRAGMA busy_timeout=60000')
    if not readonly:
        con.execute('PRAGMA journal_mode=WAL'); con.execute('PRAGMA synchronous=FULL')
    return con

def initialize():
    ROOT.mkdir(parents=True,exist_ok=True)
    with connect() as con: con.executescript((ROOT/'schema.sql').read_text())

@contextlib.contextmanager
def transaction(con,retries=8):
    for i in range(retries):
        try:
            con.execute('BEGIN IMMEDIATE'); break
        except sqlite3.OperationalError as e:
            if 'locked' not in str(e).lower() or i==retries-1: raise
            time.sleep(.1*2**i)
    try:
        yield con; con.commit()
    except Exception:
        con.rollback(); raise

def lookup_exact(con,state_uid,eta_uid,controller_uid,seeds:Iterable[str]):
    seeds=list(seeds)
    if not seeds: return {'status':'EXACT_REUSE','records':[],'missing_seeds':[]}
    q=','.join('?'*len(seeds))
    rows=con.execute(f'''SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?
      AND conflict_quarantined=0 AND numerical_failure=0 AND seed_key IN ({q})''',[state_uid,eta_uid,controller_uid,*seeds]).fetchall()
    found={r['seed_key'] for r in rows}; missing=[s for s in seeds if s not in found]
    status='EXACT_REUSE' if not missing else ('PARTIAL_SEED_REUSE' if rows else 'INCOMPATIBLE')
    if not rows:
        ag=con.execute('''SELECT * FROM aggregate_evidence WHERE state_uid=? AND eta_uid=? AND controller_uid=?
          AND conflict_quarantined=0 ORDER BY n_trials DESC LIMIT 1''',(state_uid,eta_uid,controller_uid)).fetchone()
        if ag: status='AGGREGATE_REUSE'
    return {'status':status,'records':[dict(r) for r in rows],'missing_seeds':missing}

def robust_status_15of16(con,state_uid,eta_uid,controller_uid,seed_keys=None):
    if seed_keys:
        x=lookup_exact(con,state_uid,eta_uid,controller_uid,seed_keys)
        if not x['missing_seeds']:
            s=sum(r['success'] for r in x['records']); return {'status':'B15' if s>=15 else 'NOT_B15','n_success':s,'n_trials':16,'basis':'STANDARD_SEEDS'}
    r=con.execute('''SELECT n_trials,n_success,evidence_class FROM aggregate_evidence WHERE state_uid=? AND eta_uid=? AND controller_uid=?
      AND certified_b15=1 AND conflict_quarantined=0 ORDER BY n_trials DESC LIMIT 1''',(state_uid,eta_uid,controller_uid)).fetchone()
    return {'status':'CERTIFIED_B15_FROM_STRONGER_AGGREGATE','n_success':r['n_success'],'n_trials':r['n_trials'],'basis':r['evidence_class']} if r else {'status':'UNDERRESOLVED'}

def get_missing_seeds(con,state_uid,eta_uid,controller_uid,seeds): return lookup_exact(con,state_uid,eta_uid,controller_uid,seeds)['missing_seeds']
def find_nearby_eta(con,eta,radius=.1,limit=20):
    v=[float(x) for x in eta]
    rows=con.execute('SELECT * FROM eta').fetchall(); out=[]
    for r in rows:
        d=sum((r[f'eta{i+1}']-v[i])**2 for i in range(3))**.5
        if d<=radius: out.append({**dict(r),'distance':d})
    return sorted(out,key=lambda x:x['distance'])[:limit]

