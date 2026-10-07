"""Adapters used by the two shared Basin_C1 rollout execution layers."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from .rollout_db import connect,canonical,eta_identity,uid
from .cache_writer import append_journal

BASIS='51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38'
FLOW={'ToyGiveWay':'8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32','DoubleBottleneck_4A':'6e2ed4e31443bbb34457d3b3aabe0e7d741b904259391f8893b1104c143546bd'}

def _scenario_uid(con,name):
    r=con.execute('SELECT scenario_uid FROM scenario WHERE name=?',(name,)).fetchone();return r[0] if r else None
def _state_uid(con,scenario_uid,task,state):
    cond=state.get('h_conditioning_identifier') or task.get('h_conditioning_identifier')
    sg=state.get('source_group') or task.get('source_group') or state.get('family_id') or task.get('family_id')
    alias=task.get('state_id') or task.get('episode_id')
    if cond: content=str(cond)
    elif sg:content=hashlib.sha256(canonical({'scenario':scenario_uid,'source_group':str(sg)}).encode()).hexdigest()
    else:
        rows=con.execute('SELECT state_uid FROM state_alias WHERE scenario_uid=? AND alias=?',(scenario_uid,str(alias))).fetchall()
        return rows[0][0] if len(rows)==1 else None
    return uid('state',{'scenario':scenario_uid,'content':content})
def _controller(con,scenario_uid,rng_name):
    rows=con.execute('''SELECT controller_uid FROM controller_config WHERE scenario_uid=? AND flow_checkpoint_sha256=? AND orthoflow3_sha256=?
      AND compatibility_quality='EXACT_PROFILE' AND rng_semantics_version LIKE ?''',(scenario_uid,FLOW.get(con.execute('SELECT name FROM scenario WHERE scenario_uid=?',(scenario_uid,)).fetchone()[0]),BASIS,'%'+rng_name+'%')).fetchall()
    return rows[0][0] if len(rows)==1 else None
def _raw_record(con,rollout_uid):
    x=con.execute('''SELECT sf.path,rs.source_line FROM rollout_source rs JOIN source_file sf USING(source_uid)
      WHERE rs.rollout_uid=? AND sf.classification='SEED_EXACT' ORDER BY length(sf.path),sf.path LIMIT 1''',(rollout_uid,)).fetchone()
    if not x:return None
    try:
      with open(x['path'],errors='replace') as f:
       for i,line in enumerate(f,1):
        if i==x['source_line']:return json.loads(line)
    except:return None
    return None

def reuse_toy_tasks(tasks,states,future_root):
    """Return exact cached raw rows keyed as Oracle._insert expects."""
    out=[]
    with connect(True) as con:
      sc=_scenario_uid(con,'ToyGiveWay');ctl=_controller(con,sc,'matched_future_index_v1') if sc else None
      if not ctl:return out
      for t in tasks:
        st=states.get(t['state_id'],{});sid=_state_uid(con,sc,t,st);eid=eta_identity(t['eta'])[0]
        # Root and namespace are already locked by controller/state identities.
        candidates=[canonical({'future_index':int(t['future_index'])})]
        r=None
        for sk in candidates:
          r=con.execute('''SELECT rollout_uid FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?
            AND compatibility_quality='EXACT_REUSE' AND conflict_quarantined=0 AND numerical_failure=0''',(sid,eid,ctl,sk)).fetchone()
          if r:break
        if r:
          raw=_raw_record(con,r['rollout_uid'])
          if raw:out.append({**raw,**t,'global_cache_reuse':True,'global_rollout_uid':r['rollout_uid']})
    return out

def reuse_db_jobs(jobs):
    """Materialize exact cached DB rows for the legacy shared DB executor."""
    out={}
    with connect(True) as con:
      sc=_scenario_uid(con,'DoubleBottleneck_4A');ctl=_controller(con,sc,'seed_rollout_foldin_v1') if sc else None
      if not ctl:return out
      for t in jobs:
        sid=_state_uid(con,sc,t,{'family_id':t.get('family_id')});eid=eta_identity(t.get('theta',t.get('eta')))[0]
        sk=canonical({'seed':t.get('seed'),'rollout_id':t.get('rollout_id')})
        r=con.execute('''SELECT rollout_uid FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?
          AND compatibility_quality='EXACT_REUSE' AND conflict_quarantined=0 AND numerical_failure=0''',(sid,eid,ctl,sk)).fetchone()
        if r:
          raw=_raw_record(con,r['rollout_uid'])
          if raw:out[t['job_id']]={**raw,**t,'global_cache_reuse':True,'global_rollout_uid':r['rollout_uid']}
    return out

def journal_toy(records,experiment_path,future_root):
    exp=uid('exp',{'path':str(Path(experiment_path).resolve())})
    enriched=[{**r,'scenario':'ToyGiveWay','representation':'P1-OrthoFlow3','future_root_seed':future_root} for r in records]
    return append_journal(enriched,exp)

def journal_db(records,experiment_path):
    exp=uid('exp',{'path':str(Path(experiment_path).resolve())})
    enriched=[{**r,'scenario':'DoubleBottleneck_4A','representation':'P1-OrthoFlow3'} for r in records]
    return append_journal(enriched,exp)
