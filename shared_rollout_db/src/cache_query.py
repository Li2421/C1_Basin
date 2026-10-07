from __future__ import annotations
from .rollout_db import connect,lookup_exact as _lookup_exact,robust_status_15of16 as _robust_status,find_nearby_eta,eta_identity,uid

def _resolve(con,scenario,state,eta,controller_config):
    sc=con.execute('SELECT scenario_uid FROM scenario WHERE scenario_uid=? OR name=?',(scenario,scenario)).fetchall()
    if len(sc)!=1:return None,None,None,'AMBIGUOUS_SCENARIO'
    sc=sc[0][0]
    if str(state).startswith('state_'):sid=state
    else:
        z=con.execute('SELECT DISTINCT state_uid FROM state_alias WHERE scenario_uid=? AND alias=?',(sc,state)).fetchall()
        if len(z)!=1:return None,None,None,'AMBIGUOUS_STATE'
        sid=z[0][0]
    eid=eta if str(eta).startswith('eta_') else eta_identity(eta)[0]
    cid=controller_config if str(controller_config).startswith('ctl_') else uid('ctl',controller_config)
    return sid,eid,cid,None

def lookup_exact(scenario,state,eta,controller_config,seeds):
    with connect(True) as con:
        sid,eid,cid,err=_resolve(con,scenario,state,eta,controller_config)
        return {'status':'AMBIGUOUS','records':[],'missing_seeds':list(seeds),'reason':err} if err else _lookup_exact(con,sid,eid,cid,seeds)

def get_missing_seeds(scenario,state,eta,controller_config,seeds):
    return lookup_exact(scenario,state,eta,controller_config,seeds)['missing_seeds']

def robust_status_15of16(scenario,state,eta,controller_config,seed_keys=None):
    with connect(True) as con:
        sid,eid,cid,err=_resolve(con,scenario,state,eta,controller_config)
        return {'status':'AMBIGUOUS','reason':err} if err else _robust_status(con,sid,eid,cid,seed_keys)

def get_state_eta_matrix(state_uids=None,scenario_uid=None):
    with connect(True) as con:
        sql='''SELECT s.scenario_uid,r.state_uid,e.eta_uid,e.eta1,e.eta2,e.eta3,r.controller_uid,
          COUNT(*) n_trials,SUM(r.success) n_success FROM rollout r JOIN state s USING(state_uid) JOIN eta e USING(eta_uid)
          WHERE r.conflict_quarantined=0 AND r.numerical_failure=0'''; args=[]
        if scenario_uid: sql+=' AND s.scenario_uid=?';args.append(scenario_uid)
        if state_uids:
            sql+=' AND r.state_uid IN ('+','.join('?'*len(state_uids))+')';args+=list(state_uids)
        sql+=' GROUP BY r.state_uid,e.eta_uid,r.controller_uid'
        return [dict(r) for r in con.execute(sql,args)]

def find_eta_history(state_uid):
    with connect(True) as con:return [dict(r) for r in con.execute('''SELECT e.*,r.controller_uid,COUNT(*) n_trials,SUM(r.success) n_success
      FROM rollout r JOIN eta e USING(eta_uid) WHERE r.state_uid=? AND r.conflict_quarantined=0 GROUP BY e.eta_uid,r.controller_uid''',(state_uid,))]
def find_state_prevalence(eta_uid):
    with connect(True) as con:return [dict(r) for r in con.execute('''SELECT s.scenario_uid,r.controller_uid,COUNT(DISTINCT r.state_uid) states,
      SUM(CASE WHEN c.n_success>=15 AND c.n_trials>=16 THEN 1 ELSE 0 END) b15_pairs FROM rollout r JOIN state s USING(state_uid)
      JOIN state_eta_seed_coverage c USING(state_uid,eta_uid,controller_uid) WHERE r.eta_uid=? GROUP BY s.scenario_uid,r.controller_uid''',(eta_uid,))]
