from __future__ import annotations
import argparse,json
from pathlib import Path
from .rollout_db import connect,eta_identity,lookup_exact,uid,canonical

def resolve(con,r):
    if all(k in r for k in ('state_uid','eta_uid','controller_uid')):return r['state_uid'],r['eta_uid'],r['controller_uid'],None
    eta=eta_identity(r['eta'])[0]
    sc=r.get('scenario_uid')
    if not sc and r.get('scenario'):
        q=con.execute('SELECT scenario_uid FROM scenario WHERE name=?',(r['scenario'],)).fetchall()
        if len(q)==1:sc=q[0][0]
    states=con.execute('''SELECT DISTINCT a.state_uid FROM state_alias a JOIN state s USING(state_uid)
      WHERE a.alias=? AND (? IS NULL OR s.scenario_uid=?)''',(r['state'],sc,sc)).fetchall()
    if len(states)!=1:return None,eta,None,'AMBIGUOUS_STATE'
    ctl=r.get('controller_uid')
    if not ctl and r.get('controller_config'):
        ctl=uid('ctl',r['controller_config'])
    if not ctl:return states[0][0],eta,None,'AMBIGUOUS_CONTROLLER'
    return states[0][0],eta,ctl,None

def preflight(manifest):
    obj=json.load(open(manifest)); req=obj['requests'] if isinstance(obj,dict) else obj
    totals={'total_requested':0,'exact_reusable':0,'partial_reusable':0,'aggregate_reusable':0,'ambiguous':0,'incompatible':0,'genuinely_missing':0};details=[]
    with connect(True) as con:
      for r in req:
        seeds=r.get('seed_keys') or ([r['seed_key']] if 'seed_key' in r else [])
        totals['total_requested']+=len(seeds)
        sid,eid,cid,err=resolve(con,r)
        if err:
            totals['ambiguous']+=len(seeds);details.append({'request':r,'status':'AMBIGUOUS','reason':err,'missing_seeds':seeds});continue
        x=lookup_exact(con,sid,eid,cid,seeds);found=len(seeds)-len(x['missing_seeds'])
        if x['status']=='EXACT_REUSE':totals['exact_reusable']+=len(seeds)
        elif x['status']=='PARTIAL_SEED_REUSE':totals['partial_reusable']+=found;totals['genuinely_missing']+=len(x['missing_seeds'])
        elif x['status']=='AGGREGATE_REUSE':totals['aggregate_reusable']+=len(seeds)
        else:totals['genuinely_missing']+=len(seeds);totals['incompatible']+=len(seeds)
        details.append({'state_uid':sid,'eta_uid':eid,'controller_uid':cid,'status':x['status'],'reused':found,'missing_seeds':x['missing_seeds']})
    return {'manifest':str(manifest),'summary':totals,'details':details}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--output');a=ap.parse_args();x=preflight(a.manifest)
    text=json.dumps(x,indent=2,sort_keys=True)+'\n'
    if a.output:Path(a.output).write_text(text)
    print(text,end='')
if __name__=='__main__':main()

