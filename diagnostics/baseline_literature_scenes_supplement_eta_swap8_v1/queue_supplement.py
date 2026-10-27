import bootstrap
from experiment import *
from shared_rollout_db.src import planner
from shared_rollout_db.src.rollout_db import connect

def build(stage):
    folder=HERE/'QUEUES'/stage;folder.mkdir(parents=True,exist_ok=True);assert not (folder/'QUEUE.receipt.json').exists()
    cases=read(HERE/f'{stage}_CASES.json');manifest=HERE/f'{stage}_PLANNED_ROLLOUTS.json';pf=planner.preflight(manifest);write(folder/'CACHE_PREFLIGHT.json',pf);assert not pf['summary']['ambiguous']
    missing={(r['state_uid'],r['eta_uid'],r['controller_uid'],s) for r in pf['details'] for s in r['missing_seeds']};groups={};unknowns=[]
    with connect(True) as db:
        for c in cases:
            key=(c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key'])
            if key not in missing:continue
            old=db.execute('SELECT numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            if old:
                assert not old['conflict_quarantined'] and old['numerical_failure'];unknowns.append(c['rollout_uid']);continue
            s=c['state'];g=(s['scene'],c['controller_payload']['nominal']['controller'],s['N'],s['index'],s.get('agent'),c['requested_action'])
            groups.setdefault(g,[]).append(c)
    index=[]
    for key,rows in groups.items():
        gid=uid('group',key);p=folder/'groups'/f'{gid}.json';write(p,rows);nom=rows[0]['controller_payload']['nominal'];factor={'conflict_based_search':10,'prioritized_astar':4,'gradient_mpc':3}.get(nom['type_code'],1)
        cost=rows[0]['scenario_payload']['config']['max_steps']*rows[0]['state']['N']*len(rows)*factor
        index.append(dict(group=gid,path=str(p),sha256=sha(p),count=len(rows),scheduling_cost=cost))
    index.sort(key=lambda r:(-r['scheduling_cost'],r['group']));write(folder/'INDEX.json',index);write(folder/'PRESERVED_UNKNOWNS.json',unknowns)
    write(folder/'QUEUE.receipt.json',dict(utc=utc(),stage=stage,planned=len(cases),missing=sum(r['count'] for r in index),preserved_unknowns=len(unknowns),preflight=pf['summary'],files={str(p):sha(p) for p in (manifest,folder/'CACHE_PREFLIGHT.json',folder/'INDEX.json')}))
    with (HERE/'REQUEST_LEDGER.jsonl').open('a') as f:f.write(canonical(read(folder/'QUEUE.receipt.json'))+'\n')
    print(stage,'groups',len(index),'missing',sum(r['count'] for r in index),flush=True)
if __name__=='__main__':build(sys.argv[1])
