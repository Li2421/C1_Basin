import resume_bootstrap
from experiment import *
from shared_rollout_db.src import planner
from shared_rollout_db.src.rollout_db import connect

def build(stages,phase):
    folder=HERE/'RESUME_QUEUES'/phase;folder.mkdir(parents=True,exist_ok=True)
    allcases=[]
    for stage in stages:
        suffix='PILOT_CASES_RESUME' if phase.startswith('pilot') else 'CASES'
        allcases.extend(dict(c,output_stage=stage) for c in read(HERE/f'{stage}_{suffix}.json'))
    manifest=folder/'PLANNED_ROLLOUTS.json'
    write(manifest,dict(requests=[{k:c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key')} for c in allcases]))
    pf=planner.preflight(manifest);write(folder/'CACHE_PREFLIGHT.json',pf)
    assert not pf['summary']['ambiguous'],pf['summary']
    missing={(r['state_uid'],r['eta_uid'],r['controller_uid'],s) for r in pf['details'] for s in r['missing_seeds']}
    todo=[];preserved_unknown=[]
    with connect(True) as db:
        for c in allcases:
            if (c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key']) not in missing:continue
            existing=db.execute('SELECT numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            if existing:
                assert not existing['conflict_quarantined'],('quarantined_identity',c['rollout_uid'])
                assert existing['numerical_failure'],('unexpected_existing_preflight_missing',c['rollout_uid'])
                preserved_unknown.append(c['rollout_uid']);continue
            todo.append(c)
    write(folder/'PRESERVED_UNKNOWNS.json',preserved_unknown)
    groups={}
    for c in todo:
        key=(c['output_stage'],c['controller_payload']['nominal']['controller'],c['state']['N'],c['state']['index'])
        groups.setdefault(key,[]).append(c)
    index=[]
    for key,rows in groups.items():
        gid=uid('group',key);path=folder/'groups'/f'{gid}.json';write(path,rows)
        nominal=rows[0]['controller_payload']['nominal'];factor={'conflict_based_search':10,'prioritized_astar':4,'gradient_mpc':3}.get(nominal['type_code'],1)
        cost=rows[0]['scenario_payload']['config']['max_steps']*rows[0]['state']['N']*len(rows)*factor
        index.append(dict(group=gid,path=str(path),sha256=sha(path),count=len(rows),scheduling_cost=cost))
    # Execution/RNG/state identities are independent across groups. Start the
    # expensive existing actors first to reduce idle cores at the batch tail.
    index.sort(key=lambda r:(-r['scheduling_cost'],r['group']))
    write(folder/'INDEX.json',index)
    write(folder/'QUEUE.receipt.json',dict(utc=utc(),phase=phase,stages=stages,planned=len(allcases),missing=len(todo),preserved_unknowns=len(preserved_unknown),preflight=pf['summary'],
          files={str(p):sha(p) for p in (manifest,folder/'CACHE_PREFLIGHT.json',folder/'INDEX.json')}))
    print(phase,'planned',len(allcases),'missing',len(todo),'groups',len(index),flush=True)
if __name__=='__main__':build(sys.argv[2:],sys.argv[1])
