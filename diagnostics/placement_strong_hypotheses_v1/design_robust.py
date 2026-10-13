"""B15 confirmation on the first 32 uniform-box Sobol points in two Ring states."""
import json
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from shared_rollout_db.src.rollout_db import eta_identity
from shared_rollout_db.src.planner import preflight


def main():
    p=json.loads((HERE/'protocol.json').read_text())
    states=[s for s in p['states'] if s['scenario']=='ring_exchange'][:2]
    original={(c['uid'],c['chain'],c['index']):c for c in p['cases'] if c['kind']=='sobol64'}
    jobs=[]
    for st in states:
        for index in range(32):
            for chain in ('TT','FF'):
                base=original[(st['uid'],chain,index)]
                for seed in range(1,16):
                    jobs.append(dict(uid=st['uid'],scene=st['scenario'],chain=chain,index=index,eta=base['eta'],
                                     eta_uid=base['eta_uid'],controller_uid=base['controller_uid'],seed=seed))
    q=dict(states=states,jobs=jobs,seed0_source='panel_results',
           selection='first two frozen Ring states, first 32 scrambled Sobol points; no outcome selection',
           criterion='B15 success if >=15 of 16 paired future seeds succeed')
    (HERE/'robust_protocol.json').write_text(json.dumps(q,indent=2)+'\n')
    path=HERE/'robust_planned_rollouts.json'
    path.write_text(json.dumps(dict(requests=[dict(state_uid=x['uid'],eta_uid=x['eta_uid'],
        controller_uid=x['controller_uid'],seed_keys=[json.dumps(dict(future_root=p['future_root'],seed=x['seed']),separators=(',',':'))])
        for x in jobs]),indent=2)+'\n')
    pre=preflight(path);(HERE/'robust_cache_preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
    print(len(jobs),pre['summary'])


if __name__=='__main__':main()
