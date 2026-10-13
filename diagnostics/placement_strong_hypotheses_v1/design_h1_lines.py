"""Select six Ring boundary lines deterministically from frozen Q16 seed-0 labels."""
import itertools
import json
import sqlite3
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
sys.path.insert(0,str(ROOT))
from shared_rollout_db.src.rollout_db import eta_identity
from shared_rollout_db.src.planner import preflight


def main():
    protocol=json.loads((HERE/'protocol.json').read_text())
    states=[s for s in protocol['states'] if s['scenario']=='ring_exchange']
    eta=np.asarray(json.loads((HARD/'screening_manifest.json').read_text())['eta'])
    con=sqlite3.connect(HARD/'rollout.sqlite')
    lines=[];jobs=[]
    for st in states:
        labels={};controllers={}
        for chain,i,data,ctl in con.execute('SELECT chain,eta_index,result_json,controller_uid FROM rollout WHERE state_uid=? AND chain IN ("TT","FF")',(st['uid'],)):
            r=json.loads(data);controllers[chain]=ctl
            if r['seed']==0:labels[(chain,i)]=bool(r['success'])
        pairs=[(float(np.linalg.norm(eta[a]-eta[b])),a,b) for a,b in itertools.combinations(range(16),2)
               if labels[('TT',a)]!=labels[('TT',b)] and labels[('FF',a)]!=labels[('FF',b)]]
        if not pairs:
            lines.append(dict(uid=st['uid'],available=False));continue
        dist,a,b=min(pairs)
        line=dict(uid=st['uid'],available=True,candidate_indices=[a,b],distance=dist,
                  endpoint_eta=[eta[a].tolist(),eta[b].tolist()],
                  endpoint_labels={ch:[labels[(ch,a)],labels[(ch,b)]] for ch in ('TT','FF')},
                  controllers=controllers,
                  selection='shortest Q16 candidate pair whose seed-0 labels flip for both TT and FF')
        lines.append(line)
        for n in range(1,16):
            t=n/16;point=((1-t)*eta[a]+t*eta[b]).tolist();eid=eta_identity(point)[0]
            for chain in ('TT','FF'):
                jobs.append(dict(uid=st['uid'],chain=chain,fraction=t,eta=point,eta_uid=eid,
                                 controller_uid=controllers[chain]))
    con.close()
    (HERE/'h1_line_protocol.json').write_text(json.dumps(dict(lines=lines,jobs=jobs),indent=2)+'\n')
    path=HERE/'h1_line_planned_rollouts.json'
    path.write_text(json.dumps(dict(requests=[dict(state_uid=x['uid'],eta_uid=x['eta_uid'],controller_uid=x['controller_uid'],
        seed_keys=['{"future_root":2026100601,"seed":0}']) for x in jobs]),indent=2)+'\n')
    pre=preflight(path);(HERE/'h1_line_cache_preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
    print(len(jobs),pre['summary'])


if __name__=='__main__':main()
