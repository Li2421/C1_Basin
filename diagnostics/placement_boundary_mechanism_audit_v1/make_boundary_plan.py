"""Deterministic post-hoc boundary brackets from the existing seed-0 panel."""
import json
import sqlite3
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from shared_rollout_db.src.rollout_db import eta_identity
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
states=json.loads((HERE/'closed_loop_protocol.json').read_text())['selected_states']
etas=np.asarray(json.loads((HARD/'screening_manifest.json').read_text())['eta'])
con=sqlite3.connect(HARD/'rollout.sqlite')
segments=[];requests=[]
for state in states:
    uid=state['uid'];labels={};cases={};controllers={}
    for chain,idx,rj,cid,caseid in con.execute('select chain,eta_index,result_json,controller_uid,case_id from rollout where state_uid=? and seed=0 and chain in ("TT","FF")',(uid,)):
        labels.setdefault(chain,{})[idx]=bool(json.loads(rj)['success'])
        cases[(chain,idx)]=caseid;controllers[chain]=cid
    pairs=sorted((float(np.linalg.norm(etas[i]-etas[j])),i,j)
                 for i in range(len(etas)) for j in range(i+1,len(etas))
                 if labels['TT'][i]!=labels['TT'][j] and labels['FF'][i]!=labels['FF'][j])
    if not pairs:segments.append(dict(uid=uid,status='NO_COMMON_BRACKET'));continue
    dist,i,j=pairs[0]
    grid=[]
    for frac in (.25,.5,.75):
        eta=((1-frac)*etas[i]+frac*etas[j]).tolist()
        grid.append(dict(fraction=frac,eta=eta,eta_uid=eta_identity(eta)[0]))
        for chain in ('TT','FF'):
            requests.append(dict(state_uid=uid,eta_uid=eta_identity(eta)[0],controller_uid=controllers[chain],
                seed_keys=['{"future_root":2026100601,"seed":0}']))
    segments.append(dict(uid=uid,scene=state['scenario'],selection='closest Euclidean candidate pair with success flip in both placements at seed0; post-hoc',
        endpoints=[i,j],distance=dist,endpoint_eta=[etas[i].tolist(),etas[j].tolist()],
        endpoint_labels={ch:[labels[ch][i],labels[ch][j]] for ch in ('TT','FF')},
        endpoint_case_ids={ch:[cases[(ch,i)],cases[(ch,j)]] for ch in ('TT','FF')},
        controller_uid=controllers,grid=grid))
con.close()
(HERE/'boundary_protocol.json').write_text(json.dumps(segments,indent=2)+'\n')
(HERE/'boundary_planned_rollouts.json').write_text(json.dumps({'requests':requests},indent=2)+'\n')
print(json.dumps([{k:v for k,v in x.items() if k in ('uid','endpoints','endpoint_labels')} for x in segments],indent=2))
