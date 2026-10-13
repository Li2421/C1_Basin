"""Serialize every proposed new closed-loop probe before execution."""
import json
import sqlite3
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from shared_rollout_db.src.rollout_db import eta_identity
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
states=json.loads((HARD/'confirmation_manifest.json').read_text())['states']
selected=[]
for scene in ('toy_give_way','ring_exchange'):
    candidates=sorted((s for s in states if s['scenario']==scene),key=lambda s:s['uid'])
    selected.extend([candidates[0],candidates[12]])
eta_candidate=json.loads((HARD/'screening_manifest.json').read_text())['eta'][4]
eta0=[0.,0.,0.]
points=[]
for center_name,center,epsilons in [('zero',eta0,[.01,.003]),('candidate4',eta_candidate,[.01,.003])]:
    points.append((center_name,'center',center))
    for epsilon in epsilons:
        for j in range(3):
            for sign in (-1,1):
                v=list(center);v[j]+=sign*epsilon
                points.append((center_name,f'e{j}_{sign:+d}_{epsilon}',v))
con=sqlite3.connect(HARD/'rollout.sqlite')
requests=[]
for st in selected:
    for placement in ('TT','FF'):
        controller_uid=con.execute('select controller_uid from rollout where state_uid=? and chain=? limit 1',(st['uid'],placement)).fetchone()[0]
        for center_name,variant,eta in points:
            eid=eta_identity(eta)[0]
            requests.append(dict(state_uid=st['uid'],eta_uid=eid,controller_uid=controller_uid,
                seed_keys=['{"future_root":2026100601,"seed":0}']))
con.close()
(HERE/'planned_rollouts.json').write_text(json.dumps({'requests':requests},indent=2)+'\n')
(HERE/'closed_loop_protocol.json').write_text(json.dumps(dict(
    selected_states=selected,seed=0,eta_candidate_index=4,eta_candidate=eta_candidate,
    points=[dict(center=center_name,variant=variant,eta=eta) for center_name,variant,eta in points],
    selection='first and thirteenth lexicographically sorted frozen confirmation states per scene; before viewing outcome labels',
    design='zero and pre-existing candidate4; fixed paired future key at every step; eta fixed for continuation',
    requested_episodes=len(requests),
),indent=2)+'\n')
print(len(requests))
