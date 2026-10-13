"""Frozen design for the strong-hypothesis audit, before viewing its outcomes."""
import json
import sqlite3
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HARD = Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
sys.path.insert(0, str(ROOT))
from shared_rollout_db.src.rollout_db import eta_identity


def main():
    source = json.loads((HARD/'screening_manifest.json').read_text())
    candidates = np.asarray(source['eta'], dtype=np.float64)
    lo = candidates.min(axis=0)
    hi = candidates.max(axis=0)
    pad = 0.1 * (hi-lo)
    lo -= pad
    hi += pad
    states = json.loads((HARD/'confirmation_manifest.json').read_text())['states']
    selected = []
    for scene in ('ring_exchange', 'toy_give_way'):
        ss = sorted((s for s in states if s['scenario']==scene), key=lambda s:s['uid'])
        selected += ss[:6]
    sampler = qmc.Sobol(d=3, scramble=True, seed=20261008)
    sobol = qmc.scale(sampler.random_base2(m=6), lo, hi)
    grid = np.asarray(list(product(*[np.linspace(lo[j],hi[j],5) for j in range(3)])))
    points = []
    for i,p in enumerate(sobol):points.append(dict(kind='sobol64',index=i,eta=p.tolist()))
    for i,p in enumerate(grid):points.append(dict(kind='grid5',index=i,eta=p.tolist()))
    con = sqlite3.connect(HARD/'rollout.sqlite')
    controller_by_scene_chain={}
    for st in selected:
        for chain in ('TT','FF'):
            controller_by_scene_chain[(st['scenario'],chain)]=con.execute(
                'SELECT controller_uid FROM rollout WHERE state_uid=? AND chain=? LIMIT 1',
                (st['uid'],chain)).fetchone()[0]
    requests=[];cases=[]
    seed_key=json.dumps({'future_root':source['future_root'],'seed':0},separators=(',',':'))
    for st in selected:
        grid_enabled=st['uid'].endswith(('920001','920002','820001','820002'))
        for pt in points:
            if pt['kind']=='grid5' and not grid_enabled:continue
            for chain in ('TT','FF'):
                ctl=controller_by_scene_chain[(st['scenario'],chain)]
                eid=eta_identity(pt['eta'])[0]
                case=dict(uid=st['uid'],scenario=st['scenario'],chain=chain,kind=pt['kind'],index=pt['index'],
                          eta=pt['eta'],eta_uid=eid,controller_uid=ctl,seed=0)
                cases.append(case)
                requests.append(dict(state_uid=st['uid'],eta_uid=eid,controller_uid=ctl,seed_keys=[seed_key]))
    con.close()
    protocol=dict(source_manifest=str(HARD/'screening_manifest.json'),future_root=source['future_root'],
        eta_box=dict(lower=lo.tolist(),upper=hi.tolist(),rule='candidate coordinate min/max plus 10% of each span'),
        selection='first six lexicographically sorted frozen confirmation states per scenario, before new outcome labels',
        grid_states=[s['uid'] for s in selected if s['uid'].endswith(('920001','920002','820001','820002'))],
        states=selected,points=points,cases=cases,
        topology_resolution='nested 3^3 and 5^3 grid; follow-up refinement only if justified',
        theory_eta_domain='R^3; numerical box is only an observation window')
    HERE.mkdir(exist_ok=True,parents=True)
    (HERE/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (HERE/'planned_rollouts.json').write_text(json.dumps({'requests':requests},indent=2)+'\n')
    print(len(cases),len(requests),lo,hi)


if __name__=='__main__':main()
