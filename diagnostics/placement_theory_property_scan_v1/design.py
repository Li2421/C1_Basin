"""Outcome-blind action and nearby-state scan design; no rollout here."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')

def main():
    states=json.loads((HARD/'confirmation_manifest.json').read_text())['states']
    source=json.loads((HARD/'screening_manifest.json').read_text())
    chosen=[]
    for scene in ('ring_exchange','toy_give_way'):
        chosen+=sorted((s for s in states if s['scenario']==scene),key=lambda s:s['uid'])[:6]
    eta=np.asarray(source['eta'],float)
    points=[{'name':'zero','eta':[0.,0.,0.]},
            {'name':'candidate0','eta':eta[0].tolist()},
            {'name':'candidate8','eta':eta[8].tolist()}]
    state_uids=('hard_confirm_ring_1_920004','hard_confirm_ring_1_920005',
                'hard_confirm_toy_1_820001','hard_confirm_toy_1_820002')
    state_points=[{'name':'zero','eta':[0.,0.,0.]},
                  {'name':'candidate4','eta':eta[4].tolist()}]
    rng=np.random.default_rng(2026100817)
    directions={}
    for st in chosen:
        d=rng.normal(size=np.asarray(st['physical']['positions']).shape)
        d/=np.linalg.norm(d)
        directions[st['uid']]=d.tolist()
    out=dict(source_manifest=str(HARD/'screening_manifest.json'),
        source_states=str(HARD/'confirmation_manifest.json'),
        states=chosen,action_points=points,
        state_probe_uids=state_uids,state_probe_points=state_points,
        state_position_directions=directions,
        action_eps=[.01,.003,.001,.0003],state_radii=[.001,.003,.01,.03],
        state_jacobian_epsilon=.001,action_active_tolerance=1e-5,
        future_root=source['future_root'],seed=0,
        selection='First six frozen confirmation states per scene; positions and eta indices fixed before new outcomes.',
        note='Theory eta domain is R^3; finite candidate points are only probes.')
    (HERE/'protocol.json').write_text(json.dumps(out,indent=2)+'\n')
    print(len(chosen),len(points),state_uids)

if __name__=='__main__':main()
