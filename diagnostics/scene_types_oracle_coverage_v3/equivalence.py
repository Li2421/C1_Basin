"""Preregister and execute >=50 paired open-scene route on/off episodes."""
from dispatch import *
from scenes import build

KEYS=('positions','velocities','nominal','safe','actions','projection_rewrite','swept_clearance','clearance_types')
OUTCOMES=('success','collision','timeout','numerical_failure','outcome','episode_length','constraint_violation_steps','sampled_speed_violation_steps','restoration_steps','minimum_swept_clearance','final_goal_distances')

def main():
    scenes=[build(1,k)for k in range(3)];ctls=controllers();pairs=[]
    for s in scenes:
        for ctl in ctls:
            for i in range(4):pairs.append((s,ctl,i,None))
    for s in scenes:
        for ctl in ctls:
            for i in range(4):
                for robot in range(s['config']['num_agents']):
                    if len(pairs)>=50:break
                    pairs.append((s,ctl,i,robot))
    cases=[];labels=[]
    for s,ctl,i,solo in pairs:
        off=case(s,ctl,i,solo=solo,force_route=False,purpose='route_equivalence')
        on=case(s,ctl,i,solo=solo,force_route=True,purpose='route_equivalence')
        assert off['rng_state_uid']==on['rng_state_uid']
        cases.extend([off,on]);labels.append([off['rollout_uid'],on['rollout_uid']])
    dump(HERE/'ROUTE_EQUIVALENCE_PLANNED.json',dict(pairs=labels,comparison=list(KEYS),outcomes=list(OUTCOMES),selection='50 pairs: first four calibration states only; no heldout state accessed; safety-only, original two projections.'))
    rows={r['rollout_uid']:r for r in execute(cases,'route_equivalence')};comparisons=[]
    for a,b in labels:
        if a not in rows or b not in rows:comparisons.append(dict(off=a,on=b,equal=False,reason='incomplete'));continue
        with np.load(rows[a]['trace'])as x,np.load(rows[b]['trace'])as y:
            detail={k:x[k].shape==y[k].shape and x[k].dtype==y[k].dtype and x[k].tobytes()==y[k].tobytes()for k in KEYS}
        outcome={k:rows[a][k]==rows[b][k]for k in OUTCOMES}
        comparisons.append(dict(off=a,on=b,equal=all(detail.values())and all(outcome.values()),arrays=detail,outcomes=outcome))
    passed=len(comparisons)>=50 and all(x['equal']for x in comparisons)
    dump(HERE/'ROUTE_EQUIVALENCE.json',dict(passed=passed,pairs=len(comparisons),comparisons=comparisons))
    if not passed:raise RuntimeError('HARD_STOP route guidance equivalence failed')
    print('ROUTE EQUIVALENCE PASSED',len(comparisons),'paired full episodes',flush=True)

if __name__=='__main__':main()
