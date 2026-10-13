"""Constructive compression alias using an archived real probe tensor.

The modified tensor is algebraic, not a real Flow checkpoint. It shows that
R3 cannot identify which pose/time supplied a response Jacobian.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from repaired_features import descriptors

HERE=Path(__file__).resolve().parent
def main():
    name='base_u_v2_macflow'
    with np.load(HERE/'features'/(name+'.npz')) as z:
        nominal=z['nominal'][0].reshape(4,14)
        field=z['raw_field'][0].reshape(4,7,3,5,2)
    changed=field.copy()
    # Agent 0, same pseudo-time: exchange deviations from the two intercepts,
    # between far/current pose 0 and near-goal moving pose 4.
    p,t,q=0,1,4
    a=field[0,p,t,1:]-field[0,p,t,0]
    b=field[0,q,t,1:]-field[0,q,t,0]
    changed[0,p,t,1:]=field[0,p,t,0]+b
    changed[0,q,t,1:]=field[0,q,t,0]+a
    r3,r5=descriptors(nominal,field.reshape(4,210))
    r3b,r5b=descriptors(nominal,changed.reshape(4,210))
    result=dict(source_controller=name,source_state_index=0,synthetic=True,
        r3_max_abs_difference=float(np.max(np.abs(r3-r3b))),
        r5_l2_difference=float(np.linalg.norm(r5-r5b)),
        near_goal_moving_l2_difference=float(np.linalg.norm(field[0,q,t]-changed[0,q,t])),
        interpretation='A real archived tensor supplies numerical values; swapping conditioned response blocks creates an R3 alias, not a new actual Flow controller.')
    assert result['r3_max_abs_difference']<1e-5 and result['r5_l2_difference']>0
    (HERE/'alias_counterexample.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
