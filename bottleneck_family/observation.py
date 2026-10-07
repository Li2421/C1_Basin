"""Physical local-goal guidance for the fixed Gap-1 map.

The waypoint is computed from public static geometry and current positions;
it contains no robot priority, occupancy reservation or hidden expert mode.
"""
import numpy as np


def gate_waypoints(env):
    c=env.config
    if len(c.barrier_x) != 1 or len(c.openings[0]) != 1:
        raise ValueError('gate-waypoint observation currently supports Gap 1 only')
    gate_x=c.barrier_x[0]
    gate_y=c.openings[0][0][0]
    offset=c.barrier_thickness/2+env.instance.geometry.margin+.12
    result=[]
    for p,g in zip(env.positions,env.goals):
        direction=1 if g[0] > p[0] else -1
        # Use initial-side orientation from the goal, which remains stable
        # after crossing. Opposite-side goal is part of the physical task.
        direction=1 if g[0] > gate_x else -1
        entry=np.array((gate_x-direction*offset,gate_y))
        exit=np.array((gate_x+direction*offset,gate_y))
        before=(p[0]-gate_x)*direction < -offset+.04
        aligned=abs(p[1]-gate_y)<.08
        in_gate=(p[0]-gate_x)*direction < offset-.04
        target=entry if before or (in_gate and not aligned) else exit if in_gate else g
        result.append(target)
    return np.asarray(result)


def policy_observation(env):
    return np.concatenate((env.observation()['agents'],
                           gate_waypoints(env)-env.positions),axis=1).astype(np.float32)
