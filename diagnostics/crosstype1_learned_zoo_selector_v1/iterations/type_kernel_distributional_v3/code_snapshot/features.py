"""Mechanism features from outcome-free trajectories only; no label imports."""
import numpy as np

TIMES=(50,100,200,300)
NAMES=('strict_freeze','moving_no_progress',
       *(f'progress_{arm}_{t}' for arm in ('safety_baseline','F13','F10') for t in TIMES),
       *(f'projection_{arm}' for arm in ('safety_baseline','F13','F10')),
       'rewrite_magnitude','mixed_circulation',*(f'eta13_minus_eta10_{t}' for t in TIMES),
       'nominal_randomness')


def one_trace(trace,dt,max_speed,goal_tolerance):
    required=('positions','velocities','nominal_actions','safe_actions','projection_active','swept_clearance','goals')
    if any(k not in trace for k in required):
        raise ValueError('Missing mechanism trace arrays')
    p,v=np.asarray(trace['positions']),np.asarray(trace['velocities'])
    n,a=np.asarray(trace['nominal_actions']),np.asarray(trace['safe_actions'])
    if p.shape != (301,4,2) or v.shape!=p.shape or n.shape!=(300,4,2) or a.shape!=n.shape:
        raise ValueError('Need complete 300-step probe; missing probes cannot be silently averaged away')
    if not all(np.isfinite(trace[k]).all() for k in required):
        raise ValueError('Nonfinite probe')
    distance=np.linalg.norm(np.asarray(trace['goals'])[None]-p,axis=-1)
    speed=np.linalg.norm(v,axis=-1)
    remaining=distance>goal_tolerance
    freeze=np.mean(remaining[20:301] & (speed[20:301]<.05))
    window=round(1/dt)
    starts=np.arange(max(20,window),301)
    moving=np.mean(remaining[starts] & (speed[starts]>=.05) &
                   (distance[starts-window]-distance[starts]<.05))
    initial=distance[0].sum()
    progress=np.asarray([(initial-distance[t].sum())/initial for t in TIMES])
    activation=float(np.mean(trace['projection_active']))
    rewrite=float(np.mean(np.linalg.norm(a-n,axis=-1))/max_speed)
    # Derivative d(theta)/dt has sign cross(position, velocity).
    angular=p[100:301,:,0]*v[100:301,:,1]-p[100:301,:,1]*v[100:301,:,0]
    mask=remaining[100:301]
    mixed=float(np.mean(np.any((angular>0)&mask,axis=1)&np.any((angular<0)&mask,axis=1)))
    return dict(freeze=float(freeze),moving=float(moving),progress=progress,
                activation=activation,rewrite=rewrite,mixed=mixed)


def aggregate(traces,random_actions,dt,max_speed,goal_tolerance):
    """Exactly 16 traces per arm; random actions shape state × 8 keys × 4 × 2."""
    by_arm={}
    for arm in ('safety_baseline','F13','F10'):
        if len(traces[arm])!=16:
            raise ValueError('All 16 probe states required')
        values=[one_trace(t,dt,max_speed,goal_tolerance) for t in traces[arm]]
        by_arm[arm]={k:np.mean([v[k] for v in values],axis=0) for k in values[0]}
    base=by_arm['safety_baseline']
    random_actions=np.asarray(random_actions)
    if random_actions.shape!=(16,8,4,2):
        raise ValueError('Expected 16 initial observations × 8 random keys')
    random=float(np.mean(np.std(random_actions,axis=1)))
    features=np.asarray([base['freeze'],base['moving'],
        *np.concatenate([by_arm[a]['progress'] for a in ('safety_baseline','F13','F10')]),
        *[by_arm[a]['activation'] for a in ('safety_baseline','F13','F10')],
        base['rewrite'],base['mixed'],*(by_arm['F13']['progress']-by_arm['F10']['progress']),random],float)
    assert len(features)==len(NAMES) and np.isfinite(features).all()
    return features
