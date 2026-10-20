"""Restore stored probe states and infer missing arrays without rerunning physics."""
from common import *
from history_audit import signature
from ring_exchange.environment import point_segment_distance
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for


def restore(env,positions,velocities,step):
    augmented=env.augmented_state()
    augmented.update(positions=positions,last_applied_velocity=velocities,step=step,done=False)
    env.restore_augmented_state(augmented)


def clearance(first,second,cfg):
    obstacle=point_segment_distance(np.zeros(2),first,second)-cfg.obstacle_radius-cfg.agent_radius
    outer=np.minimum(cfg.outer_radius-cfg.agent_radius-np.linalg.norm(first,axis=-1),
                     cfg.outer_radius-cfg.agent_radius-np.linalg.norm(second,axis=-1))
    ii,jj=np.triu_indices(4,1)
    pair=point_segment_distance(np.zeros(2),first[ii]-first[jj],second[ii]-second[jj])-2*cfg.agent_radius
    return min(float(obstacle.min())-cfg.collision_margin,float(outer.min())-cfg.collision_margin,
               float(pair.min())-cfg.agent_collision_margin)


def load_prefix(c,p,base,env,state):
    path=HERE/'PROBE_PREFIX_INVENTORY.json'
    if not path.exists():return None
    rows=read(path)['prefixes']
    matching=[r for r in rows if r['controller']==c['base_controller_name'] and r['arm']==c['arm']
              and r['state_signature']==signature(state) and r['placement']==c['placement']]
    if not matching:return None
    source=matching[0]
    assert source['checkpoint_sha256']==p['profiles'][c['base_controller_name']]['payload']['flow_checkpoint_sha256']
    assert sha(source['raw_path'])==source['raw_sha256'] and sha(source['trace_path'])==source['trace_sha256']
    record=read(source['raw_path'])
    with np.load(source['trace_path'],allow_pickle=False) as z:
        positions=z['positions'].copy();actions=z['actions'].copy();active=z['projection_active'].copy()
        assert np.array_equal(z['goals'],env.goals)
    n=len(actions)
    assert n==source['old_physical_steps'] and len(positions)==n+1 and n<=200
    assert positions[0].tobytes()==env.positions.tobytes()
    velocities=np.concatenate((env.velocities[None],actions),axis=0)
    nominal=[];swept=[]
    for step in range(n):
        restore(env,positions[step],velocities[step],step)
        _,action,_=base.nominal(env,key_for(state,c['seed'],step))
        nominal.append(np.asarray(action).copy())
        swept.append(clearance(positions[step],positions[step+1],env.config))
    restore(env,positions[-1],velocities[-1],n)
    return dict(positions=list(positions),velocities=list(velocities),actions=list(actions),nominal=nominal,
                active=list(active),swept=swept,error=record['error'],source=source,
                start_step=n,can_continue=bool(record['complete']))
