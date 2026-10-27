import bootstrap
from experiment import *
from case_factory import make as original_make,action,candidates,NEW,OLD
from registry import case,register
from swap_geometry_user_fix_20261023 import config,static_walls
from functools import lru_cache

REVISED=('lit_swap8_obstacles','lit_swap8_moving')
@lru_cache(None)
def revision_profile():return read(HERE/'USER_FIX_IMPLEMENTATION_FROZEN_20261023.receipt.json')
def make(scene,ctl,s,a,seed=0):
    base=original_make(scene,ctl,s,a,seed)
    if scene not in REVISED:return base
    revision='static_straight_corridors_20261023' if scene.endswith('obstacles') else 'moving_worstcase_speed_bound_20261023'
    scenario=dict(base['scenario_payload'],version='swap8_single_integrator_user_fix_20261023',geometry_sha256=sha(HERE/'swap_geometry_user_fix_20261023.py'),walls=static_walls(scene).tolist(),user_scene_revision=revision)
    if scene.endswith('moving'):scenario.update(moving_constraint=dict(normal='outward unit radial',speed_bound=.25,gamma=read(HERE/'MOVING_USER_FIX_UNIT_GATE.json')['selected_gamma'],epsilon=.001001,one_step_radius_inflation=0.))
    payload={k:v for k,v in base['controller_payload'].items() if k not in ('method','eta_b','theta')};payload.update(execution_sources=revision_profile()['files'],scene_revision=revision,original_implementation_receipt_sha256=sha(HERE/'IMPLEMENTATION_FROZEN.receipt.json'))
    provenance=dict(base['state_provenance'],user_scene_revision=revision,original_physical_state_uid=base['state_uid'],robot_samples_unchanged=True)
    c=case(s,'swap8_si_user_fix_20261023',a['key'],scenario,payload,seed,a['eta_a'],a['eta_b'],a['theta'],provenance)
    c.update(safety_layer='distributed',requested_action=a['key'],scene_revision=revision,rng_state_uid=base.get('rng_state_uid',base['state_uid']))
    assert c['rollout_uid']!=base['rollout_uid'] and c['state_uid']!=base['state_uid'];return c

def capability():
    rows=[]
    for scene in REVISED:
        for ctl in read(HERE/'CONTROLLER_MANIFEST.json'):
            for s in read(HERE/f'FLOW_STATES_{scene}.json'):
                for i in range(8):
                    solo=dict(s,N=1,parent_N=8,agent=i,parent_positions=s['positions'],positions=[s['positions'][i]],goals=[s['goals'][i]],velocities=[[0.,0.]])
                    for seed in (0,1):rows.append(make(scene,ctl,solo,action('safety'),seed))
    assert len(rows)==49152;register('capability_user_fix_20261023',rows)
if __name__=='__main__':capability()
