import bootstrap
from experiment import *
from audit_supplement import V2,evidence
from registry import case,register
from swap_geometry import config,static_walls
from functools import lru_cache
import copy,numpy as np

NEW=('lit_swap8_free','lit_swap8_obstacles','lit_swap8_moving')
OLD=('lit_narrow_passage','lit_crowded_random','lit_circle_five')
@lru_cache(None)
def frozen():return read(HERE/'IMPLEMENTATION_FROZEN.receipt.json')
@lru_cache(None)
def history():return read(HERE/'HISTORY_EXECUTION_INDEX.json'),read(HERE/'HISTORY_TEMPLATES.json')
@lru_cache(None)
def candidates():
    p=ROOT/'diagnostics/controller_loco_real_flow_v1/protocol.json';evidence(p);return read(p)['eta']
def action(a,b=None):
    if a=='safety':return dict(key='safety',eta_a=[0.,0.,0.],eta_b=[0.,0.,0.],theta=None)
    a=int(a);b=a if b is None else int(b)
    return dict(key=f'eta{a}' if a==b else f'eta{a}_to_{b}',eta_a=candidates()[a],eta_b=candidates()[b],theta=None if a==b else .968)
def semantic(scene,ctl,s,seed,a):return canonical([scene,ctl['controller'],s['N'],s['index'],seed,[np.asarray(a['eta_a'],np.float64).tobytes().hex(),np.asarray(a['eta_b'],np.float64).tobytes().hex(),a['theta']]])

def make(scene,ctl,s,a,seed=0):
    if scene in OLD:
        old,templates=history();key=semantic(scene,ctl,s,seed,a)
        if key in old:
            p=Path(old[key]['path']);evidence(p);raw=read(p);c=copy.deepcopy(raw['case']);c.update(requested_action=a['key'],reuse_source=str(p),reuse_source_sha256=sha(p));return c
        template=templates[canonical([scene,ctl['controller'],s['N'],s['index']])]
        # Use original scientific source profile; 32 bitwise replays certify relocated execution.
        payload={k:v for k,v in template['controller_payload'].items() if k not in ('method','eta_b','theta')}
        if np.any(a['eta_a']) or np.any(a['eta_b']):payload.update(correction_semantics='contact_conflict_relative_gate_20261021',correction_revision_receipt_sha256=sha(V2/'CORRECTION_REVISION.receipt.json'),correction_basis='orthoflow3_local_contact_gated',arrived_flag_broadcast=True)
        c=case(s,'flow',a['key'],template['scenario_payload'],payload,seed,a['eta_a'],a['eta_b'],a['theta'],template['state_provenance'])
        c.update(safety_layer='distributed',rng_state_uid=template.get('rng_state_uid',template['state_uid']),requested_action=a['key']);return c
    cfg=config(scene,s['N']).to_dict();profile=frozen()
    scenario=dict(version='swap8_single_integrator_v1',config=cfg,walls=static_walls(scene).tolist(),geometry_sha256=sha(HERE/'swap_geometry.py'),moving_policy='state-fixed random waypoints seed2026102313' if scene.endswith('moving') else None)
    payload=dict(nominal=ctl,execution_sources=profile['files'],safety_layer='distributed',sensing_radius=.5,correction_semantics='contact_conflict_relative_gate_20261021',correction_basis='orthoflow3_local_contact_gated',arrived_flag_broadcast=True,precision='jax_x64_false',restoration='speed_only_then_safety_minimax')
    c=case(s,'swap8_si',a['key'],scenario,payload,seed,a['eta_a'],a['eta_b'],a['theta'],dict(sampler_seed=2026102312 if scene.endswith('obstacles') else 2026102311,index=s['index'],agent=s.get('agent'),parent_N=s.get('parent_N'),moving_path_parent=s.get('parent_physical_identity')))
    if scene in ('lit_swap8_free','lit_swap8_moving'):
        # Match free/moving robot RNG on parent physical identity, independent of obstacle.
        c['rng_state_uid']=uid('swap_rng',dict(physical_parent=s['parent_physical_identity'],agent=s.get('agent'),N=s['N']))
    c.update(safety_layer='distributed',requested_action=a['key']);return c

def capability():
    rows=[]
    for scene in NEW:
        for ctl in read(HERE/'CONTROLLER_MANIFEST.json'):
            for s in read(HERE/f'FLOW_STATES_{scene}.json'):
                for i in range(8):
                    solo=dict(s,N=1,parent_N=8,agent=i,parent_positions=s['positions'],positions=[s['positions'][i]],goals=[s['goals'][i]],velocities=[[0.,0.]])
                    for seed in (0,1):rows.append(make(scene,ctl,solo,action('safety'),seed))
    assert len(rows)==73728;register('capability',rows)
if __name__=='__main__':capability()
