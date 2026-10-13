"""Existing masked entity adapter restricted to physical state only.

The source templates provide fixed geometry and robot dimensions. No Flow,
controller response, context tensor, scenario ID or future rollout is queried.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from diagnostics.orthoflow3_unified_representation_v1 import representation as rep

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'diagnostics/orthoflow3_loso_root_cause_v1/all_source_states.json'
SCENES = {'toy_give_way':'toy_giveway', 'double_bottleneck':'double_bottleneck',
          'four_way_intersection':'four_way_intersection', 'ring_exchange':'ring_exchange'}


def templates():
    rows = json.loads(SOURCE.read_text())
    result = {}
    for requested, source_name in SCENES.items():
        examples = [r['physical'] for r in rows if r['scenario'] == source_name]
        assert examples, requested
        first = examples[0]
        assert all(e['obstacles'] == first['obstacles'] for e in examples), requested
        assert all(e['radius'] == first['radius'] and e['goal_tolerance'] == first['goal_tolerance']
                   for e in examples), requested
        result[requested] = dict(obstacles=copy.deepcopy(first['obstacles']),
                                 radius=copy.deepcopy(first['radius']),
                                 goal_tolerance=copy.deepcopy(first['goal_tolerance']))
    return result


_TEMPLATES = None


def physical_scene(scene_name, physical):
    global _TEMPLATES
    assert scene_name in SCENES
    if _TEMPLATES is None:
        _TEMPLATES = templates()
    p = {k:np.asarray(physical[k], float) for k in ('positions','velocities','goals')}
    n = len(p['positions'])
    assert p['positions'].shape == p['velocities'].shape == p['goals'].shape == (n,2)
    assert physical['timestep'] == 0
    template = _TEMPLATES[scene_name]
    assert len(template['radius']) == n
    # rep.entities needs these fields to build its existing physical geometry
    # channels; controller-related and scenario-routing fields are neutral.
    return dict(positions=p['positions'].tolist(),
                velocities=p['velocities'].tolist(),goals=p['goals'].tolist(),
                flow=np.zeros((n,2),float).tolist(),
                radius=template['radius'],goal_tolerance=template['goal_tolerance'],
                obstacles=template['obstacles'],
                policy_origin=[0.,0.],policy_axis=[1.,0.],
                policy_axis_required=False,flow_committed=False,
                remaining_fraction=1.,remaining_seconds=0.,
                dt=0.,max_speed=1.,max_seconds=0.,
                wall_margin=0.,agent_margin=0.,monitor_active=False,
                progress_window_seconds=0.,deadlock_hold_seconds=0.,
                progress_epsilon=0.,speed_epsilon_fraction=0.,
                monitor_elapsed=0.,monitor_history_empty=True)


def entities(scene_name, physical):
    x = rep.entities(physical_scene(scene_name, physical))
    x['agents'] = x['agents'].copy()
    x['agents'][:,3:7] = 0. # nominal and bounded Flow channels
    x['agents'][:,10:15] = 0. # policy-frame/scenario-routing channels
    x['globals'] = np.zeros_like(x['globals']) # controller/monitor constants
    assert np.all(x['agents'][:,3:7] == 0.)
    assert np.all(x['agents'][:,10:15] == 0.)
    assert np.all(x['globals'] == 0.)
    return x


def batch(rows):
    return rep.batch([entities(r['scene'], r['physical']) for r in rows])
