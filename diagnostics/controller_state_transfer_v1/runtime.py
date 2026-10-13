"""Toy Give-Way corridor-width override for the archived FF Flow-endpoint programs.

Identical execution to controller_ff_factorial_repair_v1 (parent worker loop with
FlowEndpointMatchedField); the ONLY change is Toy Config.corridor_width, validated by
the unmodified Config.__post_init__. No source file outside this directory is modified.
"""
from __future__ import annotations

import dataclasses, time
import numpy as np

from diagnostics.controller_q_diversity_v1.runtime import native_runtime
from diagnostics.controller_flow_endpoint_probe_v1.runtime import FlowEndpointMatchedField

SCENE = 'toy_give_way'
CONTROLLERS = ('flow_isotropic', 'flow_anisotropic')
COLLISION_KEYS = ('wall_collision', 'obstacle_collision', 'outer_collision', 'agent_collision')


def base_runtime(width=None):
    base = native_runtime(SCENE)
    if width is not None:
        base.config = dataclasses.replace(base.config, corridor_width=float(width))
    return base


def controller(base, name):
    assert name in CONTROLLERS
    return FlowEndpointMatchedField(base, name.removeprefix('flow_'))


def rollout(rt, state, eta, seed):
    """Byte-for-byte the loop of controller_response_families_v1.study.worker."""
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    env = rt.make_env(state)
    term, err, steps, calls, first, t0 = 'timeout', None, 0, 0, None, time.monotonic()
    for t in range(rt.config.max_steps):
        try:
            a, _, n = rt.action(env, key_for(state, seed, t), np.asarray(eta), 'field')
            calls += n; steps += 1
            if first is None:
                first = np.asarray(a).tolist()
            _, _, done, info = env.step(a)
            term = info['termination']
            if done:
                break
        except Exception as ex:
            err = dict(step=t, type=type(ex).__name__, message=str(ex)[:500]); term = 'numerical_failure'; break
    summary = env.summary()
    collision = any(summary.get(k, False) for k in COLLISION_KEYS)
    row = dict(success=int(summary['collision_free_success'] and term == 'success' and err is None),
               deadlock=int(term == 'deadlock'), timeout=int(term == 'timeout'), collision=int(collision),
               numerical_failure=int(err is not None), episode_length=steps, outcome=term,
               elapsed_seconds=time.monotonic() - t0, first_action=first, projection_calls=calls, error=err, summary=summary)
    assert not (row['success'] and (row['collision'] or row['numerical_failure']))
    return row
