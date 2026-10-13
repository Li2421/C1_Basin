"""Geometry-override execution for the identifiability validation experiment.

Execution is the archived endpoint-matched goal-reference FF path used by
controller_response_families_v1 / controller_four_scene_mechanism_v1:
  analytic goal-reference field (REFERENCE parameters) with one affine operator
  -> OrthoFlow3 correction lifted into the FF field -> admissible-field safety
  projection -> Double-Bottleneck simulator, native evaluator and horizon.
The ONLY change is the Double-Bottleneck ``bottleneck_width`` passed to the
unmodified environment Config (its own __post_init__ validates the geometry).
No source file outside this directory is modified.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from diagnostics.controller_q_diversity_v1.runtime import native_runtime
from diagnostics.controller_response_families_v1.runtime import EndpointMatchedField, FAMILIES, REFERENCE

SCENE = 'double_bottleneck'
COLLISION_KEYS = ('wall_collision', 'obstacle_collision', 'outer_collision', 'agent_collision')


def base_runtime(width=None):
    """Native DB runtime; optionally with a different bottleneck width only."""
    base = native_runtime(SCENE)
    if width is not None:
        base.config = dataclasses.replace(base.config, bottleneck_width=float(width))
    return base


def controller(base, family):
    assert family in FAMILIES
    # AnalyticField copies base.config at construction: the override must precede it.
    return EndpointMatchedField(base, family)


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
        except Exception as ex:  # retained as numerical_failure, never rerun
            err = dict(step=t, type=type(ex).__name__, message=str(ex)[:500])
            term = 'numerical_failure'
            break
    summary = env.summary()
    collision = any(summary.get(k, False) for k in COLLISION_KEYS)
    row = dict(success=int(summary['collision_free_success'] and term == 'success' and err is None),
               deadlock=int(term == 'deadlock'), timeout=int(term == 'timeout'), collision=int(collision),
               numerical_failure=int(err is not None), episode_length=steps, outcome=term,
               elapsed_seconds=time.monotonic() - t0, first_action=first, projection_calls=calls,
               error=err, summary=summary)
    assert not (row['success'] and (row['collision'] or row['numerical_failure']))
    return row
