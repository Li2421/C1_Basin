"""First-event masks using the authoritative environment, outside derivatives.

Only discrete event decisions are detached. Positions and executed controls
remain differentiable in the rollout. The callback is pure and reconstructs
the minimal state needed by GiveWayEnv.step; it never retains a live env.
"""
import math
import jax
import numpy as np

from single_integrator.environment import GiveWayEnv
from single_integrator.outcomes import first_event

EVENTS = {'running': 0, 'success': 1, 'wall_collision': 2,
          'agent_collision': 3, 'safe_deadlock': 4, 'other_timeout': 5}


def _enabled_terminal_code(info, plant):
    """Classify only events that are enabled to terminate this episode.

    This differs from :func:`outcomes.first_event` only for diagnostic
    configurations with a disabled ``terminate_on_*`` flag.  The environment
    remains authoritative about whether the step terminated.
    """
    if info['termination'] == 'running':
        return EVENTS['running']
    if plant.terminate_on_collision and info['agent_collision']:
        return EVENTS['agent_collision']
    if plant.terminate_on_collision and info['wall_collision']:
        return EVENTS['wall_collision']
    if plant.terminate_on_success and info['task_success']:
        return EVENTS['success']
    if plant.terminate_on_deadlock and info['deadlock']:
        return EVENTS['safe_deadlock']
    if info['termination'] == 'timeout':
        return EVENTS['other_timeout']
    raise AssertionError(f"unclassified enabled terminal event: {info['termination']}")


def _monitor_oracle(p, u, lo_errors, hi_errors, since, first_deadlock,
                    enabled, step_value, plant):
    """Pure NumPy mirror of one authoritative monitor/termination update."""
    step = int(step_value)
    n = len(p)
    codes = np.zeros(n, np.int32)
    updated_since = np.array(since, dtype=np.int32, copy=True)
    updated_first = np.array(first_deadlock, dtype=np.int32, copy=True)
    raw_deadlock = np.zeros(n, bool)
    candidate = np.zeros(n, bool)
    raw_flags = np.zeros((n, 4), bool)  # success, wall, agent, done
    window_start = max(0., step + 1 - plant.progress_window_seconds / plant.dt)
    lo, hi = int(math.floor(window_start)), int(math.ceil(window_start))
    for i in range(n):
        if not enabled[i]:
            continue
        env = GiveWayEnv(plant)
        env.positions = np.array(p[i], dtype=np.float64, copy=True)
        env.step_count = step
        env.distance_history = [np.zeros(2)] * (step + 1)
        if lo <= step:
            env.distance_history[lo] = np.asarray(lo_errors[i])
        if hi <= step:
            env.distance_history[hi] = np.asarray(hi_errors[i])
        env.candidate_since = None if since[i] < 0 else int(since[i])
        env.first_deadlock_step = (None if first_deadlock[i] < 0
                                   else int(first_deadlock[i]))
        _, _, done, info = env.step(u[i])
        updated_since[i] = (-1 if env.candidate_since is None
                            else env.candidate_since)
        updated_first[i] = (-1 if env.first_deadlock_step is None
                            else env.first_deadlock_step)
        raw_deadlock[i] = info['deadlock']
        candidate[i] = info['candidate_deadlock']
        raw_flags[i] = (info['task_success'], info['wall_collision'],
                        info['agent_collision'], done)
        if done:
            codes[i] = _enabled_terminal_code(info, plant)
    return (codes, updated_since, updated_first, raw_deadlock, candidate,
            raw_flags)


def monitor_step(before, controls, past_lo, past_hi, candidate_since,
                 first_deadlock_step, alive, step, plant):
    """Exact raw-monitor update plus enabled first-terminal-event code.

    ``alive`` and ``first_deadlock_step`` are pre-action values.  The returned
    raw deadlock is independent of ``terminate_on_deadlock``.  All discrete
    monitor outputs are intentionally outside derivatives; state and controls
    in the surrounding rollout remain differentiable.
    """
    def oracle(p, u, lo_errors, hi_errors, since, first, enabled, step_value):
        return _monitor_oracle(p, u, lo_errors, hi_errors, since, first,
                               enabled, step_value, plant)

    args = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        (before, controls, past_lo, past_hi, candidate_since,
         first_deadlock_step, alive,
         np.int32(step) if isinstance(step, int) else step))
    n = len(before)
    one = jax.ShapeDtypeStruct((n,), np.int32)
    flag = jax.ShapeDtypeStruct((n,), np.bool_)
    flags = jax.ShapeDtypeStruct((n, 4), np.bool_)
    return jax.pure_callback(oracle, (one, one, one, flag, flag, flags),
                             *args)


def event_step(before, controls, past_lo, past_hi, candidate_since, alive, step, plant):
    def oracle(p, u, lo_errors, hi_errors, since, enabled, step_value):
        step=int(step_value)
        codes=np.zeros(len(p),np.int32);updated=np.array(since,dtype=np.int32,copy=True)
        window_start=max(0.,step+1-plant.progress_window_seconds/plant.dt)
        lo,hi=int(math.floor(window_start)),int(math.ceil(window_start))
        for i in range(len(p)):
            if not enabled[i]:
                continue
            env=GiveWayEnv(plant)
            env.positions=np.array(p[i],dtype=np.float64,copy=True)
            env.step_count=step
            # step() reads only the two interpolation endpoints, then appends
            # the new state's distances. All other slots are irrelevant.
            env.distance_history=[np.zeros(2)]*(step+1)
            if lo<=step:env.distance_history[lo]=np.asarray(lo_errors[i])
            if hi<=step:env.distance_history[hi]=np.asarray(hi_errors[i])
            env.candidate_since=None if since[i]<0 else int(since[i])
            _,_,done,info=env.step(u[i])
            updated[i]=-1 if env.candidate_since is None else env.candidate_since
            if done:codes[i]=EVENTS[first_event(info)]
        return codes,updated
    args=jax.tree_util.tree_map(jax.lax.stop_gradient,
                                (before,controls,past_lo,past_hi,candidate_since,alive,np.int32(step) if isinstance(step,int) else step))
    shape=jax.ShapeDtypeStruct((len(before),),np.int32)
    return jax.pure_callback(oracle,(shape,shape),*args)
