"""Exact full-horizon continuation runner for true-Q geometry diagnosis."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from diagnostics.cl_fhcb.closed_loop import AugmentedState, DiagnosticCorrector, DiagnosticPhi
from single_integrator.cbf import CBFConfig, barrier_constraints, project_velocity
from single_integrator.environment import Config, GiveWayEnv, bounded_nominal


def restore_environment(state: AugmentedState, config: Config) -> GiveWayEnv:
    if state.terminal != "running":
        raise ValueError("continuations require a nonterminal augmented state")
    env = GiveWayEnv(config)
    env.reset(state.positions)
    env.positions = np.asarray(state.positions, dtype=np.float64).copy()
    env.velocities = np.asarray(state.last_velocity, dtype=np.float64).copy()
    env.step_count = int(state.step)
    env.distance_history = [np.zeros(2, dtype=np.float64) for _ in range(state.step + 1)]
    for offset, errors in enumerate(np.asarray(state.error_history, dtype=np.float64)):
        absolute = state.history_start_step + offset
        if 0 <= absolute <= state.step:
            env.distance_history[absolute] = errors.copy()
    # The current endpoint must always be exact.
    env.distance_history[state.step] = np.linalg.norm(env.goals - env.positions, axis=-1)
    env.candidate_since = state.candidate_since
    env.stuck_timer = (
        0.0 if state.candidate_since is None
        else (state.step - state.candidate_since) * config.dt
    )
    env.max_stuck_timer = env.stuck_timer
    env.ever_candidate_deadlock = state.candidate_since is not None
    env.first_success_step = None
    env.first_deadlock_step = None
    env.first_wall_collision_step = None
    env.first_agent_collision_step = None
    env.done = False
    return env


def active_signature(A, lower, action, max_speed, tolerance=1e-6):
    flat = np.asarray(action, dtype=np.float64).reshape(4)
    cbf_active = (np.asarray(A) @ flat - np.asarray(lower)) <= tolerance
    speed_active = np.abs(np.linalg.norm(flat.reshape(2, 2), axis=-1) - max_speed) <= tolerance
    return np.concatenate((cbf_active, speed_active))


def _event(info):
    name = info["termination"]
    if name not in ("running", "collision", "success", "deadlock", "timeout"):
        raise AssertionError(name)
    return name


def rollout_from_state(
    policy,
    state: AugmentedState,
    config: Config,
    phi: DiagnosticPhi,
    flow_seed: int,
    rollout_index: int,
    cbf_config: CBFConfig | None = None,
) -> dict[str, np.ndarray | str | int | float]:
    """Continue the same state-feedback G_phi through the authoritative horizon."""

    cbf_config = cbf_config or CBFConfig()
    env = restore_environment(state, config)
    corrector = DiagnosticCorrector(phi)
    episode_key = jax.random.fold_in(jax.random.PRNGKey(flow_seed), rollout_index)
    fields = {name: [] for name in (
        "positions_before", "positions_after", "last_velocity_before", "u_flow",
        "u_safe", "g", "w", "u_exec", "event", "candidate_since_before",
        "stuck_timer_after", "window_progress_after", "first_active", "second_active",
        "first_min_cbf_residual", "second_min_cbf_residual", "flow_key_data",
    )}
    for absolute_step in range(state.step, config.max_steps):
        observation = env.observation()
        positions_before = env.positions.copy()
        velocity_before = env.velocities.copy()
        candidate_since = -1 if env.candidate_since is None else int(env.candidate_since)
        key = jax.random.fold_in(episode_key, absolute_step)
        raw = np.asarray(
            policy.sample_actions(jnp.asarray(observation[None]), seed=key)[0],
            dtype=np.float64,
        )
        u_flow = bounded_nominal(raw, config.max_speed)
        A, lower, _ = barrier_constraints(env.snapshot(), cbf_config)
        u_safe, _ = project_velocity(u_flow, A, lower, config.max_speed, cbf_config)
        correction = corrector(observation, u_safe, config.max_speed)
        candidate = u_safe + correction
        u_exec, _ = project_velocity(candidate, A, lower, config.max_speed, cbf_config)
        _, _, done, info = env.step(u_exec)
        event = _event(info)
        values = {
            "positions_before": positions_before,
            "positions_after": env.positions.copy(),
            "last_velocity_before": velocity_before,
            "u_flow": u_flow.copy(),
            "u_safe": u_safe.copy(),
            "g": correction.copy(),
            "w": candidate.copy(),
            "u_exec": u_exec.copy(),
            "event": event,
            "candidate_since_before": candidate_since,
            "stuck_timer_after": float(info["stuck_timer"]),
            "window_progress_after": np.asarray(info["window_progress"], dtype=np.float64),
            "first_active": active_signature(A, lower, u_safe, config.max_speed),
            "second_active": active_signature(A, lower, u_exec, config.max_speed),
            "first_min_cbf_residual": float(np.min(A @ u_safe.reshape(4) - lower)),
            "second_min_cbf_residual": float(np.min(A @ u_exec.reshape(4) - lower)),
            "flow_key_data": np.asarray(jax.random.key_data(key)),
        }
        for name in fields:
            fields[name].append(values[name])
        if done:
            break
    if corrector.call_count != len(fields["event"]):
        raise AssertionError("G_phi was not recomputed at every continuation step")
    if not fields["event"] or fields["event"][-1] == "running":
        raise AssertionError("continuation did not terminate")
    arrays = {name: np.asarray(values) for name, values in fields.items()}
    arrays.update({
        "outcome": str(fields["event"][-1]),
        "start_step": int(state.step),
        "terminal_step": int(state.step + len(fields["event"])),
        "phi": np.asarray(phi.vector, dtype=np.float64),
        "flow_seed": int(flow_seed),
    })
    return arrays


def diagnostic_state_features(state: AugmentedState, config: Config, goals: np.ndarray):
    errors = np.linalg.norm(np.asarray(goals) - state.positions, axis=-1)
    ready = len(state.error_history) >= 41 and state.step >= 40
    progress = (
        state.error_history[-41] - state.error_history[-1]
        if ready else np.full(2, np.nan)
    )
    return {
        "max_task_error": float(np.max(errors)),
        "sum_task_error": float(np.sum(errors)),
        "recent_progress": progress.tolist(),
        "recent_progress_sum": float(np.nansum(progress)) if ready else None,
        "relative_longitudinal_order": float(state.positions[0, 0] - state.positions[1, 0]),
        "inter_agent_distance": float(np.linalg.norm(state.positions[0] - state.positions[1])),
        "joint_speed_max": float(np.linalg.norm(state.last_velocity, axis=-1).max()),
        "strict_timer_age": (
            0.0 if state.candidate_since is None
            else float((state.step - state.candidate_since) * config.dt)
        ),
        "candidate_active": state.candidate_since is not None,
        "time_to_go_steps": int(config.max_steps - state.step),
        "time_to_go_seconds": float((config.max_steps - state.step) * config.dt),
    }
