"""Authoritative CPU rollout for the isolated CL-FHCB construction.

This module intentionally does not import an old C1 risk implementation.  It
uses only the frozen Flow sampler, the two hard CBF projections, and the
authoritative environment/monitor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.cbf import CBFConfig, barrier_constraints, project_velocity
from single_integrator.environment import Config, GiveWayEnv, bounded_nominal


@dataclass(frozen=True)
class DiagnosticPhi:
    """Three shared scalar gains for state-dependent four-vector bases."""

    goal_feedback: float
    safe_feedback: float
    relative_feedback: float
    name: str = "unnamed"

    def __post_init__(self) -> None:
        values = np.asarray(self.vector, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("diagnostic phi must be finite")

    @property
    def vector(self) -> tuple[float, float, float]:
        return (
            float(self.goal_feedback),
            float(self.safe_feedback),
            float(self.relative_feedback),
        )


class DiagnosticCorrector:
    """Small deterministic replacement when no trained final G_phi is present.

    The same three gains are reused for the whole episode.  The four-vector is
    recomputed from the current observation and current first projection:

      g = phi_goal * bounded(goal displacement)
        + phi_safe * u_safe
        + phi_relative * bounded(away-from-other displacement).

    All bases are agent-exchange equivariant and contain no hard-coded yielding,
    passing-side, or agent-priority behavior.
    """

    output_dim = 4
    stochastic = False

    def __init__(self, phi: DiagnosticPhi):
        self.phi = phi
        self.call_count = 0

    @staticmethod
    def _bounded_rows(vectors: np.ndarray, max_speed: float) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors * np.minimum(1.0, max_speed / np.maximum(norms, 1e-30))

    def __call__(
        self, observation: np.ndarray, u_safe: np.ndarray, max_speed: float
    ) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float64)
        u_safe = np.asarray(u_safe, dtype=np.float64)
        if observation.shape != (2, 10) or u_safe.shape != (2, 2):
            raise ValueError("expected observation [2,10] and safe action [2,2]")
        goal_basis = self._bounded_rows(observation[:, 4:6], max_speed)
        # observation[:, 6:8] is other-minus-self; negate it for a generic
        # symmetric separation response.
        relative_basis = self._bounded_rows(-observation[:, 6:8], max_speed)
        a, b, c = self.phi.vector
        self.call_count += 1
        return a * goal_basis + b * u_safe + c * relative_basis


@dataclass(frozen=True)
class AugmentedState:
    """Exact state needed by plant, controller, and strict-deadlock monitor."""

    positions: np.ndarray
    last_velocity: np.ndarray
    step: int
    error_history: np.ndarray
    history_start_step: int
    candidate_since: Optional[int]
    terminal: str

    @property
    def remaining_steps(self) -> int:
        raise AttributeError("remaining steps depend on the frozen Config")


@dataclass(frozen=True)
class ClosedLoopStep:
    state: AugmentedState
    observation: np.ndarray
    flow_key_data: np.ndarray
    u_flow: np.ndarray
    u_safe: np.ndarray
    correction: np.ndarray
    candidate: np.ndarray
    u_exec: np.ndarray
    first_projection_status: str
    second_projection_status: str
    first_min_cbf_residual: float
    second_min_cbf_residual: float
    next_state: AugmentedState
    event: str
    monitor: dict


@dataclass(frozen=True)
class ClosedLoopTrace:
    rollout_id: str
    pair_id: int
    flow_seed: int
    phi: DiagnosticPhi
    initial_state: AugmentedState
    steps: tuple[ClosedLoopStep, ...]
    outcome: str


def snapshot_augmented(env: GiveWayEnv, terminal: str = "running") -> AugmentedState:
    # Forty transitions require 41 endpoint error samples.  Keeping exactly the
    # tail is sufficient for all future frozen-window interpolation.
    history = np.asarray(env.distance_history[-41:], dtype=np.float64).copy()
    return AugmentedState(
        positions=env.positions.copy(),
        last_velocity=env.velocities.copy(),
        step=int(env.step_count),
        error_history=history,
        history_start_step=int(env.step_count - len(history) + 1),
        candidate_since=(None if env.candidate_since is None else int(env.candidate_since)),
        terminal=str(terminal),
    )


def state_metrics(state: AugmentedState, config: Config, goals: np.ndarray) -> dict:
    current_errors = np.linalg.norm(np.asarray(goals) - state.positions, axis=-1)
    window_steps = int(round(config.progress_window_seconds / config.dt))
    window_ready = state.step >= window_steps and len(state.error_history) >= window_steps + 1
    if window_ready:
        recent_progress = state.error_history[-(window_steps + 1)] - state.error_history[-1]
    else:
        recent_progress = np.full(2, np.nan, dtype=np.float64)
    candidate_age = (
        0.0
        if state.candidate_since is None
        else (state.step - state.candidate_since) * config.dt
    )
    return {
        "step": state.step,
        "remaining_steps": max(0, config.max_steps - state.step),
        "positions": state.positions.copy(),
        "last_velocity": state.last_velocity.copy(),
        "goal_errors": current_errors,
        "window_ready": bool(window_ready),
        "recent_progress": recent_progress,
        "candidate_active": state.candidate_since is not None,
        "candidate_age": float(candidate_age),
        "max_speed": float(np.linalg.norm(state.last_velocity, axis=-1).max()),
        "terminal": state.terminal,
    }


def _event_name(info: dict) -> str:
    termination = info["termination"]
    if termination == "collision":
        return "collision"
    if termination == "success":
        return "success"
    if termination == "deadlock":
        return "deadlock"
    if termination == "timeout":
        return "timeout"
    if termination == "running":
        return "running"
    raise AssertionError(f"unknown environment termination {termination!r}")


def rollout_closed_loop(
    policy,
    initial_positions: np.ndarray,
    config: Config,
    phi: DiagnosticPhi,
    flow_seed: int,
    rollout_index: int,
    pair_id: int = -1,
    cbf_config: CBFConfig | None = None,
) -> ClosedLoopTrace:
    """Run one complete first-event episode with persistent closed-loop G_phi."""

    if not all(
        (
            config.terminate_on_collision,
            config.terminate_on_success,
            config.terminate_on_deadlock,
        )
    ):
        raise ValueError("CL-FHCB requires authoritative first-event termination")
    cbf_config = cbf_config or CBFConfig()
    env = GiveWayEnv(config)
    env.reset(initial_positions)
    corrector = DiagnosticCorrector(phi)
    episode_key = jax.random.fold_in(jax.random.PRNGKey(flow_seed), rollout_index)
    initial = snapshot_augmented(env)
    records: list[ClosedLoopStep] = []

    for step in range(config.max_steps):
        state = snapshot_augmented(env)
        observation = env.observation()
        step_key = jax.random.fold_in(episode_key, step)
        raw = np.asarray(
            policy.sample_actions(jnp.asarray(observation[None]), seed=step_key)[0],
            dtype=np.float64,
        )
        u_flow = bounded_nominal(raw, config.max_speed)
        A, lower, _ = barrier_constraints(env.snapshot(), cbf_config)
        u_safe, first_status = project_velocity(
            u_flow, A, lower, config.max_speed, cbf_config
        )
        correction = corrector(observation, u_safe, config.max_speed)
        candidate = u_safe + correction
        u_exec, second_status = project_velocity(
            candidate, A, lower, config.max_speed, cbf_config
        )
        _, _, done, info = env.step(u_exec)
        event = _event_name(info)
        next_state = snapshot_augmented(env, event if done else "running")
        records.append(
            ClosedLoopStep(
                state=state,
                observation=np.asarray(observation, dtype=np.float64).copy(),
                flow_key_data=np.asarray(jax.random.key_data(step_key)).copy(),
                u_flow=u_flow.copy(),
                u_safe=u_safe.copy(),
                correction=correction.copy(),
                candidate=candidate.copy(),
                u_exec=u_exec.copy(),
                first_projection_status=str(first_status),
                second_projection_status=str(second_status),
                first_min_cbf_residual=float(np.min(A @ u_safe.reshape(4) - lower)),
                second_min_cbf_residual=float(np.min(A @ u_exec.reshape(4) - lower)),
                next_state=next_state,
                event=event,
                monitor={
                    "window_ready": bool(info["window_ready"]),
                    "window_progress": np.asarray(info["window_progress"], dtype=np.float64).copy(),
                    "candidate_deadlock": bool(info["candidate_deadlock"]),
                    "stuck_timer": float(info["stuck_timer"]),
                    "deadlock": bool(info["deadlock"]),
                },
            )
        )
        if done:
            break

    if corrector.call_count != len(records):
        raise AssertionError("G_phi was not recomputed exactly once per physical step")
    if not records or records[-1].event == "running":
        raise AssertionError("full-horizon rollout did not terminate")
    return ClosedLoopTrace(
        rollout_id=f"pair{pair_id:03d}_phi-{phi.name}_seed{flow_seed}_rid{rollout_index}",
        pair_id=int(pair_id),
        flow_seed=int(flow_seed),
        phi=phi,
        initial_state=initial,
        steps=tuple(records),
        outcome=records[-1].event,
    )
