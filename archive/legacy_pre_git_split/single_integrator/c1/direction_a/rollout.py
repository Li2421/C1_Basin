"""Operational Direction A rollout through frozen Flow-BC, Safety and monitor."""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from single_integrator.c1.differentiable_rollout import (ResidualFlowField,
                                                          bounded_nominal)
from single_integrator.cbf import (CBFConfig, CBFSolverError, barrier_constraints,
                                   project_velocity)
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace
from single_integrator.environment import GiveWayEnv

from .estimators import deformation_estimate, trajectory_scores
from .policy import sample_residual


class DirectionARolloutError(RuntimeError):
    """Numerical/safety failure that must never be converted to an outcome."""


@dataclass(frozen=True)
class ProjectionAudit:
    status: str
    min_linear_residual: float
    max_speed_excess: float


def validate_projected_action(action, A, lower, max_speed, status, cbf):
    action = np.asarray(action, dtype=np.float64).reshape(4)
    A, lower = np.asarray(A), np.asarray(lower)
    if not np.isfinite(action).all():
        raise DirectionARolloutError("nonfinite projected action")
    if status not in {"nominal_feasible", "solved", "solved_kkt_certified"}:
        raise DirectionARolloutError(f"unaccepted projection status: {status}")
    linear = A@action-lower
    speed_excess = float(np.linalg.norm(action.reshape(2, 2), axis=-1).max()-max_speed)
    if linear.min() < -cbf.feasibility_tol or speed_excess > cbf.speed_tol:
        raise DirectionARolloutError(
            f"unsafe projection: min_linear={linear.min()}, speed_excess={speed_excess}")
    return ProjectionAudit(status, float(linear.min()), speed_excess)


def compose_double_projection(flow_control, residual, projection):
    """Small exact controller seam used by both the adapter and contract tests."""
    safe = projection(np.asarray(flow_control, dtype=np.float64).reshape(4))
    applied = projection(np.asarray(safe, dtype=np.float64).reshape(4)
                         + np.asarray(residual, dtype=np.float64).reshape(4))
    return np.asarray(safe).reshape(4), np.asarray(applied).reshape(4)


def _terminal_label(info, plant):
    if plant.terminate_on_collision and info["agent_collision"]:
        return "agent_collision"
    if plant.terminate_on_collision and info["wall_collision"]:
        return "wall_collision"
    if plant.terminate_on_success and info["task_success"]:
        return "success"
    if plant.terminate_on_deadlock and info["deadlock"]:
        return "safe_deadlock"
    if info["termination"] == "timeout":
        return "other_timeout"
    raise DirectionARolloutError("terminal step has no enabled first event")


class FrozenDirectionAController:
    """Single-step controller; only ``params`` belongs to G_phi."""

    def __init__(self, baseline, model, params, distribution, plant,
                 cbf=None, projector=project_velocity):
        self.baseline = baseline
        self.model = model
        self.params = params
        self.distribution = distribution
        self.plant = plant
        self.cbf = cbf or CBFConfig()
        self.projector = projector
        self._frozen_flow = ResidualFlowField(baseline, model)

    def _flow_sample(self, observations, noise):
        # Call the existing frozen implementation rather than maintaining a
        # second copy of its Flow integration or normalization semantics.
        return self._frozen_flow.baseline_sample(observations, noise)

    def act(self, env, tape, split, scenario, continuation, time):
        if env.done or env.step_count != time:
            raise DirectionARolloutError("controller time must equal live environment step")
        observation = env.observation()[None]
        flow_noise = tape.flow_noise(split, scenario, continuation, time)[None]
        raw = np.asarray(self._flow_sample(observation, flow_noise))[0]
        nominal = np.asarray(bounded_nominal(jnp.asarray(raw), self.plant.max_speed))
        A, lower, _ = barrier_constraints(env.snapshot(), self.cbf)

        def solve(target):
            try:
                action, status = self.projector(target, A, lower,
                                                self.plant.max_speed, self.cbf)
            except (CBFSolverError, ValueError, FloatingPointError) as exc:
                raise DirectionARolloutError(f"projection failure: {exc}") from exc
            audit = validate_projected_action(action, A, lower,
                                              self.plant.max_speed, status, self.cbf)
            return np.asarray(action).reshape(4), audit

        safe, first_audit = solve(nominal)
        flattened, _ = self.baseline._flatten(
            jnp.asarray(observation, dtype=jnp.float32),
            jnp.zeros((1, 4), dtype=jnp.float32))
        outputs = self.model.apply(
            self.params, jnp.asarray(safe[None]), jnp.zeros((1, 1)),
            jnp.asarray(flattened))[0:3]
        # Remove only the batch dimension; all three heads retain their values.
        outputs = (outputs[0][0], outputs[1][0], outputs[2][0])
        gate, residual, values = sample_residual(
            outputs,
            tape.gate_uniform(split, scenario, continuation, time),
            tape.gaussian_residual(split, scenario, continuation, time),
            self.distribution)
        applied, second_audit = solve(safe+np.asarray(residual))
        return dict(observation=np.asarray(flattened)[0], raw_flow=raw,
                    nominal=nominal, safe=safe, gate=bool(np.asarray(gate)),
                    residual=np.asarray(residual), applied=applied,
                    mu=np.asarray(values["mu"]), alpha=float(values["alpha"]),
                    beta=float(values["beta"]), p=float(values["p"]),
                    sigma=float(values["sigma"]), epsilon=np.asarray(values["epsilon"]),
                    first_projection=first_audit,
                    second_projection=second_audit)


def rollout_episode(controller, initial_positions, tape, split, scenario,
                    continuation, metric):
    """Complete operational episode with exact monitor and no post-terminal pad."""
    env = GiveWayEnv(controller.plant)
    env.reset(initial_positions)
    records = []
    terminal_label = None
    while not env.done:
        time = env.step_count
        pre_latch = env.first_deadlock_step is not None
        row = controller.act(env, tape, split, scenario, continuation, time)
        _, _, done, info = env.step(row["applied"].reshape(2, 2))
        row.update(pre_action_latch=pre_latch,
                   positions=np.asarray(info["positions"]),
                   goal_errors=np.asarray(info["goal_errors"]),
                   speeds=np.asarray(info["speeds"]),
                   raw_deadlock=bool(info["deadlock"]),
                   candidate=bool(info["candidate_deadlock"]),
                   termination=info["termination"])
        records.append(row)
        if done:
            terminal_label = _terminal_label(info, controller.plant)
    if terminal_label is None:
        raise DirectionARolloutError("episode exited without a terminal label")
    max_speeds = np.asarray([r["speeds"].max() for r in records])
    goal_errors = np.asarray([r["goal_errors"] for r in records])
    reclassified, stalled_detail = classify_timeout_trace(
        dict(max_speed=max_speeds, goal_errors=goal_errors), terminal_label,
        controller.plant.dt)
    eligible_stalled = reclassified == "stalled_deadlock"
    historical_strict = env.first_deadlock_step is not None
    event = bool(historical_strict or eligible_stalled)
    safe = np.asarray([r["safe"] for r in records])
    observations = np.asarray([r["observation"] for r in records])
    gates = np.asarray([r["gate"] for r in records])
    residuals = np.asarray([r["residual"] for r in records])
    applied = np.asarray([r["applied"] for r in records])
    latches = np.asarray([r["pre_action_latch"] for r in records])
    zeros = np.zeros((len(records), 1))
    scores = trajectory_scores(
        controller.model, controller.params, safe, zeros, observations,
        gates, residuals, latches, controller.distribution)
    deformation = deformation_estimate(
        controller.model, controller.params, safe, zeros, observations,
        gates, residuals, applied, metric, controller.distribution,
        dt=controller.plant.dt)
    return dict(D_H=event, historical_strict=historical_strict,
                eligible_stalled=eligible_stalled,
                task_success=env.first_success_step is not None,
                ordinary_timeout=terminal_label == "other_timeout" and not eligible_stalled,
                terminal_label=terminal_label, stalled_details=stalled_detail,
                first_deadlock_step=env.first_deadlock_step,
                episode_steps=len(records), records=records, scores=scores,
                deformation=deformation)
