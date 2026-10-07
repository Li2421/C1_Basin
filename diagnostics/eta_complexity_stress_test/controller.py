"""Checkpoint-free analytic nominal, shared-eta controller, and rollout logs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np

from shared_control.diagnostic_corrector import DiagnosticCorrector, DiagnosticEta, bounded_rows
from shared_control.hard_projection import CBFSolverError, HardSafetyFilter

from .environment import CoupledDualIntersectionEnv, StressConfig


Variant = Literal["nominal", "safe", "eta"]
GROUPS = ("L2R", "R2L", "J1_down", "J1_up", "J2_down", "J2_up")


def route_waypoints(goals: np.ndarray) -> tuple[np.ndarray, ...]:
    """Geometry-only routes; no yields, reservation, phase, or priority state."""
    return (
        np.asarray(((-2.90, -0.10), (-2.70, -0.06), (-1.50, -0.06), (1.50, -0.06), (2.70, -0.06), (2.90, -0.06), (3.00, -0.75), goals[0])),
        np.asarray(((-2.90, 0.10), (-2.70, 0.06), (-1.50, 0.06), (1.50, 0.06), (2.70, 0.06), (2.90, 0.06), (3.00, 0.75), goals[1])),
        np.asarray(((2.90, 0.10), (2.70, 0.06), (1.50, 0.06), (-1.50, 0.06), (-2.70, 0.06), (-2.90, 0.06), (-3.00, 0.75), goals[2])),
        np.asarray(((2.90, -0.10), (2.70, -0.06), (1.50, -0.06), (-1.50, -0.06), (-2.70, -0.06), (-2.90, -0.06), (-3.00, -0.75), goals[3])),
        np.asarray(((-1.68, 0.0), goals[4])),
        np.asarray(((-1.32, 0.0), goals[5])),
        np.asarray(((1.68, 0.0), goals[6])),
        np.asarray(((1.32, 0.0), goals[7])),
    )


class WaypointNominalController:
    """Every agent independently seeks its next fixed geometric waypoint."""

    def __init__(self, goals: np.ndarray, max_speed: float, waypoint_tolerance: float = 0.04):
        self.routes = route_waypoints(np.asarray(goals, dtype=np.float64))
        self.max_speed = float(max_speed)
        self.waypoint_tolerance = float(waypoint_tolerance)
        self.indices = np.zeros(8, dtype=np.int64)

    def target(self, positions: np.ndarray, agent: int) -> np.ndarray:
        route = self.routes[agent]
        while self.indices[agent] < len(route) - 1 and np.linalg.norm(route[self.indices[agent]] - positions[agent]) <= self.waypoint_tolerance:
            self.indices[agent] += 1
        return route[self.indices[agent]].copy()

    def complete(self, positions: np.ndarray, agent: int) -> bool:
        return bool(self.indices[agent] == len(self.routes[agent]) - 1 and np.linalg.norm(self.routes[agent][-1] - positions[agent]) <= self.waypoint_tolerance)

    def action(self, positions: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float64)
        enabled = np.ones(8, dtype=bool) if active is None else np.asarray(active, dtype=bool)
        if positions.shape != (8, 2) or enabled.shape != (8,):
            raise ValueError("expected positions [8,2] and active mask [8]")
        target = np.stack([self.target(positions, agent) for agent in range(8)])
        displacement = target - positions
        direction = displacement / np.maximum(np.linalg.norm(displacement, axis=-1, keepdims=True), 1e-12)
        return self.max_speed * direction * enabled[:, None]


def _entry_orders(junction: np.ndarray, bridge: np.ndarray) -> tuple[str, str, str]:
    """Return coarse first-entry sequences for J1, bridge, and J2."""
    horizontal = (((0, 1), "L2R"), ((2, 3), "R2L"))
    vertical = (((4,), "J1_down"), ((5,), "J1_up"), ((6,), "J2_down"), ((7,), "J2_up"))

    def order_at(index: int) -> str:
        entries: list[tuple[int, str]] = []
        for agents, label in horizontal:
            hits = np.flatnonzero(junction[:, list(agents), index].any(axis=-1))
            if len(hits):
                entries.append((int(hits[0]), label))
        for agents, label in vertical:
            if (index == 0 and label.startswith("J1")) or (index == 1 and label.startswith("J2")):
                hits = np.flatnonzero(junction[:, list(agents), index].any(axis=-1))
                if len(hits):
                    entries.append((int(hits[0]), label))
        return ">".join(label for _, label in sorted(entries)) or "none"

    bridge_entries = []
    for agents, label in horizontal:
        hits = np.flatnonzero(bridge[:, list(agents)].any(axis=-1))
        if len(hits):
            bridge_entries.append((int(hits[0]), label))
    return order_at(0), ">".join(label for _, label in sorted(bridge_entries)) or "none", order_at(1)


def coordination_signature(junction: np.ndarray, bridge: np.ndarray) -> str:
    first, middle, second = _entry_orders(junction, bridge)
    return f"J1:{first}|bridge:{middle}|J2:{second}"


def failure_mode(termination: str, infos: list[dict], bridge: np.ndarray, junction: np.ndarray) -> str:
    if termination == "success":
        return "success"
    if termination == "projection_failure":
        return "projection_failure"
    if termination == "collision":
        final = infos[-1]
        if final["agent_collision"] and final["wall_collision"]:
            return "agent_and_wall_collision"
        return "agent_collision" if final["agent_collision"] else "wall_collision"
    if termination == "deadlock":
        return "bridge_blockage" if bridge[-40:].any() else "strict_deadlock"
    if termination == "timeout":
        if bridge[-80:].any():
            return "bridge_blockage"
        active_groups = sum(bool(junction[-80:, :, j].any()) for j in range(2))
        return "junction_cyclic_waiting" if active_groups == 2 else "stalled_congestion"
    return termination


@dataclass(frozen=True)
class RolloutTrace:
    variant: str
    eta: tuple[float, float, float]
    termination: str
    failure_mode: str
    positions: np.ndarray
    u_nom: np.ndarray
    u_safe: np.ndarray
    b_goal: np.ndarray
    b_rel: np.ndarray
    correction: np.ndarray
    u_exec: np.ndarray
    junction_occupancy: np.ndarray
    bridge_occupancy: np.ndarray
    infos: tuple[dict, ...]

    @property
    def success(self) -> bool:
        return self.termination == "success"

    @property
    def episode_steps(self) -> int:
        return len(self.u_exec)

    @property
    def summary(self) -> dict:
        final = self.infos[-1] if self.infos else {}
        return {
            "variant": self.variant, "eta": list(self.eta), "termination": self.termination,
            "failure_mode": self.failure_mode, "success": self.success,
            "episode_steps": self.episode_steps,
            "final_max_goal_error": float(np.max(final.get("goal_errors", np.full(8, np.nan)))),
            "final_sum_goal_error": float(np.sum(final.get("goal_errors", np.full(8, np.nan)))),
            "coordination_signature": coordination_signature(self.junction_occupancy, self.bridge_occupancy),
            "max_raw_correction_norm": float(np.linalg.norm(self.correction, axis=-1).max(initial=0.0)),
            "max_executed_speed": float(np.linalg.norm(self.u_exec, axis=-1).max(initial=0.0)),
        }

    def save(self, directory: str | Path, stem: str) -> tuple[Path, Path]:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        arrays = destination / f"{stem}.npz"
        metadata = destination / f"{stem}.json"
        np.savez_compressed(
            arrays, positions=self.positions, u_nom=self.u_nom, u_safe=self.u_safe,
            b_goal=self.b_goal, b_rel=self.b_rel, correction=self.correction,
            u_exec=self.u_exec, junction_occupancy=self.junction_occupancy,
            bridge_occupancy=self.bridge_occupancy,
        )
        metadata.write_text(json.dumps(self.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return arrays, metadata


def rollout(
    initial_positions: np.ndarray,
    eta: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    variant: Variant = "eta",
    config: StressConfig | None = None,
    projection: HardSafetyFilter | None = None,
) -> RolloutTrace:
    """Run a deterministic analytic controller; no learned nominal is present."""
    if variant not in ("nominal", "safe", "eta"):
        raise ValueError("variant must be nominal, safe, or eta")
    eta_value = tuple(float(x) for x in np.asarray(eta, dtype=np.float64))
    if len(eta_value) != 3 or not np.isfinite(np.asarray(eta_value)).all():
        raise ValueError("eta must contain three finite values")
    env = CoupledDualIntersectionEnv(config)
    env.reset(initial_positions)
    nominal = WaypointNominalController(env.goals, env.config.max_speed)
    corrector = DiagnosticCorrector(DiagnosticEta(*eta_value))
    safety = projection or HardSafetyFilter()
    positions, raw_actions, safe_actions, goal_bases, relation_bases, corrections, executed = [env.positions.copy()], [], [], [], [], [], []
    junction, bridge, infos = [], [], []
    termination = "running"
    for _ in range(env.config.max_steps):
        u_nom = nominal.action(env.positions)
        try:
            if variant == "nominal":
                u_safe = u_nom.copy()
                b_goal = bounded_rows(env.goals - env.positions, env.config.max_speed)
                b_rel = np.zeros_like(u_nom)
                correction = np.zeros_like(u_nom)
                u_exec = u_nom.copy()
            else:
                first = safety(env.snapshot(), u_nom)
                u_safe = np.asarray(first.velocity, dtype=np.float64)
                if variant == "eta":
                    correction, b_goal, b_rel = corrector(env.positions, env.goals, u_safe, env.config.max_speed)
                    u_exec = np.asarray(safety(env.snapshot(), u_safe + correction).velocity, dtype=np.float64)
                else:
                    b_goal = bounded_rows(env.goals - env.positions, env.config.max_speed)
                    b_rel = np.zeros_like(u_nom)
                    correction = np.zeros_like(u_nom)
                    u_exec = u_safe.copy()
        except CBFSolverError as error:
            termination = "projection_failure"
            infos.append({"termination": termination, "solver_status": error.status, "goal_errors": np.linalg.norm(env.goals - env.positions, axis=-1)})
            break
        _, _, done, info = env.step(u_exec)
        raw_actions.append(u_nom); safe_actions.append(u_safe); goal_bases.append(b_goal); relation_bases.append(b_rel); corrections.append(correction); executed.append(u_exec)
        junction.append(info["junction_occupancy"]); bridge.append(info["bridge_occupancy"]); infos.append(info); positions.append(env.positions.copy())
        termination = str(info["termination"])
        if done:
            break
    if termination == "running":
        raise AssertionError("rollout did not reach a terminal event")
    arrays = lambda rows, shape, dtype=float: np.asarray(rows, dtype=dtype) if rows else np.empty((0, *shape), dtype=dtype)
    junction_array = arrays(junction, (2, 8), bool)
    bridge_array = arrays(bridge, (8,), bool)
    return RolloutTrace(
        variant=variant, eta=eta_value, termination=termination,
        failure_mode=failure_mode(termination, infos, bridge_array, junction_array),
        positions=np.asarray(positions, dtype=np.float64),
        u_nom=arrays(raw_actions, (8, 2)), u_safe=arrays(safe_actions, (8, 2)),
        b_goal=arrays(goal_bases, (8, 2)), b_rel=arrays(relation_bases, (8, 2)),
        correction=arrays(corrections, (8, 2)), u_exec=arrays(executed, (8, 2)),
        junction_occupancy=junction_array, bridge_occupancy=bridge_array, infos=tuple(infos),
    )
