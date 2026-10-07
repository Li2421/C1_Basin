"""Interpretable empirical Bellman continuation certificate for CL-FHCB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from single_integrator.environment import Config, GiveWayEnv

from .closed_loop import ClosedLoopTrace, DiagnosticPhi, state_metrics


BIN_NAMES = (
    "candidate_late",
    "candidate_early",
    "progress_starved",
    "resolved_progress",
    "bay_progress",
    "advancing",
    "unresolved",
)
BIN_INDEX = {name: index for index, name in enumerate(BIN_NAMES)}
GUARDED_BINS = (
    BIN_INDEX["candidate_late"],
    BIN_INDEX["candidate_early"],
    BIN_INDEX["progress_starved"],
)
TERMINAL_COLUMNS = ("deadlock", "success", "collision")


def abstract_bin(metrics: dict, config: Config) -> int:
    """Map an augmented state to one documented, interpretable risk cell."""

    if metrics["candidate_active"]:
        if metrics["candidate_age"] >= 0.5 * config.deadlock_hold_seconds:
            return BIN_INDEX["candidate_late"]
        return BIN_INDEX["candidate_early"]

    errors = np.asarray(metrics["goal_errors"], dtype=np.float64)
    positions = np.asarray(metrics["positions"], dtype=np.float64)
    progress = np.asarray(metrics["recent_progress"], dtype=np.float64)
    ready = bool(metrics["window_ready"])

    # This guard deliberately ignores the strict timer.  A timer reset or
    # small jitter therefore cannot disguise a low-progress unresolved state.
    if (
        ready
        and float(np.max(np.abs(progress))) < 2.0 * config.progress_epsilon
        and float(np.max(errors)) > config.goal_tolerance
    ):
        return BIN_INDEX["progress_starved"]

    signed_order = float(positions[0, 0] - positions[1, 0])
    progress_sum = float(np.nansum(progress)) if ready else 0.0
    if signed_order >= 0.10 and (
        progress_sum > config.progress_epsilon
        or float(np.max(errors)) <= 3.0 * config.goal_tolerance
    ):
        return BIN_INDEX["resolved_progress"]

    bay_center_threshold = (
        config.corridor_width / 2.0 + config.agent_radius
    )
    if (
        ready
        and float(np.max(positions[:, 1])) >= bay_center_threshold
        and progress_sum > 0.5 * config.progress_epsilon
    ):
        return BIN_INDEX["bay_progress"]

    if ready and progress_sum > config.progress_epsilon:
        return BIN_INDEX["advancing"]
    return BIN_INDEX["unresolved"]


@dataclass(frozen=True)
class EmpiricalTransitionModel:
    phi: DiagnosticPhi
    # Rows are current bins.  Columns are D, S, C, then next nonterminal bins.
    counts: np.ndarray
    probabilities: np.ndarray


@dataclass(frozen=True)
class EmpiricalContinuationCertificate:
    """Piecewise-constant-in-state, exact-finite-horizon empirical certificate."""

    phi: DiagnosticPhi
    horizon: int
    transition_counts: np.ndarray
    transition_probabilities: np.ndarray
    table: np.ndarray

    def value(self, metrics: dict, remaining_steps: int, config: Config) -> float:
        terminal = str(metrics.get("terminal", "running"))
        if terminal == "deadlock":
            return 1.0
        if terminal in ("success", "collision", "timeout"):
            return 0.0
        n = int(remaining_steps)
        if n <= 0:
            return 0.0
        if n > self.horizon:
            raise ValueError("remaining horizon exceeds frozen certificate")
        return float(self.table[n, abstract_bin(metrics, config)])

    def bellman_slacks(self) -> np.ndarray:
        """Return B_s(n)-T_hat B_s(n-1) for every empirical row and n."""

        p = self.transition_probabilities
        p_deadlock = p[:, 0]
        p_running = p[:, 3:]
        slacks = []
        for n in range(1, self.horizon + 1):
            rhs = p_deadlock + p_running @ self.table[n - 1]
            slacks.append(self.table[n] - rhs)
        return np.asarray(slacks)


@dataclass(frozen=True)
class PrefixRisk:
    value: float
    prefix_steps: int
    prefix_event: str
    continuation_bin: str | None
    remaining_steps: int


def transition_counts(
    traces: Iterable[ClosedLoopTrace], config: Config
) -> np.ndarray:
    goals = GiveWayEnv(config).goals
    size = len(BIN_NAMES)
    result = np.zeros((size, 3 + size), dtype=np.int64)
    for trace in traces:
        for record in trace.steps:
            current = abstract_bin(state_metrics(record.state, config, goals), config)
            if record.event == "deadlock":
                result[current, 0] += 1
            elif record.event == "success":
                result[current, 1] += 1
            elif record.event == "collision":
                result[current, 2] += 1
            else:
                # A deadline transition is ordinary no-event physical dynamics;
                # b(.,0)=0 supplies the timeout boundary in the recursion.
                following = abstract_bin(
                    state_metrics(record.next_state, config, goals), config
                )
                result[current, 3 + following] += 1
    return result


def _empirical_probabilities(counts: np.ndarray) -> np.ndarray:
    probabilities = np.zeros_like(counts, dtype=np.float64)
    totals = counts.sum(axis=1)
    for row, total in enumerate(totals):
        if total:
            probabilities[row] = counts[row] / total
        else:
            # No extrapolation: an unobserved cell is maximally risky.
            probabilities[row, 0] = 1.0
    return probabilities


def build_empirical_certificate(
    traces: Iterable[ClosedLoopTrace],
    config: Config,
    phi: DiagnosticPhi,
) -> EmpiricalContinuationCertificate:
    selected = [trace for trace in traces if trace.phi.vector == phi.vector]
    if not selected:
        raise ValueError(f"no construction traces for phi={phi.name}")
    counts = transition_counts(selected, config)
    probabilities = _empirical_probabilities(counts)
    table = np.zeros((config.max_steps + 1, len(BIN_NAMES)), dtype=np.float64)
    p_deadlock = probabilities[:, 0]
    p_running = probabilities[:, 3:]
    for n in range(1, config.max_steps + 1):
        table[n] = p_deadlock + p_running @ table[n - 1]
        table[n, list(GUARDED_BINS)] = 1.0
        table[n] = np.clip(table[n], 0.0, 1.0)
    certificate = EmpiricalContinuationCertificate(
        phi=phi,
        horizon=config.max_steps,
        transition_counts=counts,
        transition_probabilities=probabilities,
        table=table,
    )
    if float(np.min(certificate.bellman_slacks())) < -1e-12:
        raise AssertionError("constructed table is not an empirical supersolution")
    return certificate


def score_prefix(
    trace: ClosedLoopTrace,
    start: int,
    K: int,
    certificate: EmpiricalContinuationCertificate,
    config: Config,
) -> PrefixRisk:
    """Pathwise R_K: prefix event indicator or surviving continuation value."""

    if K <= 0 or start < 0 or start >= len(trace.steps):
        raise ValueError("invalid positive K or prefix start")
    if trace.phi.vector != certificate.phi.vector:
        raise ValueError("trace and continuation policy phi do not match")
    available = config.max_steps - trace.steps[start].state.step
    observed = min(int(K), available)
    stop = min(start + observed, len(trace.steps))
    for record in trace.steps[start:stop]:
        if record.event == "deadlock":
            return PrefixRisk(1.0, record.next_state.step - trace.steps[start].state.step,
                              "deadlock", None, max(0, available - observed))
        if record.event in ("success", "collision", "timeout"):
            return PrefixRisk(0.0, record.next_state.step - trace.steps[start].state.step,
                              record.event, None, max(0, available - observed))

    if stop < start + observed:
        raise AssertionError("trace ended without a terminal event")
    boundary = trace.steps[stop - 1].next_state
    remaining = max(0, available - observed)
    metrics = state_metrics(boundary, config, GiveWayEnv(config).goals)
    cell = abstract_bin(metrics, config)
    return PrefixRisk(
        certificate.value(metrics, remaining, config),
        observed,
        "survived_prefix",
        BIN_NAMES[cell],
        remaining,
    )
