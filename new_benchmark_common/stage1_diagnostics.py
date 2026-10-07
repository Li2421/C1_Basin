"""Small, reusable persistence layer for frozen Stage-I diagnostics.

It is deliberately policy- and geometry-neutral: scenarios supply the rollout
adapter, independently drawn cases and expert-reference windows.  The module
does not collect data or alter a model, so it is safe to use for a final test.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .evaluation import EvaluationCase, KStepWindow, evaluate_closed_loop, k_step_stability, teacher_forced_rmse


def _row_public(row: dict) -> dict:
    """Keep provenance/outcomes, omitting large non-JSON rollout internals."""
    return {key: value for key, value in row.items() if key not in {"observations", "snapshots", "infos"}}


def run_and_save(
    output: str | Path, *, adapter, agent, cases: Iterable[EvaluationCase],
    train_observations: np.ndarray, evaluation_observations: np.ndarray,
    evaluation_actions: np.ndarray, windows: Iterable[KStepWindow], seed: int,
    max_steps: int, action_postprocess=None,
) -> dict:
    """Persist all requested no-safety metrics and failure points atomically enough for reports."""
    result = evaluate_closed_loop(
        adapter, agent, cases, train_observations=train_observations,
        dev_observations=evaluation_observations, seed=seed, max_steps=max_steps,
        action_postprocess=action_postprocess,
    )
    k_summary, k_rows = k_step_stability(
        adapter, agent, windows, seed=seed + 71, action_postprocess=action_postprocess,
    )
    teacher = teacher_forced_rmse(agent, evaluation_observations, evaluation_actions, seed=seed + 29)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    public_rows = [_row_public(row) for row in result["rollouts"]]
    failure_points = []
    for row in result["rollouts"]:
        if row["success"] or not row["infos"]:
            continue
        last = row["infos"][-1]
        if "positions" in last:
            failure_points.append(np.asarray(last["positions"], dtype=np.float64))
    np.save(output / "failure_points.npy", np.asarray(failure_points, dtype=np.float64))
    payload = {
        "aggregate": result["aggregate"], "teacher_forced": teacher,
        "k_step": k_summary, "rollouts": public_rows, "k_step_rows": k_rows,
        "scientific_scope": "raw no-safety Stage-I MACFlow; no eta, basin, OrthoFlow3, or G_phi",
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload
