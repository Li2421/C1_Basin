"""Verify H=1 pilot parity and one-step-only H=4 cadence semantics.

Run after producing one episode in the ``smoke_h1_parity`` and
``smoke_h4_semantics`` namespaces.  This script does not launch rollouts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_fixed_cadence_ablation_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"


def _record(root: Path, controller: str) -> dict[str, Any]:
    return json.loads((root / "raw" / controller / "episode_0000.json").read_text())


def _trajectory(root: Path, row: dict[str, Any], base: Path) -> dict[str, np.ndarray]:
    path = base / row["trajectory_file"]
    with np.load(path, allow_pickle=False) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def main() -> None:
    pilot_root = PILOT / "runs/production"
    h1_root = HERE / "runs/smoke_h1_parity"
    h4_root = HERE / "runs/smoke_h4_semantics"
    pilot = _record(pilot_root, "learned")
    h1 = _record(h1_root, "h1")
    h4 = _record(h4_root, "h4")
    pilot_arrays = _trajectory(pilot_root, pilot, PILOT)
    h1_arrays = _trajectory(h1_root, h1, HERE)
    h4_arrays = _trajectory(h4_root, h4, HERE)

    parity_row_fields = (
        "episode_index", "ic_seed", "flow_seed", "initial_positions", "outcome",
        "failure_type", "episode_steps", "success", "deadlock", "timeout",
        "collision", "wall_collision", "agent_collision", "J_def",
        "J_def_startup_0_40", "J_def_post_startup", "corrected_timesteps",
        "projection_failures", "invalid_actions",
    )
    parity_rows = {}
    for name in parity_row_fields:
        if isinstance(pilot[name], float):
            parity_rows[name] = bool(np.isclose(
                pilot[name], h1[name], rtol=1e-7, atol=2e-9
            ))
        else:
            parity_rows[name] = pilot[name] == h1[name]
    common_arrays = sorted(set(pilot_arrays) & set(h1_arrays))
    parity_arrays = {}
    parity_max_abs = {}
    for name in common_arrays:
        left, right = pilot_arrays[name], h1_arrays[name]
        if left.dtype.kind in "fc" and right.dtype.kind in "fc":
            parity_arrays[name] = bool(np.allclose(
                left, right, rtol=2e-6, atol=2e-7, equal_nan=True
            ))
            parity_max_abs[name] = float(np.max(np.abs(left - right))) if left.size else 0.0
        else:
            parity_arrays[name] = bool(np.array_equal(left, right))

    steps = np.asarray(h4_arrays["step"], dtype=np.int64)
    scheduled = np.asarray(h4_arrays["cadence_scheduled"], dtype=bool)
    expected_schedule = steps % 4 == 0
    inactive = ~scheduled
    semantics = {
        "schedule_exact": bool(np.array_equal(scheduled, expected_schedule)),
        "record_query_count_matches_schedule": (
            h4["model_query_count"] == int(scheduled.sum())
        ),
        "record_scheduled_count_matches_schedule": (
            h4["scheduled_correction_timesteps"] == int(scheduled.sum())
        ),
        "inactive_g_hat_exact_zero": bool(np.all(h4_arrays["g_hat"][inactive] == 0.0)),
        "inactive_raw_target_equals_u_safe": bool(np.array_equal(
            h4_arrays["raw_second_target"][inactive], h4_arrays["u_safe"][inactive]
        )),
        "inactive_u_exec_equals_u_safe": bool(np.array_equal(
            h4_arrays["u_exec"][inactive], h4_arrays["u_safe"][inactive]
        )),
        "inactive_raw_norm_exact_zero": bool(np.all(
            h4_arrays["raw_correction_norm"][inactive] == 0.0
        )),
        "inactive_executed_norm_exact_zero": bool(np.all(
            h4_arrays["executed_correction_norm"][inactive] == 0.0
        )),
        "inactive_rewrite_norm_exact_zero": bool(np.all(
            h4_arrays["projection_rewrite_norm"][inactive] == 0.0
        )),
        "feature_history_updates_every_step": bool(np.array_equal(
            h4_arrays["real_history_length"], np.minimum(steps + 1, 41)
        )),
    }
    checks = {
        "h1_row_parity": parity_rows,
        "h1_trajectory_tight_parity": parity_arrays,
        "h1_trajectory_max_absolute_difference": parity_max_abs,
        "h4_one_step_cadence_semantics": semantics,
    }
    passed = (
        all(parity_rows.values())
        and all(parity_arrays.values())
        and all(semantics.values())
    )
    payload = {
        "status": "PASS" if passed else "FAIL",
        "note": "H=1 CPU smoke compared against prior GPU production episode 0000 (float arrays rtol=2e-6, atol=2e-7; discrete arrays exact); H=4 checked for exact zero/no hold between scheduled steps.",
        "checks": checks,
    }
    output = HERE / "runs/smoke_integrity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
