"""Verify WIDE Flow-key parity, Safety semantics, and H=4 no-hold cadence.

Expected prerequisite commands (one episode each)::

  run_evaluation.py --controller safety --namespace smoke_safety --limit 1
  run_evaluation.py --controller h4 --namespace smoke_h4 --limit 1

This verifier launches no rollout itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HISTORICAL = (
    SYSROOT
    / "results/risk_audit_seed0_baselines/mac_cbf/rollout_0000.npz"
)


def _record(namespace: str, controller: str) -> dict[str, Any]:
    path = (
        HERE / "runs" / namespace / "raw" / controller / "episode_0000.json"
    )
    return json.loads(path.read_text())


def _trajectory(row: dict[str, Any]) -> dict[str, np.ndarray]:
    path = HERE / row["trajectory_file"]
    with np.load(path, allow_pickle=False) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def main() -> None:
    import jax
    jax.config.update("jax_enable_x64", True)

    safety = _record("smoke_safety", "safety")
    h4 = _record("smoke_h4", "h4")
    safety_arrays = _trajectory(safety)
    h4_arrays = _trajectory(h4)
    with np.load(HISTORICAL, allow_pickle=False) as values:
        historical_initial = np.asarray(values["initial_positions"])
        historical_u_nom_0 = np.asarray(values["u_nom"])[0]

    def expected_keys(steps: np.ndarray, rollout_id: int) -> np.ndarray:
        root = jax.random.PRNGKey(np.uint32(42))
        episode = jax.random.fold_in(root, rollout_id)
        return np.stack([
            np.asarray(jax.random.fold_in(episode, int(step)), dtype=np.uint32)
            for step in steps
        ])

    safety_steps = np.asarray(safety_arrays["step"], dtype=np.int64)
    h4_steps = np.asarray(h4_arrays["step"], dtype=np.int64)
    safety_scheduled = np.asarray(
        safety_arrays["cadence_scheduled"], dtype=bool
    )
    h4_scheduled = np.asarray(h4_arrays["cadence_scheduled"], dtype=bool)
    inactive = ~h4_scheduled
    checks = {
        "same_frozen_initial": safety["initial_positions"]
        == h4["initial_positions"],
        "historical_initial_exact": np.array_equal(
            np.asarray(safety["initial_positions"], dtype=np.float32),
            historical_initial,
        ),
        "safety_nested_fold_in_keys_exact": np.array_equal(
            safety_arrays["flow_step_key"],
            expected_keys(safety_steps, safety["rollout_id"]),
        ),
        "h4_nested_fold_in_keys_exact": np.array_equal(
            h4_arrays["flow_step_key"],
            expected_keys(h4_steps, h4["rollout_id"]),
        ),
        "safety_never_scheduled": not bool(safety_scheduled.any()),
        "safety_query_count_zero": safety["model_query_count"] == 0,
        "safety_u_exec_equals_u_safe": np.array_equal(
            safety_arrays["u_exec"], safety_arrays["u_safe"]
        ),
        "safety_J_def_exact_zero": safety["J_def"] == 0.0,
        "h4_schedule_exact": np.array_equal(
            h4_scheduled, h4_steps % 4 == 0
        ),
        "h4_query_count_matches_schedule": h4["model_query_count"]
        == int(h4_scheduled.sum()),
        "h4_inactive_g_hat_exact_zero": bool(
            np.all(h4_arrays["g_hat"][inactive] == 0.0)
        ),
        "h4_inactive_raw_target_equals_u_safe": np.array_equal(
            h4_arrays["raw_second_target"][inactive],
            h4_arrays["u_safe"][inactive],
        ),
        "h4_inactive_u_exec_equals_u_safe": np.array_equal(
            h4_arrays["u_exec"][inactive], h4_arrays["u_safe"][inactive]
        ),
        "h4_inactive_norms_exact_zero": bool(
            np.all(h4_arrays["raw_correction_norm"][inactive] == 0.0)
            and np.all(
                h4_arrays["executed_correction_norm"][inactive] == 0.0
            )
            and np.all(
                h4_arrays["projection_rewrite_norm"][inactive] == 0.0
            )
        ),
        "safety_feature_history_every_step": np.array_equal(
            safety_arrays["real_history_length"],
            np.minimum(safety_steps + 1, 41),
        ),
        "h4_feature_history_every_step": np.array_equal(
            h4_arrays["real_history_length"], np.minimum(h4_steps + 1, 41)
        ),
        "matched_first_step_flow": np.array_equal(
            safety_arrays["u_flow"][0], h4_arrays["u_flow"][0]
        ),
        "matched_first_step_safety_projection": np.array_equal(
            safety_arrays["u_safe"][0], h4_arrays["u_safe"][0]
        ),
        "matched_first_step_feature": np.array_equal(
            safety_arrays["feature"][0], h4_arrays["feature"][0]
        ),
    }
    passed = all(checks.values())
    payload = {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "non_gating_diagnostics": {
            "historical_step0_flow_nominal_tight": bool(np.allclose(
                safety_arrays["u_flow"][0],
                historical_u_nom_0,
                rtol=2e-6,
                atol=2e-7,
            )),
            "historical_step0_flow_nominal": historical_u_nom_0.tolist(),
            "current_step0_flow_nominal": safety_arrays["u_flow"][0].tolist(),
        },
        "note": (
            "Historical terminal/action parity is non-gating because the "
            "historical trace used older FlowBC/projection source hashes. "
            "The frozen initial is exact; nested JAX keys are independently "
            "recomputed and exact; Safety/H4 agree before intervention. "
            "Current controllers use the narrow-cadence authoritative "
            "FlowBC implementation and required 841a2d... CBF plus "
            "e29d51... retry projector."
        ),
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
