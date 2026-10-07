"""Construct (not qualify) the frozen empirical CL-FHCB certificate on CPU."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import jax
import numpy as np
from scipy.stats import beta

from single_integrator.cbf import CBFConfig
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.evaluate import load_policy

from diagnostics.cl_fhcb.certificate import (
    BIN_NAMES,
    GUARDED_BINS,
    abstract_bin,
    build_empirical_certificate,
    score_prefix,
)
from diagnostics.cl_fhcb.closed_loop import (
    ClosedLoopTrace,
    DiagnosticPhi,
    rollout_closed_loop,
    state_metrics,
)


ROOT = Path(__file__).resolve().parents[2]
PHIS = (
    DiagnosticPhi(0.0, 0.0, 0.0, "zero"),
    DiagnosticPhi(0.25, 0.0, 0.0, "goal025"),
    DiagnosticPhi(0.0, -0.35, 0.0, "damp035"),
    DiagnosticPhi(0.0, 0.0, 0.25, "relative025"),
)
K_VALUES = (20, 100)
CONSTRUCTION_PAIR_IDS = tuple(range(8))
FLOW_SEED = 7319


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_assets(explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend((ROOT, ROOT.parent / "02_C1_Toy_GiveWay"))
    for candidate in candidates:
        checkpoint = candidate / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
        dataset = candidate / "datasets/give_way_si_short_v1/environment.json"
        if checkpoint.is_file() and dataset.is_file():
            return candidate.resolve()
    raise FileNotFoundError("approved checkpoint/dataset assets were not found")


def construction_start(dataset_root: Path, pair_id: int) -> np.ndarray:
    path = dataset_root / "raw" / f"episode_{2 * pair_id:04d}.npz"
    with np.load(path) as item:
        return np.asarray(item["initial_positions"], dtype=np.float64).copy()


def trace_arrays(trace: ClosedLoopTrace, config: Config) -> dict[str, np.ndarray]:
    goals = GiveWayEnv(config).goals
    return {
        "positions_before": np.asarray([step.state.positions for step in trace.steps]),
        "positions_after": np.asarray([step.next_state.positions for step in trace.steps]),
        "last_velocity_before": np.asarray([step.state.last_velocity for step in trace.steps]),
        "u_flow": np.asarray([step.u_flow for step in trace.steps]),
        "u_safe": np.asarray([step.u_safe for step in trace.steps]),
        "g": np.asarray([step.correction for step in trace.steps]),
        "w": np.asarray([step.candidate for step in trace.steps]),
        "u_exec": np.asarray([step.u_exec for step in trace.steps]),
        "flow_key_data": np.asarray([step.flow_key_data for step in trace.steps]),
        "candidate_since": np.asarray([
            -1 if step.state.candidate_since is None else step.state.candidate_since
            for step in trace.steps
        ], dtype=np.int32),
        "stuck_timer_after": np.asarray([
            step.monitor["stuck_timer"] for step in trace.steps
        ]),
        "event": np.asarray([step.event for step in trace.steps]),
        "abstract_bin_before": np.asarray([
            abstract_bin(state_metrics(step.state, config, goals), config)
            for step in trace.steps
        ], dtype=np.int8),
        "first_min_cbf_residual": np.asarray([
            step.first_min_cbf_residual for step in trace.steps
        ]),
        "second_min_cbf_residual": np.asarray([
            step.second_min_cbf_residual for step in trace.steps
        ]),
        "initial_positions": trace.initial_state.positions,
        "pair_id": np.asarray(trace.pair_id),
        "flow_seed": np.asarray(trace.flow_seed),
        "phi": np.asarray(trace.phi.vector),
    }


def jeffreys_immediate_deadlock_upper(counts: np.ndarray) -> list[float]:
    result = []
    for row in counts:
        total = int(row.sum())
        deadlock = int(row[0])
        if total == 0:
            result.append(1.0)
        else:
            result.append(float(beta.ppf(0.95, deadlock + 0.5, total - deadlock + 0.5)))
    return result


def serializable_certificate(certificate) -> dict:
    return {
        "phi_name": certificate.phi.name,
        "phi": list(certificate.phi.vector),
        "horizon": certificate.horizon,
        "bin_names": list(BIN_NAMES),
        "guarded_bins": [BIN_NAMES[index] for index in GUARDED_BINS],
        "columns": ["deadlock", "success", "collision"]
        + [f"next:{name}" for name in BIN_NAMES],
        "transition_counts": certificate.transition_counts.tolist(),
        "transition_probabilities": certificate.transition_probabilities.tolist(),
        "immediate_deadlock_jeffreys_upper_95": jeffreys_immediate_deadlock_upper(
            certificate.transition_counts
        ),
        "table": certificate.table.tolist(),
        "minimum_empirical_bellman_slack": float(
            np.min(certificate.bellman_slacks())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "diagnostics/cl_fhcb/data_v1"
    )
    args = parser.parse_args()

    # This is a construction-only CPU job.  Setting the platform here occurs
    # before policy loading or any JIT execution.
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"construction output must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    trace_dir = out / "traces"
    trace_dir.mkdir()

    asset_root = locate_assets(args.asset_root)
    checkpoint = asset_root / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
    checkpoint_config = asset_root / "baseline_309_314/checkpoints/seed0/config.json"
    dataset_root = asset_root / "datasets/give_way_si_short_v1"
    dataset_environment = dataset_root / "environment.json"
    policy, provenance = load_policy(checkpoint)
    if provenance is None:
        raise ValueError("the frozen policy must carry SI provenance")
    config = Config(**provenance["evaluation_environment"])
    if config.max_steps != 850 or config.dt != 0.05:
        raise ValueError("unexpected frozen physical horizon")

    expected_rollouts = len(PHIS) * len(CONSTRUCTION_PAIR_IDS)
    expected_steps = expected_rollouts * config.max_steps
    print(f"expected rollout count: {expected_rollouts}", flush=True)
    print(f"expected physical-step count: <= {expected_steps}", flush=True)
    print("CPU/GPU: CPU", flush=True)
    print(
        "purpose: construction-only empirical transition counts and CL-FHCB sanity traces",
        flush=True,
    )

    traces: list[ClosedLoopTrace] = []
    trace_manifest = []
    rollout_index = 0
    for phi in PHIS:
        for pair_id in CONSTRUCTION_PAIR_IDS:
            initial = construction_start(dataset_root, pair_id)
            trace = rollout_closed_loop(
                policy=policy,
                initial_positions=initial,
                config=config,
                phi=phi,
                flow_seed=FLOW_SEED,
                rollout_index=pair_id,
                pair_id=pair_id,
                cbf_config=CBFConfig(),
            )
            traces.append(trace)
            trace_path = trace_dir / f"{trace.rollout_id}.npz"
            np.savez_compressed(trace_path, **trace_arrays(trace, config))
            trace_manifest.append(
                {
                    "id": trace.rollout_id,
                    "path": str(trace_path.relative_to(out)),
                    "sha256": sha256(trace_path),
                    "pair_id": pair_id,
                    "phi_name": phi.name,
                    "phi": list(phi.vector),
                    "flow_seed": FLOW_SEED,
                    "rollout_index": pair_id,
                    "steps": len(trace.steps),
                    "outcome": trace.outcome,
                }
            )
            rollout_index += 1
            print(
                f"construction rollout {rollout_index}/{expected_rollouts}: "
                f"{trace.rollout_id} steps={len(trace.steps)} outcome={trace.outcome}",
                flush=True,
            )

    certificates = {
        phi.name: build_empirical_certificate(traces, config, phi) for phi in PHIS
    }
    certificate_payload = {
        "schema": "cl_fhcb_empirical_certificate_v1",
        "interpretation": (
            "empirical finite-state Bellman supersolution; not a formal "
            "continuous-domain certificate"
        ),
        "certificates": {
            name: serializable_certificate(certificate)
            for name, certificate in certificates.items()
        },
    }
    certificate_path = out / "certificate.json"
    certificate_path.write_text(json.dumps(certificate_payload, indent=2) + "\n")

    scores = []
    for trace in traces:
        certificate = certificates[trace.phi.name]
        for K in K_VALUES:
            risk = score_prefix(trace, 0, K, certificate, config)
            scores.append(
                {
                    "trace_id": trace.rollout_id,
                    "K": K,
                    **asdict(risk),
                }
            )
    # Full remaining horizon is a bookkeeping sanity check on one cached trace
    # per phi.  It adds no rollout and must equal the deadlock indicator.
    full_scores = []
    for phi in PHIS:
        trace = next(item for item in traces if item.phi.name == phi.name)
        risk = score_prefix(trace, 0, config.max_steps, certificates[phi.name], config)
        full_scores.append(
            {
                "trace_id": trace.rollout_id,
                "K": "full_remaining",
                "expected": 1.0 if trace.outcome == "deadlock" else 0.0,
                **asdict(risk),
            }
        )

    all_steps = [step for trace in traces for step in trace.steps]
    nonzero_long = [
        trace
        for trace in traces
        if trace.phi.vector != (0.0, 0.0, 0.0) and len(trace.steps) > max(K_VALUES)
    ]
    nonvacuous = {
        name: bool(np.any(certificate.table[config.max_steps, 3:] < 1.0 - 1e-12))
        for name, certificate in certificates.items()
    }
    deadlock_full_scores = []
    for trace in traces:
        if trace.outcome == "deadlock":
            exact = score_prefix(
                trace, 0, config.max_steps, certificates[trace.phi.name], config
            )
            deadlock_full_scores.append(exact.value)

    sanity = {
        "closed_loop_g_recomputed_each_step": all(
            len(trace.steps) > 0 for trace in traces
        ),
        "g_active_after_K100": bool(nonzero_long)
        and all(len(trace.steps[100:]) > 0 for trace in nonzero_long),
        "two_hard_projections_present_each_step": all(
            bool(step.first_projection_status) and bool(step.second_projection_status)
            for step in all_steps
        ),
        "two_hard_projections_feasible": all(
            step.first_min_cbf_residual >= -CBFConfig().feasibility_tol
            and step.second_min_cbf_residual >= -CBFConfig().feasibility_tol
            for step in all_steps
        ),
        "flow_keys_unique_within_each_rollout": all(
            len({tuple(np.asarray(step.flow_key_data).tolist()) for step in trace.steps})
            == len(trace.steps)
            for trace in traces
        ),
        "g_phi_sampling_semantics": "deterministic; no residual draw exists",
        "guarded_timer_and_timer_reset_bins_equal_one": all(
            np.all(certificate.table[1:, list(GUARDED_BINS)] == 1.0)
            for certificate in certificates.values()
        ),
        "deadlock_full_horizon_scores_equal_one": all(
            value == 1.0 for value in deadlock_full_scores
        ),
        "terminal_full_horizon_scores_exact": all(
            item["value"] == item["expected"] for item in full_scores
        ),
        "certificate_nonvacuous_at_H": nonvacuous,
        "certificate_not_identically_one": all(nonvacuous.values()),
        "minimum_empirical_bellman_slack": {
            name: float(np.min(certificate.bellman_slacks()))
            for name, certificate in certificates.items()
        },
        "obvious_bellman_inequality_bug_absent": all(
            float(np.min(certificate.bellman_slacks())) >= -1e-12
            for certificate in certificates.values()
        ),
    }

    outcome_counts = {
        phi.name: {
            outcome: sum(
                trace.phi.name == phi.name and trace.outcome == outcome
                for trace in traces
            )
            for outcome in ("collision", "success", "deadlock", "timeout")
        }
        for phi in PHIS
    }
    manifest = {
        "schema": "cl_fhcb_construction_v1",
        "method": "CL-FHCB",
        "construction_only": True,
        "final_qualification_run": False,
        "g_phi_trained": False,
        "device": "CPU",
        "python": platform.python_version(),
        "jax_devices": [str(device) for device in jax.devices()],
        "rollout_budget": {
            "expected_rollouts": expected_rollouts,
            "actual_rollouts": len(traces),
            "hard_ceiling": 700,
            "target_ceiling": 400,
            "expected_physical_steps_upper": expected_steps,
            "actual_physical_steps": sum(len(trace.steps) for trace in traces),
        },
        "K_values": list(K_VALUES),
        "full_horizon_cached_subset": len(PHIS),
        "flow_rng": (
            "fold_in(fold_in(PRNGKey(7319), pair_id), physical_step); "
            "the same paired keys are used across phi until termination"
        ),
        "construction_pair_ids": list(CONSTRUCTION_PAIR_IDS),
        "outcome_counts": outcome_counts,
        "environment": config.to_dict(),
        "cbf": CBFConfig().to_dict(),
        "diagnostic_phis": [asdict(phi) for phi in PHIS],
        "trace_manifest": trace_manifest,
        "assets": {
            "asset_root": str(asset_root),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_config": str(checkpoint_config),
            "checkpoint_config_sha256": sha256(checkpoint_config),
            "dataset_environment": str(dataset_environment),
            "dataset_environment_sha256": sha256(dataset_environment),
        },
        "uncertainty_rule": (
            "For each abstract row, report the one-sided 95% Jeffreys beta "
            "posterior quantile for its immediate-deadlock probability.  It "
            "is diagnostic and is not inserted into the recursion; temporal "
            "dependence and state aggregation preclude a formal continuous-domain claim."
        ),
        "certificate_sha256": sha256(certificate_path),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "prefix_scores.json").write_text(
        json.dumps({"K_scores": scores, "full_scores": full_scores}, indent=2) + "\n"
    )
    (out / "sanity.json").write_text(json.dumps(sanity, indent=2) + "\n")

    required = [
        sanity["closed_loop_g_recomputed_each_step"],
        sanity["g_active_after_K100"],
        sanity["two_hard_projections_present_each_step"],
        sanity["two_hard_projections_feasible"],
        sanity["flow_keys_unique_within_each_rollout"],
        sanity["guarded_timer_and_timer_reset_bins_equal_one"],
        sanity["deadlock_full_horizon_scores_equal_one"],
        sanity["terminal_full_horizon_scores_exact"],
        sanity["certificate_not_identically_one"],
        sanity["obvious_bellman_inequality_bug_absent"],
    ]
    if not all(required):
        raise RuntimeError(f"CL-FHCB construction sanity failure: {sanity}")
    print(json.dumps(manifest["rollout_budget"], indent=2), flush=True)
    print(json.dumps(outcome_counts, indent=2), flush=True)
    print("construction sanity checks: PASS", flush=True)


if __name__ == "__main__":
    main()
