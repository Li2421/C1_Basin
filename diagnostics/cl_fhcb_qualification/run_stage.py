"""Run one predeclared CPU rollout stage for frozen CL-FHCB qualification."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import jax
import numpy as np

from diagnostics.cl_fhcb.certificate import BIN_NAMES, abstract_bin, score_prefix
from diagnostics.cl_fhcb.closed_loop import rollout_closed_loop, state_metrics
from diagnostics.cl_fhcb_qualification.qualification_common import (
    K_VALUES,
    PHIS,
    REPO_ROOT,
    json_dump,
    load_certificates,
    locate_assets,
    sha256,
    verify_frozen_method,
)
from single_integrator.cbf import CBFConfig, barrier_constraints
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.evaluate import load_policy


PAIR_IDS = tuple(range(225, 233))
STAGE_SEEDS = {
    1: (19073,),
    2: (29401, 29402, 29403, 29404, 29405, 29406, 29407, 29408),
}
STAGE_PROPERTIES = {
    1: "P + independent continuation-certificate validity",
    2: "Q + PROJ + LHC + OAS + frozen-K ablation",
}


def initial_position(dataset_root: Path, pair_id: int) -> np.ndarray:
    with np.load(dataset_root / "raw" / f"episode_{2 * pair_id:04d}.npz") as item:
        return np.asarray(item["initial_positions"], dtype=np.float64).copy()


def active_signature(A, lower, action, max_speed, tolerance=1e-6) -> np.ndarray:
    flat = np.asarray(action, dtype=np.float64).reshape(4)
    cbf_active = (np.asarray(A) @ flat - np.asarray(lower)) <= tolerance
    speed_active = np.abs(
        np.linalg.norm(flat.reshape(2, 2), axis=-1) - max_speed
    ) <= tolerance
    return np.concatenate((cbf_active, speed_active))


def trace_arrays(trace, config: Config) -> dict[str, np.ndarray]:
    env = GiveWayEnv(config)
    goals = env.goals
    cbf = CBFConfig()
    first_active = []
    second_active = []
    bins_before = []
    bins_after = []
    for record in trace.steps:
        snapshot = {
            "positions": record.state.positions,
            "walls": env.walls,
            "config": config.to_dict(),
        }
        A, lower, _ = barrier_constraints(snapshot, cbf)
        first_active.append(active_signature(A, lower, record.u_safe, config.max_speed))
        second_active.append(active_signature(A, lower, record.u_exec, config.max_speed))
        bins_before.append(abstract_bin(state_metrics(record.state, config, goals), config))
        bins_after.append(abstract_bin(state_metrics(record.next_state, config, goals), config))
    return {
        "positions_before": np.asarray([record.state.positions for record in trace.steps]),
        "positions_after": np.asarray([record.next_state.positions for record in trace.steps]),
        "last_velocity_before": np.asarray([record.state.last_velocity for record in trace.steps]),
        "u_flow": np.asarray([record.u_flow for record in trace.steps]),
        "u_safe": np.asarray([record.u_safe for record in trace.steps]),
        "g": np.asarray([record.correction for record in trace.steps]),
        "w": np.asarray([record.candidate for record in trace.steps]),
        "u_exec": np.asarray([record.u_exec for record in trace.steps]),
        "event": np.asarray([record.event for record in trace.steps]),
        "candidate_since_before": np.asarray([
            -1 if record.state.candidate_since is None else record.state.candidate_since
            for record in trace.steps
        ], dtype=np.int32),
        "stuck_timer_after": np.asarray([
            record.monitor["stuck_timer"] for record in trace.steps
        ]),
        "flow_key_data": np.asarray([record.flow_key_data for record in trace.steps]),
        "first_active": np.asarray(first_active, dtype=bool),
        "second_active": np.asarray(second_active, dtype=bool),
        "first_min_cbf_residual": np.asarray([
            record.first_min_cbf_residual for record in trace.steps
        ]),
        "second_min_cbf_residual": np.asarray([
            record.second_min_cbf_residual for record in trace.steps
        ]),
        "abstract_bin_before": np.asarray(bins_before, dtype=np.int8),
        "abstract_bin_after": np.asarray(bins_after, dtype=np.int8),
        "pair_id": np.asarray(trace.pair_id),
        "flow_seed": np.asarray(trace.flow_seed),
        "phi": np.asarray(trace.phi.vector),
    }


def risk_records(trace, certificate, config: Config, stage: int) -> list[dict]:
    starts = [(0, "initial", len(trace.steps))]
    if stage == 1:
        for offset in (40, 150, 300):
            if len(trace.steps) >= offset:
                starts.append((len(trace.steps) - offset, f"terminal_minus_{offset}", offset))
    result = []
    for start, label, offset in starts:
        for K in K_VALUES:
            risk = score_prefix(trace, start, K, certificate, config)
            result.append({
                "trace_id": trace.rollout_id,
                "stage": stage,
                "pair_id": trace.pair_id,
                "flow_seed": trace.flow_seed,
                "phi_name": trace.phi.name,
                "phi": list(trace.phi.vector),
                "start_index": start,
                "checkpoint": label,
                "steps_to_terminal": offset,
                "K": K,
                "eventual_outcome": trace.outcome,
                "eventual_deadlock": trace.outcome == "deadlock",
                **asdict(risk),
            })
    full = score_prefix(trace, 0, config.max_steps, certificate, config)
    result.append({
        "trace_id": trace.rollout_id,
        "stage": stage,
        "pair_id": trace.pair_id,
        "flow_seed": trace.flow_seed,
        "phi_name": trace.phi.name,
        "phi": list(trace.phi.vector),
        "start_index": 0,
        "checkpoint": "initial",
        "steps_to_terminal": len(trace.steps),
        "K": "full_remaining_bookkeeping",
        "eventual_outcome": trace.outcome,
        "eventual_deadlock": trace.outcome == "deadlock",
        **asdict(full),
    })
    return result


def bellman_trace_records(trace, certificate, config: Config) -> list[dict]:
    goals = GiveWayEnv(config).goals
    grouped = defaultdict(list)
    values = []
    for record in trace.steps:
        before = state_metrics(record.state, config, goals)
        n = config.max_steps - record.state.step
        b_before = certificate.value(before, n, config)
        if record.event == "deadlock":
            target = 1.0
        elif record.event in ("success", "collision", "timeout"):
            target = 0.0
        else:
            after = state_metrics(record.next_state, config, goals)
            target = certificate.value(after, n - 1, config)
        cell = BIN_NAMES[abstract_bin(before, config)]
        grouped[cell].append(float(b_before - target))
        values.append(float(b_before))
    result = []
    for cell, slacks in grouped.items():
        result.append({
            "trace_id": trace.rollout_id,
            "pair_id": trace.pair_id,
            "flow_seed": trace.flow_seed,
            "phi_name": trace.phi.name,
            "cell": cell,
            "count": len(slacks),
            "mean_slack": float(np.mean(slacks)),
            "minimum_slack": float(np.min(slacks)),
            "negative_fraction": float(np.mean(np.asarray(slacks) < -1e-12)),
        })
    result.append({
        "trace_id": trace.rollout_id,
        "pair_id": trace.pair_id,
        "flow_seed": trace.flow_seed,
        "phi_name": trace.phi.name,
        "cell": "__all_values__",
        "count": len(values),
        "mean_slack": None,
        "minimum_slack": None,
        "negative_fraction": None,
        "fraction_value_one": float(np.mean(np.asarray(values) >= 1.0 - 1e-12)),
        "minimum_value": float(np.min(values)),
        "maximum_value": float(np.max(values)),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=int, choices=(1, 2))
    parser.add_argument("--asset-root", type=Path)
    args = parser.parse_args()
    stage = args.stage

    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)
    verify_frozen_method()
    assets = locate_assets(args.asset_root)
    checkpoint = assets / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
    dataset = assets / "datasets/give_way_si_short_v1"
    policy, provenance = load_policy(checkpoint)
    config = Config(**provenance["evaluation_environment"])
    certificates = load_certificates()

    out = REPO_ROOT / f"diagnostics/cl_fhcb_qualification/raw/stage{stage}"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"stage output must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    trace_dir = out / "traces"
    trace_dir.mkdir()

    seeds = STAGE_SEEDS[stage]
    rollout_count = len(PAIR_IDS) * len(PHIS) * len(seeds)
    maximum_steps = rollout_count * config.max_steps
    print(f"rollout count: {rollout_count}", flush=True)
    print(f"expected physical steps: <= {maximum_steps}", flush=True)
    print("CPU/GPU: CPU", flush=True)
    print(f"property being tested: {STAGE_PROPERTIES[stage]}", flush=True)

    manifest_records = []
    risks = []
    bellman = []
    completed = 0
    for flow_seed in seeds:
        for pair_id in PAIR_IDS:
            initial = initial_position(dataset, pair_id)
            for phi in PHIS:
                # rollout_index is pair_id for CRN pairing across policies; the
                # independent flow_seed distinguishes continuation replicates.
                trace = rollout_closed_loop(
                    policy=policy,
                    initial_positions=initial,
                    config=config,
                    phi=phi,
                    flow_seed=flow_seed,
                    rollout_index=pair_id,
                    pair_id=pair_id,
                    cbf_config=CBFConfig(),
                )
                unique_id = f"stage{stage}_{trace.rollout_id}"
                path = trace_dir / f"{unique_id}.npz"
                np.savez_compressed(path, **trace_arrays(trace, config))
                risks.extend(risk_records(trace, certificates[phi.name], config, stage))
                bellman.extend(
                    bellman_trace_records(trace, certificates[phi.name], config)
                )
                manifest_records.append({
                    "id": unique_id,
                    "source_trace_id": trace.rollout_id,
                    "relative_path": str(path.relative_to(out)),
                    "sha256": sha256(path),
                    "pair_id": pair_id,
                    "flow_seed": flow_seed,
                    "phi_name": phi.name,
                    "phi": list(phi.vector),
                    "steps": len(trace.steps),
                    "outcome": trace.outcome,
                    "deadlock_step": (
                        len(trace.steps) if trace.outcome == "deadlock" else None
                    ),
                })
                completed += 1
                print(
                    f"stage={stage} rollout={completed}/{rollout_count} "
                    f"pair={pair_id} seed={flow_seed} phi={phi.name} "
                    f"steps={len(trace.steps)} outcome={trace.outcome}",
                    flush=True,
                )

    stage_manifest = {
        "schema": f"cl_fhcb_qualification_stage{stage}_v1",
        "stage": stage,
        "property": STAGE_PROPERTIES[stage],
        "device": [str(device) for device in jax.devices()],
        "rollout_count": rollout_count,
        "expected_physical_steps_upper": maximum_steps,
        "actual_physical_steps": int(sum(item["steps"] for item in manifest_records)),
        "pair_ids": list(PAIR_IDS),
        "flow_seeds": list(seeds),
        "K": list(K_VALUES),
        "checkpoint_sha256": sha256(checkpoint),
        "environment": config.to_dict(),
        "records": manifest_records,
    }
    json_dump(out / "manifest.json", stage_manifest)
    json_dump(out / "risk_records.json", risks)
    json_dump(out / "bellman_records.json", bellman)
    print(
        json.dumps({
            "rollout_count": rollout_count,
            "actual_physical_steps": stage_manifest["actual_physical_steps"],
        }, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
