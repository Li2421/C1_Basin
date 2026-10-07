"""Gated independent replication and true-gradient validation rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.rollout_vi_r_cert import rollout

from diagnostics.rrisk_true_chunk.run_true_chunk import (
    atomic_json, chunk_conditioned_noise, chunk_diagnostics, digest,
    outcome_summary, setup_frozen, validate_assets,
)


HERE = Path(__file__).resolve().parent


def selected_pilot(protocol, analysis_path, records_path):
    analysis = json.loads(Path(analysis_path).read_text())
    gate = analysis["gate"]
    if not gate["pilot_discrimination"] or gate["replication"] is None:
        raise RuntimeError("pilot discrimination gate is closed")
    selected = gate["replication"]
    cell = selected["selected_cell"]
    pilot = json.loads(Path(records_path).read_text())
    rows = [row for row in pilot
            if row["state"]["state_id"] == cell["state_id"] and
            row["horizon_steps"] == cell["horizon_steps"]]
    by_id = {row["arm_id"]: row for row in rows}
    return analysis, cell, by_id, selected


def true_gradient(by_id, selected):
    nonzero_ids = [arm_id for arm_id in selected["arm_ids"]
                   if arm_id != "baseline"]
    if not nonzero_ids:
        raise RuntimeError("selected contrast has no nonzero arm")
    delta = float(by_id[nonzero_ids[0]]["alpha_norm"])
    if not all(np.isclose(by_id[arm_id]["alpha_norm"], delta)
               for arm_id in nonzero_ids):
        raise RuntimeError("selected contrast mixes correction magnitudes")
    gradient = np.empty(4, np.float64)
    coordinate_rows = {}
    for axis in range(4):
        plus = next(row for row in by_id.values()
                    if row["axis"] == axis and row["sign"] == 1 and
                    np.isclose(row["alpha_norm"], delta))
        minus = next(row for row in by_id.values()
                     if row["axis"] == axis and row["sign"] == -1 and
                     np.isclose(row["alpha_norm"], delta))
        gradient[axis] = (plus["Q_D"] - minus["Q_D"]) / (2 * delta)
        coordinate_rows[str(axis)] = dict(
            plus_arm=plus["arm_id"], plus_Q_D=plus["Q_D"],
            minus_arm=minus["arm_id"], minus_Q_D=minus["Q_D"],
            central_difference=float(gradient[axis]))
    norm = float(np.linalg.norm(gradient))
    if norm <= 0:
        raise RuntimeError("pilot coordinate differences give zero true gradient")
    return dict(delta=delta, gradient=gradient.tolist(), gradient_norm=norm,
                coordinate_rows=coordinate_rows)


def build_validation_arms(protocol, by_id, selected, stage):
    if stage == "replication":
        return [dict(arm_id=arm_id, alpha=by_id[arm_id]["alpha"])
                for arm_id in selected["arm_ids"]], None
    geometry = true_gradient(by_id, selected)
    gradient = np.asarray(geometry["gradient"])
    delta = geometry["delta"]
    unit = gradient / np.linalg.norm(gradient)
    arms = [
        dict(arm_id="baseline", alpha=np.zeros(4).tolist()),
        dict(arm_id="minus_g_TRUE", alpha=(-delta * unit).tolist()),
        dict(arm_id="plus_g_TRUE", alpha=(delta * unit).tolist()),
    ]
    rng = np.random.default_rng(protocol["future_seeds"]["random_direction_seed"])
    for index in range(4):
        direction = rng.normal(size=4)
        direction /= np.linalg.norm(direction)
        arms.append(dict(arm_id=f"random_{index}",
                         alpha=(delta * direction).tolist()))
    return arms, geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path,
                        default=HERE / "predeclared_protocol.json")
    parser.add_argument("--pilot-analysis", type=Path, required=True)
    parser.add_argument("--pilot-records", type=Path, required=True)
    parser.add_argument("--replication-analysis", type=Path)
    parser.add_argument("--stage", choices=("replication", "gradient_validation"),
                        required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    validate_assets(protocol)
    analysis, cell, by_id, selected = selected_pilot(
        protocol, args.pilot_analysis, args.pilot_records)
    if args.stage == "gradient_validation":
        if args.replication_analysis is None:
            raise RuntimeError("gradient validation requires replication analysis")
        replication = json.loads(args.replication_analysis.read_text())
        if not replication["true_descent_replicated"]:
            raise RuntimeError("independent replication gate is closed")
    if jax.default_backend() != "gpu":
        raise RuntimeError("true continuation outcomes require Slurm GPU")
    arms, gradient_geometry = build_validation_arms(
        protocol, by_id, selected, args.stage)
    seeds = protocol["future_seeds"][
        "replication" if args.stage == "replication" else "gradient_validation"]
    state = next(row["state"] for row in by_id.values())
    rid, action_step = int(state["rid"]), int(state["action_step"])
    horizon = int(cell["horizon_steps"])
    params, field, plant, cbf = setup_frozen(protocol)
    start_protocol = json.loads(Path(protocol["assets"]["start_protocol"]).read_text())
    starts = np.asarray(start_protocol["starts"], np.float64)
    full_fn = jax.jit(lambda initial, draws, offsets: rollout(
        params, field, initial, draws, plant, cbf, offsets))
    frozen = protocol["frozen_system"]
    reference_item = protocol["assets"]["reference_traces"][str(rid)]
    with np.load(reference_item["path"], allow_pickle=False) as trace:
        reference_before = np.asarray(trace["before"][action_step])
    args.out.mkdir(parents=True, exist_ok=True)
    runtime = dict(
        schema="rrisk_true_chunk_validation_runtime_v1", stage=args.stage,
        protocol_sha256=digest(args.protocol), source_sha256=digest(Path(__file__)),
        backend=jax.default_backend(), started=time.time(), no_training=True,
        selected_cell=cell, seeds=seeds,
    )
    atomic_json(args.out / "runtime.json", runtime)
    records = []
    for arm in arms:
        alpha = np.asarray(arm["alpha"], np.float64)
        offsets = np.zeros((plant.max_steps, 4), np.float64)
        offsets[action_step:action_step + horizon] = alpha
        outcomes = []
        for seed in seeds:
            draws = chunk_conditioned_noise(
                frozen["prefix_seed"], frozen["reference_suffix_seed"], seed,
                rid, action_step, frozen["old_q_split_action"])
            terms, trace = full_fn(jnp.asarray(starts[rid]), draws,
                                   jnp.asarray(offsets))
            state_error = float(np.max(np.abs(
                np.asarray(jax.device_get(trace["before"][action_step])) -
                reference_before)))
            if state_error > 2e-10:
                raise RuntimeError(f"reference replay mismatch: {state_error}")
            result = outcome_summary(jax.device_get(terms))
            result.update(
                seed=seed, reference_state_max_abs_error=state_error,
                chunk=chunk_diagnostics(trace, action_step, horizon, alpha))
            outcomes.append(result)
        deadlocks = int(sum(row["D_H"] for row in outcomes))
        records.append(dict(
            arm_id=arm["arm_id"], alpha=alpha.tolist(),
            alpha_norm=float(np.linalg.norm(alpha)), state=state,
            horizon_steps=horizon, horizon_seconds=horizon * plant.dt,
            continuations=len(outcomes), deadlocks=deadlocks,
            Q_D=deadlocks / len(outcomes), outcomes=outcomes))
        atomic_json(args.out / "records.json", records)
        print(f"{args.stage} {arm['arm_id']} D={deadlocks}/{len(outcomes)}",
              flush=True)
    if gradient_geometry is not None:
        atomic_json(args.out / "gradient_estimate.json", gradient_geometry)
    runtime.update(completed=time.time(), elapsed_seconds=time.time()-runtime["started"],
                   arms=len(records), continuations=sum(row["continuations"]
                                                        for row in records))
    atomic_json(args.out / "runtime.json", runtime)
    atomic_json(args.out / "complete.json", dict(
        complete=True, stage=args.stage, training_steps=0,
        arms=len(records), continuations=runtime["continuations"]))


if __name__ == "__main__":
    main()
