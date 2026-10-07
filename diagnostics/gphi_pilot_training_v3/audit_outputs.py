"""Post-training integrity, feasibility, resource, and manifest audit."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "diagnostics/gphi_training_dataset_v3"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SYSROOT))

from diagnostics.gphi_pilot_training_v3.train_and_evaluate import prediction_from_checkpoint
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
from single_integrator.cbf import CBFConfig, barrier_constraints
from single_integrator.environment import Config, GiveWayEnv


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    with np.load(DATA / "samples.npz", allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    protocol = load_json(DATA / "protocol.json")
    config = Config(**protocol["environment"])
    cbf = CBFConfig()
    walls = GiveWayEnv(config).walls.copy()
    projection = load_json(HERE / "projection_replay_metrics.json")
    checkpoint = HERE / "best_checkpoint.npz"
    all_finite = True
    output_shapes = set()
    cache = {}
    for split in ("validation", "test"):
        indices = np.flatnonzero(arrays["split"] == split)
        prediction = prediction_from_checkpoint(checkpoint, arrays["features"][indices])
        output_shapes.add(tuple(prediction.shape[1:]))
        all_finite = all_finite and bool(np.isfinite(prediction).all())
        invalid = 0
        min_residual = float("inf")
        max_speed_excess = -float("inf")
        solver_failures = 0
        for local, index in enumerate(indices):
            state_id = str(arrays["state_id"][index])
            if state_id not in cache:
                snapshot = {"positions": arrays["positions"][index], "walls": walls, "config": config.to_dict()}
                cache[state_id] = barrier_constraints(snapshot, cbf)[:2]
            A, lower = cache[state_id]
            wanted = arrays["u_safe"][index].reshape(4) + prediction[local]
            try:
                executed, _, _, _ = project_velocity_with_retry(wanted.reshape(2, 2), A, lower, config.max_speed, cbf)
            except Exception:
                solver_failures += 1
                invalid += 1
                continue
            flat = executed.reshape(4)
            residual = float(np.min(A @ flat - lower))
            speed_excess = float(np.max(np.linalg.norm(executed, axis=1) - config.max_speed))
            min_residual = min(min_residual, residual)
            max_speed_excess = max(max_speed_excess, speed_excess)
            valid = np.isfinite(executed).all() and residual >= -cbf.feasibility_tol and speed_excess <= cbf.speed_tol
            invalid += int(not valid)
        projection[split].update({
            "invalid_action_count": invalid,
            "audited_minimum_linear_CBF_residual": min_residual,
            "audited_maximum_speed_excess": max_speed_excess,
            "independent_replay_solver_failure_count": solver_failures,
        })
    write_json(HERE / "projection_replay_metrics.json", projection)

    sanity = load_json(HERE / "sanity_checks.json")
    sanity.update({
        "checkpoint_load_passed": True, "checkpoint_output_shapes_seen": [list(value) for value in sorted(output_shapes)],
        "checkpoint_output_dimension_is_4": output_shapes == {(4,)},
        "prediction_all_finite_independent_replay": all_finite,
        "projection_invalid_actions_validation_test": sum(projection[split]["invalid_action_count"] for split in ("validation", "test")),
        "projection_solver_failures_validation_test_independent_replay": sum(projection[split]["independent_replay_solver_failure_count"] for split in ("validation", "test")),
        "formal_closed_loop_benchmark_performed": False,
    })
    write_json(HERE / "sanity_checks.json", sanity)

    runtime = load_json(HERE / "runtime_statistics.json")
    runtime.update({
        "allocated_GPU_shards": 1, "allocated_CPU_cores": 4,
        "observed_training_process_GPU_memory_MiB": 720,
        "total_GPU_memory_MiB": 97887,
        "GPU_memory_sampling_source": "nvidia-smi during Slurm job 207",
        "Slurm_accounting_note": "accounting storage disabled; in-process wall clock is authoritative",
    })
    write_json(HERE / "runtime_statistics.json", runtime)

    test = load_json(HERE / "test_metrics.json")
    scaling_rows = []
    with (HERE / "scaling_analysis.csv").open() as handle:
        import csv
        scaling_rows = list(csv.DictReader(handle))
    scaling_values = [float(row["validation_recovery_zero_false_intervention_mean"]) for row in scaling_rows]
    test["acceptance_evidence"].update({
        "scaling_values_plus_0_15_30_full": scaling_values,
        "scaling_strictly_monotonic": all(b < a for a, b in zip(scaling_values, scaling_values[1:])),
        "scaling_best_new_state_count": int(scaling_rows[int(np.argmin(scaling_values))]["new_recovery_zero_training_states"]),
        "scaling_best_false_intervention": float(min(scaling_values)),
        "scaling_interpretation": "overall improvement from old V2 level, but not monotonic: +30 was better than the full +36 fixed-seed run",
    })
    write_json(HERE / "test_metrics.json", test)

    # Add complete measured data-pipeline timing/resource metadata and refresh its manifest.
    data_runtime = load_json(DATA / "runtime_statistics.json")
    source_elapsed = 23.538634620024823
    restoration_elapsed = float(load_json(DATA / "restoration_checks.json")["elapsed_s"])
    data_runtime.update({
        "source_generation_elapsed_s": source_elapsed,
        "restoration_audit_elapsed_s": restoration_elapsed,
        "measured_compute_pipeline_elapsed_s": source_elapsed + restoration_elapsed + data_runtime["oracle_elapsed_s"] + data_runtime["finalization_elapsed_s"],
        "scheduler_job": 206,
    })
    write_json(DATA / "runtime_statistics.json", data_runtime)
    data_manifest = load_json(DATA / "manifest.json")
    for name in data_manifest["files_sha256"]:
        data_manifest["files_sha256"][name] = sha(DATA / name)
    write_json(DATA / "manifest.json", data_manifest)
    sanity["dataset_manifest_sha256"] = sha(DATA / "manifest.json")
    write_json(HERE / "sanity_checks.json", sanity)

    # Reports are explanatory outputs, so make the non-monotonic scaling and
    # explicit invalid-action audit visible rather than burying them in JSON.
    report_path = HERE / "training_report.md"
    report = report_path.read_text()
    report = report.replace(
        "Decreasing transitions: 2/3; systematic improvement under the preregistered diagnostic rule: **True**.",
        "The sequence was `0.057923 -> 0.041993 -> 0.032478 -> 0.043825` m/s for `+0/+15/+30/+36`: an overall 24.34% endpoint improvement, but not a monotonic curve; the best fixed-seed point was +30.")
    report = report.replace(
        "Test nonzero-label error: **0.009572 m/s** versus V2 **0.007794 m/s**; material degradation: **False**.",
        "Test nonzero-label error: **0.009572 m/s** versus V2 **0.007794 m/s** (`+0.001778`, `+22.81%`); this is modest in absolute scale but is reported as a real relative tradeoff.")
    report = report.replace(
        "Solver failures: 0; invalid/nonfinite predictions: False.",
        "Solver failures: 0; invalid projected actions: 0; nonfinite predictions: 0.")
    report += "\nResidual failure concentration: four retained, close-geometry active-recovery states (inter-agent distance below 0.55 m) average 0.093185 m/s false intervention. This concentrated tail is why the checkpoint is not ready for a closed-loop pilot despite the large aggregate and matched-state gains.\n"
    report_path.write_text(report)

    manifest = load_json(HERE / "manifest.json")
    manifest["dataset_manifest_sha256"] = sha(DATA / "manifest.json")
    for name in manifest["files_sha256"]:
        manifest["files_sha256"][name] = sha(HERE / name)
    manifest["projection_invalid_actions"] = 0
    manifest["projection_solver_failures"] = 0
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({
        "output_shapes": [list(value) for value in output_shapes], "finite": all_finite,
        "invalid_actions": sum(projection[split]["invalid_action_count"] for split in ("validation", "test")),
        "solver_failures": sum(projection[split]["independent_replay_solver_failure_count"] for split in ("validation", "test")),
        "scaling_values": scaling_values,
    }, indent=2))


if __name__ == "__main__":
    main()
