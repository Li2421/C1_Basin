"""Stage-A integration, archive reproduction, and zero-eta invariance audit."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
SYS = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/orthoflow3_representation_migration_v1"
ARCHIVE = ROOT / "diagnostics/double_bottleneck_eta_basis_redesign"
sys.path[:0] = [str(SYS), str(ROOT)]

from diagnostics.double_bottleneck_eta_basis_redesign.tools import run_rollouts as archived_runner  # noqa: E402
from diagnostics.double_bottleneck_eta_basis_redesign.tools.bases import (  # noqa: E402
    basis_terms as archived_basis_terms,
)
from diagnostics.gphi_training_dataset_v2.build_states import restore_full  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from double_bottleneck.environment import Config as DoubleConfig  # noqa: E402
from double_bottleneck.flowbc_4a_agent import load_checkpoint as load_double_checkpoint  # noqa: E402
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset  # noqa: E402
from shared_control.basis_families import ORTHOFLOW3_SCALE, get_basis_family  # noqa: E402
from single_integrator.cbf import CBFConfig, barrier_constraints  # noqa: E402
from single_integrator.environment import Config, bounded_nominal  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def matrix_stats(terms: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict:
    matrix = np.stack([value.reshape(-1) for value in terms], axis=1)
    gram = matrix.T @ matrix
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    cosine = gram / np.maximum(norms[:, None] * norms[None, :], 1e-12)
    return {"gram": gram, "cosine_gram": cosine, "column_norms": norms}


def archived_basis_reproduction() -> dict:
    anchors = json.loads((ARCHIVE / "state_anchor_manifest.json").read_text())["anchors"]
    archived = {
        row["state_id"]: row
        for row in (
            json.loads(line)
            for line in (ARCHIVE / "offline/basis_state_audit.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    dataset_paths = {
        "existing_untouched_test": ROOT / "diagnostics/double_bottleneck_initial_state_coverage/data/untouched_test_pool",
        "fresh_untouched_test": ROOT / "diagnostics/double_bottleneck_sxl_baseline_maturation/data/fresh_test_pool",
    }
    datasets = {
        name: FlowBC4ADataset(path, "val", seed=45 if name.startswith("existing") else 46)
        for name, path in dataset_paths.items()
    }
    lookup = {
        (name, family): dataset.by_family[family][0]
        for name, dataset in datasets.items()
        for family in dataset.family_names
    }
    selected = [row for row in anchors if row["trajectory_class"] != "successful_eta_corrected"][:12]
    max_error = defaultdict(float)
    for anchor in selected:
        with np.load(ROOT / anchor["trajectory_path"], allow_pickle=False) as trace:
            positions_all = np.asarray(trace["positions"], dtype=np.float64)
            step = int(anchor["source_step"])
            positions = positions_all[step]
            u_safe = np.asarray(trace["executed_actions"][step], dtype=np.float64)
        dataset = datasets[anchor["set"]]
        episode = lookup[(anchor["set"], anchor["family_id"])]
        config = DoubleConfig(**dataset.config)
        from double_bottleneck.environment import DoubleBottleneckEnv

        env = DoubleBottleneckEnv(config)
        env.reset(episode.initial_positions, regime=anchor["regime"])
        expected = archived[anchor["state_id"]]
        for family_name, archived_name, prefix in (
            ("p0", "P0-3D", "P0"),
            ("orthoflow3", "P1-OrthoFlow3", "P1"),
        ):
            actual_terms = get_basis_family(family_name).compute(
                positions, env.goals, u_safe, config.max_speed
            ).values
            reference_terms = archived_basis_terms(
                archived_name, positions, env.goals, u_safe, config.max_speed, ORTHOFLOW3_SCALE
            )
            for index, (actual, reference) in enumerate(zip(actual_terms, reference_terms, strict=True)):
                max_error[f"{prefix}_field_{index}"] = max(
                    max_error[f"{prefix}_field_{index}"], float(np.max(np.abs(actual - reference)))
                )
            stats = matrix_stats(actual_terms)
            for field in ("gram", "cosine_gram", "column_norms"):
                max_error[f"{prefix}_{field}"] = max(
                    max_error[f"{prefix}_{field}"],
                    float(np.max(np.abs(stats[field] - np.asarray(expected[f"{prefix}_raw_joint"][field])))),
                )
    return {"archived_anchor_states": len(selected), "maximum_absolute_errors": dict(max_error)}


def zero_invariance() -> dict:
    jax.config.update("jax_enable_x64", True)
    source = json.loads((ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1/integrity_audit.json").read_text())
    config = Config(**source["environment"])
    cbf = CBFConfig(**source["cbf"])
    policy, provenance = load_policy(
        SYS / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
    )
    if not provenance or provenance["evaluation_environment"] != source["environment"]:
        raise RuntimeError("FlowBC environment mismatch")
    subset = json.loads((HERE / "migration_subset_manifest.json").read_text())
    by_category = defaultdict(list)
    for state in subset["selected_states"]:
        by_category[state["category"]].append(state)
    states = [row for category in sorted(by_category) for row in by_category[category][:2]]
    maxima = defaultdict(float)
    records = []
    for state in states:
        for seed in state["matched_flow_seeds"][:4]:
            env = restore_full(Path(state["state_file"]), config)
            observation = env.observation()
            key0 = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(state["rng_namespace"]))
            raw = np.asarray(
                policy.sample_actions(jnp.asarray(observation[None]), seed=jax.random.fold_in(key0, env.step_count))[0],
                dtype=np.float64,
            )
            flow = bounded_nominal(raw, config.max_speed)
            A, lower, _ = barrier_constraints(env.snapshot(), cbf)
            safe, _, _, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
            for family_name in ("p0", "orthoflow3"):
                correction = get_basis_family(family_name).compute(
                    env.positions, env.goals, safe, config.max_speed
                ).correction(np.zeros(3))
                executed, _, _, _ = project_velocity_with_retry(
                    safe + correction, A, lower, config.max_speed, cbf
                )
                correction_error = float(np.max(np.abs(correction)))
                action_error = float(np.max(np.abs(executed - safe)))
                maxima[f"{family_name}_g"] = max(maxima[f"{family_name}_g"], correction_error)
                maxima[f"{family_name}_u_exec"] = max(maxima[f"{family_name}_u_exec"], action_error)
                records.append(
                    {
                        "state_id": state["state_id"], "category": state["category"],
                        "seed": seed, "basis_family": family_name,
                        "max_abs_g": correction_error, "max_abs_u_exec_minus_u_safe": action_error,
                    }
                )
    return {
        "state_count": len(states), "flow_realizations_per_state": 4,
        "comparisons": len(records), "maximum_deviations": dict(maxima),
        "records": records,
        "pass": all(value <= 1e-12 for value in maxima.values()),
    }


def archived_outcome_reproduction() -> dict:
    robust = json.loads((ARCHIVE / "full_seed_robustness.json").read_text())["episodes"][:3]
    raw_rows = {}
    for path in sorted((ARCHIVE / "raw").glob("P1_full_seed*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            raw_rows[(row["episode_id"], int(row["eta_index"]), int(row["seed"]))] = row
    datasets = {
        name: FlowBC4ADataset(path, "val", seed=45 if name.startswith("existing") else 46)
        for name, path in archived_runner.DATASETS.items()
    }
    policy, _ = load_double_checkpoint(
        archived_runner.CHECKPOINT,
        next(iter({dataset.environment_fingerprint for dataset in datasets.values()})),
    )
    lookup = {
        (name, family): dataset.by_family[family][0]
        for name, dataset in datasets.items()
        for family in dataset.family_names
    }

    def migrated_correction(name, theta, positions, goals, u_safe, max_speed, scale):
        if name != "P1-OrthoFlow3" or float(scale) != ORTHOFLOW3_SCALE:
            raise RuntimeError("unexpected archived migration arm")
        return get_basis_family("orthoflow3").compute(
            positions, goals, u_safe, max_speed
        ).correction(theta)

    original = archived_runner.correction
    records = []
    try:
        archived_runner.correction = migrated_correction
        for episode in robust:
            eta_index = int(episode["eta_robust"]["eta_index"])
            archived = raw_rows[(episode["episode_id"], eta_index, 2001)]
            reproduced = archived_runner.run_one(
                policy,
                datasets[episode["set"]],
                lookup[(episode["set"], episode["family_id"])],
                archived,
                ORTHOFLOW3_SCALE,
            )
            fields = (
                "success", "outcome", "episode_steps", "wall_collision", "agent_collision",
                "minimum_wall_clearance", "minimum_agent_clearance", "final_goal_errors",
            )
            exact = all(reproduced[field] == archived[field] for field in fields)
            records.append(
                {
                    "episode_id": episode["episode_id"], "eta_index": eta_index, "seed": 2001,
                    "archived_outcome": archived["outcome"], "reproduced_outcome": reproduced["outcome"],
                    "fields_exact": exact, "episode_steps": reproduced["episode_steps"],
                }
            )
    finally:
        archived_runner.correction = original
    return {
        "new_reproduction_rollouts": len(records),
        "new_physical_steps": sum(row["episode_steps"] for row in records),
        "records": records,
        "pass": all(row["fields_exact"] for row in records),
    }


def main() -> None:
    started = time.monotonic()
    basis = archived_basis_reproduction()
    # The archived Double-Bottleneck experiment used JAX's default precision.
    # Reproduce it before the Give-Way audit enables x64 globally; JAX precision
    # configuration is process-global and changing the order alters trajectories.
    outcomes = archived_outcome_reproduction()
    zero = zero_invariance()
    tests_pass = (
        max(basis["maximum_absolute_errors"].values(), default=0.0) <= 1e-12
        and zero["pass"] and outcomes["pass"]
    )
    archived = {
        "schema": "orthoflow3_archived_reproduction_audit_v1",
        "classification": "ORTHOFLOW3_INTEGRATION_PASS" if tests_pass else "ORTHOFLOW3_INTEGRATION_FAIL",
        "basis_reproduction": basis,
        "outcome_reproduction": outcomes,
        "historical_p0_sources_modified": False,
        "second_projection_source": str((ROOT / "diagnostics/success_basin_multimodality/exact_projector.py").resolve()),
        "second_projection_sha256": sha(ROOT / "diagnostics/success_basin_multimodality/exact_projector.py"),
        "wall_seconds": time.monotonic() - started,
    }
    atomic_json(HERE / "zero_eta_invariance.json", zero)
    atomic_json(HERE / "archived_reproduction_audit.json", archived)
    print(json.dumps({"classification": archived["classification"], "zero": zero["maximum_deviations"], "outcomes": outcomes["records"], "wall_seconds": archived["wall_seconds"]}, indent=2))
    if not tests_pass:
        raise RuntimeError("OrthoFlow3 integration gate failed")


if __name__ == "__main__":
    main()
