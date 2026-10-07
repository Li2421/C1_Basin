"""Recover, exactly reproduce, and snapshot the 17 frozen strict deadlocks."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1"
OLD = ROOT / "diagnostics/gphi_wide_ic_cadence_v1"
FRESH = ROOT / "diagnostics/gphi_h8_fresh_unseen_generalization_v1"
PROTOCOL_SOURCE = ROOT / "diagnostics/success_basin_multimodality/protocol.json"
V1_PROTOCOL = ROOT / "diagnostics/gphi_training_dataset_v1/protocol.json"
# Insert the workspace first and the authoritative Toy tree second so that
# the Toy tree ends at index zero.  This is deliberately fail-closed against
# same-named workspace modules.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SYSROOT))

from diagnostics.gphi_closed_loop_pilot_v1.pilot_common import sha256, write_json
from diagnostics.gphi_training_dataset_v1.build_states import save_full
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
from single_integrator.cbf import CBFConfig, barrier_constraints
from single_integrator.environment import Config, GiveWayEnv, bounded_nominal
from single_integrator.evaluate import load_policy
from single_integrator.outcomes import first_event


EXPECTED_COUNTS = {"historical": 11, "fresh_unseen": 6}
CHECKPOINT = SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
EXPECTED_CHECKPOINT = "8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32"
OFFSETS = (("S0", None), ("S_8s", 160), ("S_4s", 80), ("S_2s", 40), ("S_1s", 20), ("S_pre", 1))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def condition_rows(directory: Path) -> list[dict[str, str]]:
    with (directory / "per_episode_results.csv").open() as stream:
        return [row for row in csv.DictReader(stream) if row["condition"] == "Safety" and row["outcome"] == "deadlock"]


def source_spec(benchmark: str, row: dict[str, str]) -> dict[str, Any]:
    base = OLD if benchmark == "historical" else FRESH
    index = int(row["episode_index"])
    raw = json.loads((base / "runs/production/raw/safety" / f"episode_{index:04d}.json").read_text())
    trajectory_path = base / raw["trajectory_file"]
    benchmark_manifest = OLD / "frozen_benchmark_manifest.json" if benchmark == "historical" else FRESH / "fresh_test_manifest.json"
    manifest = json.loads(benchmark_manifest.read_text())
    episodes = manifest.get("episodes", manifest.get("initial_conditions", manifest.get("records")))
    episode = next(item for item in episodes if int(item.get("episode_index", item.get("rollout_id", -1))) == index)
    return {
        "benchmark": benchmark, "episode_index": index,
        "case_id": f"{'old' if benchmark == 'historical' else 'fresh'}_r{index:03d}",
        "source_id": episode.get("source_id", f"historical_wide_r{index:03d}"),
        "ic_generator_seed": episode.get("ic_generator_seed", manifest.get("generator", {}).get("ic_generator_seed", 2026090902 if benchmark == "historical" else None)),
        "ic_draw_index": episode.get("ic_draw_index", index),
        "flow_root_seed": int(raw["flow_root_seed"]), "rollout_id": int(raw["rollout_id"]),
        "flow_key_semantics": raw["flow_key_semantics"],
        "initial_positions": raw["initial_positions"],
        "terminal_step": int(raw["episode_steps"]), "terminal_event": raw["outcome"],
        "source_raw_record": str(base / "runs/production/raw/safety" / f"episode_{index:04d}.json"),
        "source_raw_sha256": sha(base / "runs/production/raw/safety" / f"episode_{index:04d}.json"),
        "source_trajectory": str(trajectory_path), "source_trajectory_sha256": sha(trajectory_path),
        "source_benchmark_manifest": str(benchmark_manifest), "source_benchmark_manifest_sha256": sha(benchmark_manifest),
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    states_dir = HERE / "states"
    states_dir.mkdir(exist_ok=True)
    if sha(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise RuntimeError("Flow checkpoint mismatch")
    v1 = json.loads(V1_PROTOCOL.read_text())
    environment = v1["environment"]
    config = Config(**environment)
    cbf = CBFConfig()
    policy, provenance = load_policy(CHECKPOINT)
    if not provenance or provenance["evaluation_environment"] != environment:
        raise RuntimeError("Flow environment provenance mismatch")
    sample = jax.jit(lambda obs, key: policy.sample_actions(obs[None], seed=key)[0])
    sources = {
        "environment": Path(inspect.getsourcefile(GiveWayEnv)).resolve(),
        "projection": Path(inspect.getsourcefile(barrier_constraints)).resolve(),
        "retry": Path(inspect.getsourcefile(project_velocity_with_retry)).resolve(),
    }
    expected_sources = v1["frozen_hashes"]
    for name in ("environment", "projection", "retry"):
        if sha(sources[name]) != expected_sources[name]:
            raise RuntimeError(("frozen source mismatch", name, sha(sources[name]), expected_sources[name]))

    cases: list[dict[str, Any]] = []
    for benchmark, directory in (("historical", OLD), ("fresh_unseen", FRESH)):
        rows = condition_rows(directory)
        if len(rows) != EXPECTED_COUNTS[benchmark]:
            raise RuntimeError(("strict deadlock count mismatch", benchmark, len(rows)))
        cases.extend(source_spec(benchmark, row) for row in rows)
    if len(cases) != 17 or len({case["case_id"] for case in cases}) != 17:
        raise RuntimeError("not exactly 17 unique deadlock cases")

    query_rows: list[dict[str, Any]] = []
    reproduction: list[dict[str, Any]] = []
    max_diffs = Counter()
    for case in cases:
        with np.load(case["source_trajectory"], allow_pickle=False) as source:
            source_arrays = {key: np.asarray(source[key]) for key in source.files}
        terminal = case["terminal_step"]
        query_steps = {label: 0 if offset is None else terminal - offset for label, offset in OFFSETS}
        if min(query_steps.values()) < 0:
            raise RuntimeError(("missing requested checkpoint", case["case_id"], query_steps))
        env = GiveWayEnv(config)
        env.reset(np.asarray(case["initial_positions"], dtype=np.float64))
        episode_key = jax.random.fold_in(jax.random.PRNGKey(case["flow_root_seed"]), case["rollout_id"])
        saved: set[str] = set()
        first_events = []
        while not env.done:
            for label, step in query_steps.items():
                if env.step_count == step and label not in saved:
                    state_id = f"{case['case_id']}__{label}"
                    state_file = states_dir / f"{state_id}.npz"
                    save_full(state_file, env)
                    query_rows.append({
                        "state_id": state_id, "case_id": case["case_id"], "benchmark": case["benchmark"],
                        "episode_index": case["episode_index"], "query_label": label, "query_step": step,
                        "seconds_before_deadlock": (terminal - step) * config.dt,
                        "terminal_step": terminal, "remaining_global_steps": config.max_steps - step,
                        "state_file": str(state_file), "state_sha256": sha(state_file),
                        "rng_namespace": 310000 + len(query_rows),
                        "candidate_since": -1 if env.candidate_since is None else int(env.candidate_since),
                        "candidate_age_seconds": 0.0 if env.candidate_since is None else (env.step_count - env.candidate_since) * config.dt,
                        "stuck_timer": float(env.stuck_timer), "max_stuck_timer": float(env.max_stuck_timer),
                        "history_length": len(env.distance_history), "saved_history_tail": min(41, len(env.distance_history)),
                        "exact_flow_root_seed": case["flow_root_seed"], "exact_flow_rollout_id": case["rollout_id"],
                    })
                    saved.add(label)
            step = env.step_count
            observation = np.asarray(env.observation(), dtype=np.float32)
            step_key = jax.random.fold_in(episode_key, step)
            raw = np.asarray(sample(jnp.asarray(observation), step_key), dtype=np.float64)
            flow = bounded_nominal(raw, config.max_speed)
            A, lower, _ = barrier_constraints(env.snapshot(), cbf)
            safe, _, _, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
            if step >= len(source_arrays["step"]):
                raise RuntimeError(("source trajectory too short", case["case_id"], step))
            checks = {
                "positions_before": (env.positions, source_arrays["positions_before"][step]),
                "flow_step_key": (np.asarray(step_key, np.uint32), source_arrays["flow_step_key"][step]),
                "u_flow": (flow, source_arrays["u_flow"][step]),
                "u_safe": (safe, source_arrays["u_safe"][step]),
                "u_exec": (safe, source_arrays["u_exec"][step]),
            }
            for name, (actual, expected) in checks.items():
                diff = float(np.max(np.abs(np.asarray(actual, np.float64) - np.asarray(expected, np.float64))))
                max_diffs[name] = max(float(max_diffs[name]), diff)
                if diff > 1e-12:
                    raise RuntimeError(("Safety reproduction mismatch", case["case_id"], step, name, diff))
            _, _, _, info = env.step(safe)
            first_events.append(info["termination"])
            for name, actual in (
                ("candidate_since", -1 if env.candidate_since is None else env.candidate_since),
                ("stuck_timer", env.stuck_timer), ("max_stuck_timer", env.max_stuck_timer),
            ):
                expected = source_arrays[name][step]
                diff = float(np.max(np.abs(np.asarray(actual, np.float64) - np.asarray(expected, np.float64))))
                max_diffs[name] = max(float(max_diffs[name]), diff)
                if diff > 1e-12:
                    raise RuntimeError(("monitor reproduction mismatch", case["case_id"], step, name, diff))
        summary = env.summary()
        valid = (
            saved == set(query_steps) and env.step_count == terminal
            and summary["first_deadlock_step"] == terminal
            and first_events[-1] == "deadlock"
            and all(event == "running" for event in first_events[:-1])
        )
        reproduction.append({
            "case_id": case["case_id"], "benchmark": case["benchmark"], "episode_index": case["episode_index"],
            "expected_terminal_step": terminal, "reproduced_terminal_step": env.step_count,
            "first_deadlock_step": summary["first_deadlock_step"], "reproduced_outcome": first_events[-1],
            "query_states_saved": len(saved), "exact": valid,
        })
        if not valid:
            raise RuntimeError(("strict deadlock reproduction failed", reproduction[-1]))

    write_json(HERE / "strict_deadlock_manifest.json", {
        "schema": "strict_deadlock_source_manifest_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "authoritative Safety rows with outcome=deadlock; no manual episode inference",
        "cases": cases, "counts": dict(Counter(case["benchmark"] for case in cases)),
        "environment": environment, "cbf": cbf.to_dict(), "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha(CHECKPOINT),
        "source_paths": {name: str(path) for name, path in sources.items()},
        "source_sha256": {name: sha(path) for name, path in sources.items()},
    })
    write_json(HERE / "reproduction_audit.json", {
        "status": "PASS", "all_17_exact": all(row["exact"] for row in reproduction),
        "case_count": len(reproduction), "cases": reproduction,
        "maximum_absolute_differences": {key: float(value) for key, value in max_diffs.items()},
        "query_state_count": len(query_rows), "queries_per_episode": dict(Counter(row["case_id"] for row in query_rows)),
    })
    write_csv(HERE / "queried_states.csv", query_rows)
    write_json(HERE / "eta_search_config.json", {
        "schema": "strict_deadlock_frozen_eta_search_v1",
        "policy_family": json.loads(PROTOCOL_SOURCE.read_text())["policy_family"],
        "coarse_grid": json.loads(PROTOCOL_SOURCE.read_text())["phase_a_design"]["axes"],
        "coarse_points": 80,
        "frozen_domain": {"goal": [0.5, 1.25], "safe": [-0.5, 0.5], "relative": [0.0, 0.75]},
        "domain_reason": "exact axis-aligned envelope of the authoritative frozen 80-cell Phase-A lattice; no bound expansion",
        "stages": {
            "stage1": "all 80 authoritative lattice candidates on inherited exact Flow continuation",
            "stage2": "deterministic 0.125 one-ring refinement around exact-Flow successes, clipped to the frozen domain",
            "stage3": "256 deterministic scrambled Sobol candidates inside the same frozen domain when stages 1/2 find no success",
            "stage4": "0.0625 local one-ring refinement around successful cells, clipped to the frozen domain",
        },
        "continuation_horizon": "remaining global episode horizon through absolute step 850, matching the authoritative oracle-label pipeline",
        "eta_fixed_for_entire_continuation": True, "basis_recomputed_each_step": True,
        "robust_seed_count": 64, "robust_threshold": 63,
        "robust_seed_range": [95310001, 95310064],
        "exact_flow_stream": "inherit source benchmark root seed, rollout_id, and absolute physical step",
        "independent_flow_stream": "episode_key=fold_in(PRNGKey(oracle_seed), query_state rng_namespace); step_key=fold_in(episode_key, absolute physical step)",
        "projection": "authoritative exact hard projection with retry; same first/second chain",
        "frozen_before_search_outcomes": True,
        "authoritative_source_protocol": str(PROTOCOL_SOURCE), "authoritative_source_protocol_sha256": sha(PROTOCOL_SOURCE),
    })
    print(json.dumps({"status": "PASS", "cases": len(cases), "queries": len(query_rows), "max_diffs": dict(max_diffs)}, indent=2))


if __name__ == "__main__":
    main()
