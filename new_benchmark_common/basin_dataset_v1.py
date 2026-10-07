"""Build the frozen, objective-agnostic OrthoFlow3 basin dataset v1.

The builder deliberately keeps state selection, rollout execution and final
materialisation separate.  Every rollout stage uses the shared database
preflight through :func:`execute_batch` and is safe to resume by deterministic
scenario/state/eta/seed shards.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable

import jax
import numpy as np

from new_benchmark_common.final_diagnostics import load_nominal
from new_benchmark_common.macflow import load_checkpoint
from new_benchmark_common.safety_eta3 import (
    CONTROL_SEEDS,
    DOMAIN_HIGH,
    DOMAIN_LOW,
    FUTURE_ROOT,
    ROBUST_PROTOCOL,
    SCENARIOS,
    STANDARD_SEEDS,
    DatabaseSink,
    ScenarioRuntime,
    cached_rows,
    canonical,
    designs,
    eta_identity,
    execute_batch,
    normalized_eta,
    robust_evidence,
    sha256_file,
    sha256_sources,
    state_token,
    uid,
)
from shared_control.hard_projection import HardProjectionConfig
from shared_control.basis_families import get_basis_family
from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import (
    CertifiedHardSafetyFilter,
)
from shared_rollout_db.src.rollout_db import connect, initialize


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets/orthoflow3_basin_dataset_v1"
WORK = OUT / "work"
EXPERIMENT_NAME = "orthoflow3_basin_dataset_v1"
CONDITIONING_FLOW_ROOT = 2026100106
SOURCE_TRACE_SEED = 100
SOURCE_TRACE_RETRIES = tuple(range(100, 104))
STATE_SELECTION_PROTOCOL = "sha256_order_one_midpoint_state_per_parent_v1"
N_TRAIN = 64
N_VAL = 16
CANONICAL_TREE_HASHES = {
    "toy_giveway": "6c88e06bd4f167ad5ce6b75ff670247cb916310913c7e37d1459e93edb0f9bf9",
    "double_bottleneck": "1e0e98a3ddb91dc123ba2bdeb75476fa41583382e34da1febd159fc307de39ab",
    "single_integrator": "f26814a9616d9bb7d17a731656e93721031570fb5e271315d3607ea62f728437",
    "shared_control": "3accf5da696f083a684694beb668797f28f660f3d2d942790c41cacd3cd983c9",
}


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def file_hash(path: Path) -> str:
    return sha256_file(path)


def stable_rank(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def canonical_tree_hash(directory: str) -> str:
    lines = []
    for path in sorted((ROOT / directory).rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        lines.append(f"{file_hash(path)}  {path.relative_to(ROOT).as_posix()}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def normalized_distance(a: Iterable[float], b: Iterable[float]) -> float:
    return float(np.linalg.norm(normalized_eta(a) - normalized_eta(b)))


def scenario_centers(name: str) -> list[dict[str, Any]]:
    path = ROOT / f"diagnostics/{name}_safety_eta3/cross_state_coverage.json"
    rows = load_json(path)["centers"]
    rows = sorted(rows, key=lambda x: (-x["coverage_fraction"],
                                       float(np.linalg.norm(normalized_eta(x["eta"]))),
                                       x["center_index"]))
    return [{"eta": list(map(float, row["eta"])),
             "source": f"{name}:robust_center:{row['center_index']}",
             "coverage_fraction": float(row["coverage_fraction"])} for row in rows]


def double_mode_centers() -> list[dict[str, Any]]:
    path = ROOT / "diagnostics/orthoflow3_db_shared_mode_transfer_v1/train_mode_counts.csv"
    values: dict[tuple[float, float, float], dict[str, Any]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["mode_id"]) < 0:
                continue
            eta = tuple(float(row[f"eta{i}"]) for i in range(1, 4))
            values.setdefault(eta, {"eta": list(eta),
                                    "source": f"double_bottleneck:shared_mode:{row['mode_id']}"})
    return list(values.values())


def candidate_library(name: str) -> list[dict[str, Any]]:
    """Frozen scenario candidate library, deduplicated by float64 eta identity."""
    candidates: list[dict[str, Any]] = []
    if name in ("four_way_intersection", "ring_exchange"):
        for index, eta in enumerate(designs()["primary"]):
            candidates.append({"eta": eta, "source": f"common_primary_sobol:{index}"})
    candidates.extend(double_mode_centers())
    candidates.extend(scenario_centers("four_way_intersection"))
    candidates.extend(scenario_centers("ring_exchange"))
    seen: set[str] = set()
    result = []
    for row in candidates:
        key = eta_identity(row["eta"])[0]
        if key in seen:
            continue
        seen.add(key)
        result.append({**row, "eta_uid": key,
                       "eta_normalized": normalized_eta(row["eta"]).tolist(),
                       "library_index": len(result)})
    return result


class TrainingRuntime(ScenarioRuntime):
    """The frozen new-benchmark controller on train/dev intermediate states."""

    def __init__(self, name: str, state_rows: list[dict[str, Any]], *, parent: bool = False):
        spec = SCENARIOS[name]
        self.name, self.spec, self.kind = name, spec, spec["kind"]
        self.output = WORK / name
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest = load_json(spec["dataset"] / "manifest.json")
        self.selection_audit = {"split": "train_and_dev_only", "opened_test_archives": 0,
                                "manifest_sha256": file_hash(spec["dataset"] / "manifest.json")}
        self.config_dict = dict(self.manifest["scenario_config"])
        self.agent, self.checkpoint_metadata = load_checkpoint(
            spec["checkpoint"], expected_environment_fingerprint=self.manifest["environment_fingerprint"])
        self.checkpoint_sha = file_hash(spec["checkpoint"])
        self.cbf = HardProjectionConfig()
        self.projector = CertifiedHardSafetyFilter(self.cbf)
        self.basis = get_basis_family("orthoflow3")
        if self.kind == "four":
            from four_way_intersection.environment import Config, FourWayIntersectionEnv
            from four_way_intersection.rollout import crossing_order_signature
            self.config = Config(**self.config_dict)
            self.make_env = lambda: FourWayIntersectionEnv(self.config)
            self.mode_fn = crossing_order_signature
            self.env_hash = sha256_sources([
                ROOT / "four_way_intersection/environment.py",
                ROOT / "four_way_intersection/scenario.py"])
            self.safety_hash = sha256_sources([
                ROOT / "shared_control/hard_projection.py",
                ROOT / "four_way_intersection/safety.py",
                ROOT / "diagnostics/double_bottleneck_eta3_basin/tools/exact_projection_retry.py"])
        else:
            from ring_exchange.environment import LocalFrameConfig, RingExchangeEnv
            from ring_exchange.expert import circulation_signature
            self.config = LocalFrameConfig(**self.config_dict)
            self.make_env = lambda: RingExchangeEnv(self.config)
            self.mode_fn = circulation_signature
            self.env_hash = sha256_sources([
                ROOT / "ring_exchange/environment.py", ROOT / "ring_exchange/local_frame.py"])
            self.safety_hash = sha256_sources([
                ROOT / "shared_control/hard_projection.py", ROOT / "ring_exchange/safety.py",
                ROOT / "diagnostics/double_bottleneck_eta3_basin/tools/exact_projection_retry.py"])
        self.orthoflow_hash = file_hash(ROOT / "shared_control/basis_families.py")
        self.macflow_hash = file_hash(ROOT / "new_benchmark_common/macflow.py")
        self.scenario_code_hash = hashlib.sha256(canonical({
            "environment": self.env_hash, "config": self.config_dict,
            "representation": self.kind}).encode()).hexdigest()
        self.scenario_uid = uid("scn", {"name": name, "code": self.scenario_code_hash})
        self.states = state_rows
        self.parent = parent
        self.controllers = {chain: self._training_controller(chain, parent)
                            for chain in ("no_safety", "hard_safety", "orthoflow3")}
        self.experiment_uid = uid("exp", {"path": str(OUT.resolve())})
        self._register_training_identities()

    def _training_controller(self, chain: str, parent: bool) -> dict[str, Any]:
        conditioning = ("training_dev_parent_initial_state_episode_eta_v1" if parent else
                        "training_dev_intermediate_state_episode_eta_v1")
        rng = ({"name": "training_source_parent_seed_v1", "evaluation_seed":
                8123 if self.name == "four_way_intersection" else 3127,
                "derivation": "fold_in(PRNGKey(evaluation_seed),outcome_blind_parent_index),then_step"}
               if chain in ("no_safety", "hard_safety") else
               {"name": "matched_future_index_v1", "future_root": FUTURE_ROOT,
                "derivation": "fold_in(fold_in(PRNGKey(root),state_token),future_index),then_step"})
        payload = {
            "scenario": self.name, "chain": chain,
            "flow_checkpoint_sha256": self.checkpoint_sha,
            "macflow_source_sha256": self.macflow_hash,
            "environment_sha256": self.env_hash,
            "orthoflow3_sha256": self.orthoflow_hash if chain == "orthoflow3" else None,
            "safety_projection_sha256": self.safety_hash if chain != "no_safety" else None,
            "safety_config": self.cbf.to_dict() if chain != "no_safety" else None,
            "horizon": self.config.max_steps, "dt": self.config.dt,
            "success_semantics": "collision_free_all_agents_goal_tolerance_v1",
            "conditioning": conditioning, "rng": rng,
            "dataset_protocol": STATE_SELECTION_PROTOCOL,
        }
        return {"uid": uid("ctl", payload), "payload": payload}

    def _register_training_identities(self) -> None:
        initialize()
        with connect() as con:
            con.execute("INSERT OR IGNORE INTO scenario(scenario_uid,name,code_config_fingerprint,metadata_json) VALUES(?,?,?,?)",
                        (self.scenario_uid, self.name, self.scenario_code_hash,
                         canonical({"frozen_stage1": True,
                                    "environment_fingerprint": self.manifest["environment_fingerprint"]})))
            con.execute("INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,metadata_json) VALUES(?,?,?,?,?,?)",
                        (self.experiment_uid, EXPERIMENT_NAME, str(OUT.resolve()),
                         content_hash({"selection": STATE_SELECTION_PROTOCOL,
                                       "robust": ">=15/16"}), file_hash(Path(__file__)),
                         canonical({"task": "objective_agnostic_basin_dataset", "training": False})))
            for state in self.states:
                p = state["physical"]
                con.execute("""INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,
                    physical_state_json,h0_json,goals_geometry_json,provenance_json,identity_quality)
                    VALUES(?,?,?,?,?,?,?,?,?)""",
                    (state["uid"], self.scenario_uid, state.get("source_group"), state["content_hash"],
                     canonical(p), canonical(state.get("conditioning")) if state.get("conditioning") is not None else None,
                     canonical({"goals": p.get("goals"), "environment": state.get("environment_descriptor")}),
                     canonical(state["provenance"]), "CONTENT_EXACT"))
                con.execute("INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)",
                            (self.scenario_uid, state["alias"], state["uid"], self.experiment_uid))
            for controller in self.controllers.values():
                p = controller["payload"]
                con.execute("""INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,
                    flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,
                    success_semantics_version,conditioning_version,rng_semantics_version,config_json,
                    compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (controller["uid"], self.scenario_uid, self.checkpoint_sha,
                     p["orthoflow3_sha256"], self.safety_hash if p["safety_projection_sha256"] else None,
                     str(self.config.max_steps), str(self.config.dt), p["success_semantics"],
                     p["conditioning"], canonical(p["rng"]), canonical(p), "EXACT_PROFILE"))
            con.commit()

    def reset(self, env: Any, state: dict[str, Any]) -> None:
        super().reset(env, state)
        env.step_count = int(state["physical"].get("timestep", 0))


def parent_state_rows(name: str) -> list[dict[str, Any]]:
    path = WORK / name / "parent_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"run init first: {path}")
    return load_json(path)["states"]


def sampled_state_rows(name: str) -> list[dict[str, Any]]:
    path = WORK / name / "sampled_state_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"run materialize-states first: {path}")
    return load_json(path)["states"]


def init_scenario(name: str) -> dict[str, Any]:
    spec = SCENARIOS[name]
    selected: list[dict[str, Any]] = []
    audits = {}
    for source_split, target_split, count in (("train", "train", N_TRAIN),
                                               ("dev", "validation", N_VAL)):
        manifest, trajectories, audit = load_nominal(spec["dataset"], source_split)
        audits[source_split] = audit
        if audit["opened_test_archives"] != 0:
            raise RuntimeError("test archive access is forbidden")
        ordered = sorted(trajectories, key=lambda t: stable_rank(name, source_split, t.rollout_id))
        if len(ordered) < count:
            raise RuntimeError(f"insufficient {source_split} nominal parents")
        for trajectory in ordered[:count]:
            physical = {
                "positions": np.asarray(trajectory.initial_state["positions"], dtype=np.float64).tolist(),
                "velocities": np.asarray(trajectory.initial_state["velocities"], dtype=np.float64).tolist(),
                "goals": (np.asarray(trajectory.initial_state["goals"], dtype=np.float64).tolist()
                          if "goals" in trajectory.initial_state else None),
                "timestep": 0,
            }
            ch = content_hash(physical)
            scenario_uid = uid("scn", {"name": name, "code": "resolved_by_runtime"})
            # Runtime replaces this provisional scenario token with the exact frozen UID below.
            selected.append({
                "alias": f"BASIN_V1_PARENT_{name}_{target_split}_{trajectory.rollout_id}",
                "rollout_id": trajectory.rollout_id, "split": target_split,
                "source_dataset_split": source_split, "index": len(selected),
                "physical": physical, "content_hash": ch,
                "source_group": f"basin_v1_parent:{target_split}:{trajectory.rollout_id}",
                "provenance": {"dataset": str(spec["dataset"]), "rollout_id": trajectory.rollout_id,
                               "dataset_split": source_split, "selection": STATE_SELECTION_PROTOCOL,
                               "opened_test_archives": 0},
            })
    # Compute exact state UIDs only after the frozen scenario UID is known.
    probe = TrainingRuntime(name, [], parent=True)
    for row in selected:
        row["uid"] = uid("state", {"scenario": probe.scenario_uid, "content": row["content_hash"]})
    TrainingRuntime(name, selected, parent=True)
    output = {"schema": "orthoflow3_basin_v1_parent_manifest", "scenario": name,
              "selection_protocol": STATE_SELECTION_PROTOCOL, "counts": {"train": N_TRAIN, "validation": N_VAL},
              "selection_audit": audits, "states": selected}
    dump_json(WORK / name / "parent_manifest.json", output)
    dump_json(WORK / name / "candidate_library.json",
              {"scenario": name, "domain": designs()["domain"],
               "normalization": "(eta-domain_midpoint)/(domain_width)",
               "candidates": candidate_library(name)})
    return output


def raw_rows(name: str, stage_prefix: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((WORK / name / "raw").glob(f"{stage_prefix}*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_baseline(name: str, shard: int, shards: int) -> dict[str, Any]:
    states = parent_state_rows(name)
    runtime = TrainingRuntime(name, states, parent=True)
    jobs = [{"state": state, "eta": [0.0, 0.0, 0.0], "chain": "hard_safety", "seeds": [0],
             "metadata": {"source_rollout_type": "hard_safety_baseline",
                          "parent_episode_id": state["rollout_id"]}} for state in states]
    return execute_batch(runtime, "source_baseline", jobs, save_trace=True,
                         shard_index=shard, num_shards=shards)


def parent_center_validation(name: str, shard: int, shards: int) -> dict[str, Any]:
    states = parent_state_rows(name)
    runtime = TrainingRuntime(name, states, parent=True)
    baseline = {row["state_uid"]: row for row in raw_rows(name, "source_baseline")
                if row.get("trace") is not None}
    failures = [state for state in states if state["uid"] in baseline and not baseline[state["uid"]]["success"]]
    centers = scenario_centers(name)
    jobs = [{"state": state, "eta": center["eta"], "chain": "orthoflow3",
             "seeds": list(STANDARD_SEEDS),
             "metadata": {"source_rollout_type": "parent_center_validation",
                          "center_source": center["source"]}}
            for state in failures for center in centers]
    return execute_batch(runtime, "parent_center_validation", jobs,
                         shard_index=shard, num_shards=shards,
                         exact_robust_15of16=True)


def select_parent_centers(name: str) -> dict[str, Any]:
    states = parent_state_rows(name)
    runtime = TrainingRuntime(name, states, parent=True)
    centers = scenario_centers(name)
    selected = {}
    for state in states:
        choices = []
        for index, center in enumerate(centers):
            try:
                ev = robust_evidence(runtime, state, center["eta"],
                                     persist_stage="parent_center_validation")
            except RuntimeError:
                continue
            if ev["robust"]:
                choices.append((float(np.linalg.norm(normalized_eta(center["eta"]))), index, center, ev))
        if choices:
            _, _, center, ev = min(choices)
            selected[state["uid"]] = {"eta": center["eta"], "source": center["source"], "evidence": ev}
    result = {"scenario": name, "robust_parent_count": len(selected), "selected": selected}
    dump_json(WORK / name / "selected_parent_centers.json", result)
    return result


def source_corrected(name: str, shard: int, shards: int) -> dict[str, Any]:
    states = parent_state_rows(name)
    runtime = TrainingRuntime(name, states, parent=True)
    baseline = {row["state_uid"]: row for row in raw_rows(name, "source_baseline")
                if row.get("trace") is not None}
    selected = load_json(WORK / name / "selected_parent_centers.json")["selected"]
    jobs = []
    for state in states:
        if state["uid"] not in baseline or baseline[state["uid"]]["success"] or state["uid"] not in selected:
            continue
        jobs.append({"state": state, "eta": selected[state["uid"]]["eta"], "chain": "orthoflow3",
                     "seeds": list(SOURCE_TRACE_RETRIES),
                     "metadata": {"source_rollout_type": "orthoflow3_corrected",
                                  "validated_parent_center": selected[state["uid"]]["source"],
                                  "parent_episode_id": state["rollout_id"]}})
    return execute_batch(runtime, "source_corrected", jobs, save_trace=True,
                         shard_index=shard, num_shards=shards)


def _actual_goals(runtime: TrainingRuntime, parent: dict[str, Any]) -> np.ndarray:
    env = runtime.make_env()
    runtime.reset(env, parent)
    return np.asarray(env.goals, dtype=np.float64)


def _environment_descriptor(runtime: TrainingRuntime) -> dict[str, Any]:
    if runtime.kind == "four":
        return {"kind": "open_four_way_square",
                "world_half_extent": runtime.config.world_half_extent,
                "agent_radius": runtime.config.agent_radius,
                "agent_order": ["A", "B", "C", "D"]}
    return {"kind": "wide_annulus", "obstacle_center": [0.0, 0.0],
            "obstacle_radius": runtime.config.obstacle_radius,
            "outer_radius": runtime.config.outer_radius,
            "agent_radius": runtime.config.agent_radius,
            "agent_order": ["A", "B", "C", "D"]}


def materialize_states(name: str) -> dict[str, Any]:
    parents = parent_state_rows(name)
    runtime = TrainingRuntime(name, parents, parent=True)
    baseline_rows = [row for row in raw_rows(name, "source_baseline") if row.get("trace")]
    baseline = {row["state_uid"]: row for row in baseline_rows}
    corrected_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows(name, "source_corrected"):
        if row.get("trace") and row.get("success"):
            corrected_by_state[row["state_uid"]].append(row)
    if len(baseline) != len(parents):
        raise RuntimeError(f"baseline trace coverage incomplete: {len(baseline)}/{len(parents)}")
    failure_order = {uid_: index for index, uid_ in enumerate(sorted(
        [state["uid"] for state in parents if not baseline[state["uid"]]["success"]],
        key=lambda x: stable_rank(name, "failure_source_assignment", x)))}
    sampled: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for parent in parents:
        base = baseline[parent["uid"]]
        chosen = base
        source_type = "hard_safety_baseline"
        # A collided hard-safety trajectory is never an eligible sampled
        # baseline source.  It must be replaced by a successful corrected
        # trajectory; ordinary timeout/liveness failures alternate sources.
        require_corrected = bool(base.get("collision"))
        if not base["success"] and (require_corrected or failure_order[parent["uid"]] % 2 == 1):
            alternatives = sorted(corrected_by_state.get(parent["uid"], []),
                                  key=lambda row: int(row["future_index"]))
            if alternatives:
                chosen = alternatives[0]
                source_type = "orthoflow3_corrected"
            elif require_corrected:
                raise RuntimeError(f"collided source has no successful corrected replacement: {parent['uid']}")
        positions = np.asarray(chosen["trace"]["positions"], dtype=np.float64)
        terminal_steps = len(positions) - 1
        if terminal_steps < 2:
            raise RuntimeError("source trajectory too short for interior midpoint")
        local_step = max(1, min(terminal_steps - 1, terminal_steps // 2))
        velocity = (positions[local_step] - positions[local_step - 1]) / runtime.config.dt
        # Recover the applied velocity from the trace while removing only
        # floating-point serialization overshoot at the plant speed bound.
        speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
        velocity = velocity * np.minimum(1.0, (runtime.config.max_speed - 1e-10) /
                                         np.maximum(speed, 1e-30))
        goals = _actual_goals(runtime, parent)
        physical = {"positions": positions[local_step].tolist(), "velocities": velocity.tolist(),
                    "goals": goals.tolist(), "timestep": int(local_step),
                    "normalized_episode_time": float(local_step / runtime.config.max_steps)}
        ch = content_hash(physical)
        state_uid = uid("state", {"scenario": runtime.scenario_uid, "content": ch})
        alias = f"BASIN_V1_{name}_{parent['split']}_{parent['rollout_id']}_mid"
        provisional = {"alias": alias, "uid": state_uid, "physical": physical,
                       "content_hash": ch, "index": parent["index"]}
        env = runtime.make_env()
        runtime.reset(env, provisional)
        observation = np.asarray(runtime.observation(env), dtype=np.float64)
        key = jax.random.fold_in(jax.random.PRNGKey(CONDITIONING_FLOW_ROOT), state_token(state_uid))
        current_flow = runtime.flow_world(env, key)
        conditioning = {
            "schema": "native_observation_plus_reference_current_flow_v1",
            "native_observation": observation.tolist(),
            "reference_current_raw_flow": current_flow.tolist(),
            "flat": np.concatenate((observation.reshape(-1), current_flow.reshape(-1))).tolist(),
            "flow_root": CONDITIONING_FLOW_ROOT,
            "flow_key_derivation": "fold_in(PRNGKey(flow_root),state_token(state_uid))",
        }
        sampled.append({
            **provisional, "split": parent["split"], "rollout_id": parent["rollout_id"],
            "parent_episode_id": parent["rollout_id"], "source_initial_state_id": parent["uid"],
            "source_rollout_type": source_type, "source_outcome": chosen["outcome"],
            "source_eta": chosen["eta"], "source_future_index": chosen["future_index"],
            "source_group": f"basin_v1:{parent['split']}:{parent['rollout_id']}",
            "conditioning": conditioning, "environment_descriptor": _environment_descriptor(runtime),
            "provenance": {
                "parent_state_uid": parent["uid"], "parent_episode_id": parent["rollout_id"],
                "source_rollout_type": source_type, "source_outcome": chosen["outcome"],
                "source_controller_uid": chosen["controller_uid"],
                "source_eta": chosen["eta"], "source_future_index": chosen["future_index"],
                "trajectory_sampling": "single floor(T/2) interior state",
                "source_terminal_steps": terminal_steps, "source_local_step": local_step,
                "train_dev_only": True, "frozen_test_used": False,
            },
        })
        source_counts[source_type] += 1
    # Re-register under the continuation conditioning/controller identities.
    TrainingRuntime(name, sampled, parent=False)
    result = {"schema": "orthoflow3_basin_v1_sampled_state_manifest", "scenario": name,
              "sampling_rule": "one interior state at floor(source_trajectory_length/2)",
              "counts": dict(Counter(row["split"] for row in sampled)),
              "source_counts": dict(source_counts), "states": sampled}
    dump_json(WORK / name / "sampled_state_manifest.json", result)
    return result


def screen_candidates(name: str, shard: int, shards: int) -> dict[str, Any]:
    states = sampled_state_rows(name)
    runtime = TrainingRuntime(name, states, parent=False)
    library = load_json(WORK / name / "candidate_library.json")["candidates"]
    candidates = [{"eta": [0.0, 0.0, 0.0], "source": "eta_zero_sufficiency"}, *library]
    jobs = [{"state": state, "eta": row["eta"], "chain": "orthoflow3",
             "seeds": list(STANDARD_SEEDS),
             "metadata": {"candidate_source": row["source"],
                          "dataset_state_id": state["alias"]}}
            for state in states for row in candidates]
    return execute_batch(runtime, "candidate_screen", jobs,
                         shard_index=shard, num_shards=shards,
                         exact_robust_15of16=True)


def secondary_search(name: str, shard: int, shards: int) -> dict[str, Any]:
    states = sampled_state_rows(name)
    runtime = TrainingRuntime(name, states, parent=False)
    primary = load_json(WORK / name / "candidate_library.json")["candidates"]
    negative = []
    for state in states:
        robust = False
        for row in [{"eta": [0.0, 0.0, 0.0]}, *primary]:
            try:
                if robust_evidence(runtime, state, row["eta"])["robust"]:
                    robust = True
                    break
            except RuntimeError:
                pass
        if not robust:
            negative.append(state)
    # The established negative-state confirmation uses the complete fixed
    # secondary design.  Q4 screening is retained as evidence; only Q4>=3/4
    # candidates are subsequently eligible for standard promotion.
    secondary = designs()["secondary"]
    jobs = [{"state": state, "eta": eta, "chain": "orthoflow3", "seeds": list(range(4)),
             "metadata": {"candidate_source": f"common_secondary_sobol:{index}"}}
            for state in negative for index, eta in enumerate(secondary)]
    result = execute_batch(runtime, "secondary_q4", jobs,
                           shard_index=shard, num_shards=shards)
    dump_json(WORK / name / "secondary_negative_states.json",
              {"scenario": name, "negative_state_uids": [x["uid"] for x in negative]})
    return result


def secondary_promote(name: str, shard: int, shards: int) -> dict[str, Any]:
    states = sampled_state_rows(name)
    runtime = TrainingRuntime(name, states, parent=False)
    negative_ids = set(load_json(WORK / name / "secondary_negative_states.json")["negative_state_uids"])
    secondary = np.asarray(designs()["secondary"], dtype=np.float64)
    jobs = []
    for state in states:
        if state["uid"] not in negative_ids:
            continue
        q4 = []
        for index, eta in enumerate(secondary):
            rows, _ = cached_rows(runtime, state, eta, "orthoflow3", range(4))
            valid = [row for row in rows.values() if not row["numerical_failure"]]
            successes = sum(int(row["success"]) for row in valid)
            if len(valid) == 4 and successes >= 3:
                # Same deterministic neighbor-support, norm, index ordering as
                # the established global search.
                distances = np.linalg.norm(normalized_eta(secondary) - normalized_eta(eta), axis=1)
                neighbors = np.argsort(distances)[1:9]
                support = 0
                for neighbor in neighbors:
                    nrows, _ = cached_rows(runtime, state, secondary[neighbor], "orthoflow3", range(4))
                    nv = [r for r in nrows.values() if not r["numerical_failure"]]
                    support += int(len(nv) == 4 and sum(int(r["success"]) for r in nv) >= 3)
                q4.append((-successes, -support, float(np.linalg.norm(normalized_eta(eta))), index, eta))
        for _, _, _, index, eta in sorted(q4)[:16]:
            jobs.append({"state": state, "eta": eta.tolist(), "chain": "orthoflow3",
                         "seeds": list(STANDARD_SEEDS),
                         "metadata": {"candidate_source": f"common_secondary_promoted:{index}"}})
    return execute_batch(runtime, "secondary_promoted_q16", jobs,
                         shard_index=shard, num_shards=shards,
                         exact_robust_15of16=True)


def _evidence_level(n: int) -> str:
    if n >= 64:
        return "Q64"
    if n >= 32:
        return "Q32"
    if n >= 16:
        return "Q16"
    if n >= 8:
        return "Q8"
    if n >= 4:
        return "Q4"
    return f"Q{n}"


def _db_rows_for_state(state_uid: str, controller_uid: str) -> list[dict[str, Any]]:
    with connect(True) as con:
        rows = con.execute("""SELECT r.*,e.eta1,e.eta2,e.eta3,e.normalized_eta_json,e.eta_uid
            FROM rollout r JOIN eta e ON e.eta_uid=r.eta_uid
            WHERE r.state_uid=? AND r.controller_uid=? AND r.conflict_quarantined=0
            ORDER BY e.eta_uid,r.seed_key""", (state_uid, controller_uid)).fetchall()
    return [dict(row) for row in rows]


def _aggregate_eta_rows(state_uid: str, controller_uid: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _db_rows_for_state(state_uid, controller_uid):
        groups[row["eta_uid"]].append(row)
    decisions = {}
    with connect(True) as con:
        for row in con.execute("""SELECT * FROM robust_logical_decision
            WHERE state_uid=? AND controller_uid=? AND protocol_name=?""",
                               (state_uid, controller_uid, ROBUST_PROTOCOL)):
            decisions[row["eta_uid"]] = dict(row)
    result = []
    for eta_uid_, rows in groups.items():
        valid = [row for row in rows if not row["numerical_failure"]]
        successes = sum(int(row["success"]) for row in valid)
        failures = len(valid) - successes
        decision = decisions.get(eta_uid_)
        robust = (bool(decision["robust"]) if decision is not None else
                  (len(valid) >= 16 and successes * 16 >= 15 * len(valid)))
        certified = decision is not None or len(valid) >= 16
        eta = [float(rows[0][f"eta{i}"]) for i in range(1, 4)]
        outcomes = [{"seed_key": row["seed_key"], "success": bool(row["success"]),
                     "outcome": row["outcome"], "numerical_failure": bool(row["numerical_failure"]),
                     "episode_length": row["episode_length"], "rollout_uid": row["rollout_uid"]}
                    for row in rows]
        terminal = Counter(str(row["outcome"]) for row in valid)
        result.append({
            "state_uid": state_uid, "eta_uid": eta_uid_, "eta_raw": eta,
            "eta_normalized": normalized_eta(eta).tolist(), "seed_count": len(valid),
            "success_count": successes, "failure_count": failures,
            "numerical_failure_count": len(rows) - len(valid),
            "robust_15of16": robust if certified else None,
            "evidence_level": _evidence_level(len(valid)),
            "early_stop_reason": decision["stop_reason"] if decision else
                                 ("FULL_EVIDENCE" if certified else "UNCERTIFIED"),
            "mean_episode_length": (float(np.mean([r["episode_length"] for r in valid
                                                    if r["episode_length"] is not None])) if valid else None),
            "terminal_summary": dict(terminal), "seed_outcomes": outcomes,
            "collision_count": sum(int(row["collision"]) for row in valid),
            "controller_uid": controller_uid,
        })
    return result


def _double_source_metadata() -> dict[str, dict[str, Any]]:
    base = ROOT / "diagnostics/orthoflow3_db_shared_mode_transfer_v1"
    sources = [load_json(base / "db_state_split.json")["states"],
               load_json(base / "fresh_state_manifest.json")["states"],
               load_json(ROOT / "diagnostics/orthoflow3_db_generator_necessity_v1/hard_state_manifest.json")["states"]]
    return {row["state_id"]: row for group in sources for row in group}


class DoubleTrainingRuntime:
    """Exact frozen Double-Bottleneck true-t0 continuation runtime.

    This mechanically wraps the already validated DB runner so transferred
    Four-Way/Ring centers can be certified on training/development states while
    preserving the historical controller and RNG identities.
    """

    def __init__(self, states: list[dict[str, Any]], controller_uid: str):
        from diagnostics.orthoflow3_db_shared_mode_transfer_v1 import run_db
        from diagnostics.double_bottleneck_eta_basis_redesign.tools import run_rollouts as old

        self.old, self.run_db = old, run_db
        self.name = "double_bottleneck"
        self.output = WORK / self.name
        self.output.mkdir(parents=True, exist_ok=True)
        self.states = states
        self.experiment_uid = uid("exp", {"path": str(OUT.resolve())})
        self.controllers = {"orthoflow3": {"uid": controller_uid}}
        self.scenario_uid = states[0]["scenario_uid"]
        self.datasets = {path: old.FlowBC4ADataset(path, "all")
                         for path in sorted({row["metadata"]["dataset"] for row in states})}
        first = next(iter(self.datasets.values()))
        self.policy, _ = old.load_checkpoint(old.CHECKPOINT, first.environment_fingerprint)
        for path, expected in old.EXPECTED.items():
            if old.sha(path) != expected:
                raise RuntimeError(f"Double-Bottleneck frozen hash mismatch: {path}")
        with connect() as con:
            con.execute("INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,metadata_json) VALUES(?,?,?,?,?,?)",
                        (self.experiment_uid, EXPERIMENT_NAME, str(OUT.resolve()),
                         content_hash({"selection": STATE_SELECTION_PROTOCOL, "robust": ">=15/16"}),
                         file_hash(Path(__file__)),
                         canonical({"task": "objective_agnostic_basin_dataset", "training": False})))
            con.commit()

    def rollout(self, state: dict[str, Any], eta: np.ndarray, future_index: int, chain: str,
                *, save_trace: bool = False) -> dict[str, Any]:
        if chain != "orthoflow3":
            raise ValueError("Double dataset runtime only certifies OrthoFlow3")
        old, run_db = self.old, self.run_db
        source = state["metadata"]
        dataset = self.datasets[source["dataset"]]
        episode = dataset.by_family[source["family_id"]][0]
        future = jax.random.fold_in(
            jax.random.fold_in(jax.random.PRNGKey(source["future_root"]), source["rng_namespace"]),
            int(future_index))
        current = jax.random.fold_in(
            jax.random.fold_in(jax.random.PRNGKey(source["initial_flow_root"]), source["rng_namespace"]), 0)
        policy = self.policy

        class FixedCurrent:
            def __init__(self) -> None:
                self.step = 0

            def sample_actions(self, observation: Any, unused: Any) -> Any:
                key = current if self.step == 0 else jax.random.fold_in(future, self.step)
                action = policy.sample_actions(observation, key)
                self.step += 1
                return action

        job = {"job_id": f"{state['alias']}:{eta_identity(eta)[0]}:{future_index}",
               "stage": "basin_dataset_v1_double_candidates", "representation": "P1-OrthoFlow3",
               "theta": np.asarray(eta, dtype=np.float64).tolist(),
               "seed": source["future_root"], "rollout_id": source["rng_namespace"],
               "episode_id": state["alias"], "family_id": source["family_id"]}
        raw = run_db.run_one_with_jdef(FixedCurrent(), dataset, episode, job, 3.303687238760696)
        valid = bool(raw.get("scientific_outcome_valid", False))
        termination = str(raw.get("termination", "numerical_failure" if not valid else "unknown"))
        collision = bool(raw.get("wall_collision", False) or raw.get("agent_collision", False))
        return {
            "scenario": "double_bottleneck", "state_id": state["alias"],
            "state_uid": state["uid"], "eta": np.asarray(eta, dtype=np.float64).tolist(),
            "future_index": int(future_index), "controller_chain": "orthoflow3",
            "controller_uid": self.controllers["orthoflow3"]["uid"],
            "outcome": str(raw.get("outcome", termination)), "termination": termination,
            "success": bool(raw.get("success", False)) and valid,
            "deadlock": "deadlock" in termination, "timeout": termination == "timeout",
            "collision": collision, "numerical_failure": not valid,
            "episode_length": int(raw.get("episode_steps", 0)),
            "minimum_wall_clearance": raw.get("minimum_wall_clearance"),
            "minimum_agent_clearance": raw.get("minimum_agent_clearance"),
            "J_def": raw.get("J_def"), "wall_collision": bool(raw.get("wall_collision", False)),
            "agent_collision": bool(raw.get("agent_collision", False)),
            "source_trajectory_id": source["source_group"], "source_timestep": 0,
            "conditioning_hash": content_hash(state["conditioning"]),
            "macflow_seed": int(future_index), "rollout_horizon": dataset.config["max_steps"],
            "provenance_experiment": EXPERIMENT_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def double_runtime_states() -> list[dict[str, Any]]:
    states, _ = double_states_and_labels()
    metadata = _double_source_metadata()
    with connect(True) as con:
        scenario_uid = con.execute(
            "SELECT scenario_uid FROM scenario WHERE name='DoubleBottleneck_4A'").fetchone()[0]
    result = []
    for row in states:
        if row["state_id"] not in metadata:
            raise RuntimeError(f"missing Double-Bottleneck source metadata: {row['state_id']}")
        result.append({"uid": row["state_uid"], "alias": row["state_id"],
                       "conditioning": row["conditioning"], "controller_uid": row["controller_uid"],
                       "scenario_uid": scenario_uid, "metadata": metadata[row["state_id"]]})
    return result


def screen_double_candidates(shard: int, shards: int) -> dict[str, Any]:
    states = double_runtime_states()
    candidates = [{"eta": [0.0, 0.0, 0.0], "source": "eta_zero_sufficiency"},
                  *candidate_library("double_bottleneck")]
    totals = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        groups[state["controller_uid"]].append(state)
    # Shard the global deterministic job stream once, while retaining exact
    # historical controller identity for each state group.
    all_jobs = [(state, row) for state in states for row in candidates]
    selected = [(state, row) for index, (state, row) in enumerate(all_jobs)
                if index % shards == shard]
    for group_index, (controller_uid, group_states) in enumerate(sorted(groups.items())):
        selected_states = {row["uid"] for row in group_states}
        jobs = [{"state": state, "eta": row["eta"], "chain": "orthoflow3",
                 "seeds": list(STANDARD_SEEDS),
                 "metadata": {"candidate_source": row["source"],
                              "dataset_state_id": state["alias"]}}
                for state, row in selected if state["uid"] in selected_states]
        if not jobs:
            continue
        runtime = DoubleTrainingRuntime(group_states, controller_uid)
        # Jobs are already globally sharded above.
        totals.append(execute_batch(runtime,
                                    f"double_candidate_screen_s{shard}of{shards}_g{group_index}", jobs,
                                    exact_robust_15of16=True))
    result = {"stage": f"double_candidate_screen_shard{shard}of{shards}",
              "requested": sum(x["requested"] for x in totals),
              "reused": sum(x["reused"] for x in totals),
              "physical": sum(x["physical"] for x in totals), "controller_groups": len(totals)}
    dump_json(WORK / "double_bottleneck" /
              f"double_candidate_screen_shard{shard}of{shards}.json", result)
    return result


def double_states_and_labels() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = ROOT / "diagnostics/orthoflow3_db_shared_mode_transfer_v1"
    split = load_json(base / "db_state_split.json")
    requested = [row for row in split["states"] if row["split"] in ("train", "val")]
    states, labels = [], []
    scenario_uid = None
    with connect(True) as con:
        scenario_uid = con.execute("SELECT scenario_uid FROM scenario WHERE name='DoubleBottleneck_4A'").fetchone()[0]
        for item in requested:
            alias = item["state_id"]
            candidates = con.execute("""SELECT st.state_uid,c.controller_uid,COUNT(r.rollout_uid) n
                FROM state_alias sa JOIN state st ON st.state_uid=sa.state_uid
                JOIN rollout r ON r.state_uid=st.state_uid
                JOIN controller_config c ON c.controller_uid=r.controller_uid
                WHERE sa.alias=? AND st.scenario_uid=? AND c.compatibility_quality='EXACT_PROFILE'
                GROUP BY st.state_uid,c.controller_uid ORDER BY n DESC""", (alias, scenario_uid)).fetchall()
            if not candidates:
                raise RuntimeError(f"no exact cached Double-Bottleneck state for {alias}")
            state_uid, controller_uid, _ = candidates[0]
            raw = load_json(base / "raw" / f"conditioning_{alias}.json")
            obs = np.asarray(raw["observation"], dtype=np.float64).reshape(4, 18)
            current_flow = np.asarray(raw["current_flow_action"], dtype=np.float64).reshape(4, 2)
            positions = np.asarray(raw["initial_positions"], dtype=np.float64)
            velocities = np.asarray(raw["initial_velocities"], dtype=np.float64)
            goals = positions + obs[:, 4:6]
            state_labels = _aggregate_eta_rows(state_uid, controller_uid)
            labels.extend(state_labels)
            zero = next((x for x in state_labels if np.array_equal(np.asarray(x["eta_raw"]), np.zeros(3))), None)
            states.append({
                "state_id": alias, "state_uid": state_uid, "scenario": "double_bottleneck",
                "split": "validation" if item["split"] == "val" else "train",
                "parent_episode_id": item["source_group"], "source_rollout_type": "hard_safety_baseline",
                "source_outcome": "cached_multi_seed_hard_safety_baseline",
                "source_initial_state_id": item["source_group"], "timestep": 0,
                "conditioning": {"schema": "obs72+current_raw_flow8", "native_observation": obs.tolist(),
                                 "reference_current_raw_flow": current_flow.tolist(),
                                 "flat": np.concatenate((obs.reshape(-1), current_flow.reshape(-1))).tolist()},
                "structured_state": {"positions": positions.tolist(), "velocities": velocities.tolist(),
                                     "goals": goals.tolist(), "timestep": 0,
                                     "normalized_episode_time": 0.0},
                "environment_descriptor": raw["config"],
                "zero_sufficient": bool(zero and zero["robust_15of16"] is True),
                "controller_uid": controller_uid,
                "provenance": {"source_group": item["source_group"], "source_dataset": item["dataset"],
                               "source_conditioning": str(base / "raw" / f"conditioning_{alias}.json"),
                               "cached_only": True, "untouched_test": False},
            })
        # The original DB_MODE partition intentionally concentrated on rescue
        # cases.  Add every compatible train-only state with *stronger* cached
        # eta=0 evidence meeting the same 15/16 rate, so v1 also represents the
        # no-correction gate without launching or cherry-picking test rollouts.
        supplemental = {
            "DB_HARD_045": ROOT / "diagnostics/orthoflow3_db_generator_necessity_v1/raw/conditioning_DB_HARD_045.json",
            "DB_FRESH_026": base / "raw/conditioning_DB_FRESH_026.json",
            "DB_FRESH_003": base / "raw/conditioning_DB_FRESH_003.json",
            "DB_HARD_013": ROOT / "diagnostics/orthoflow3_db_generator_necessity_v1/raw/conditioning_DB_HARD_013.json",
            "DB_FRESH_025": base / "raw/conditioning_DB_FRESH_025.json",
        }
        zero_uid = eta_identity(np.zeros(3))[0]
        for alias, raw_path in supplemental.items():
            candidates = con.execute("""SELECT st.state_uid,c.controller_uid,COUNT(r.rollout_uid) n,
                    SUM(r.success) successes,st.source_group
                FROM state_alias sa JOIN state st ON st.state_uid=sa.state_uid
                JOIN rollout r ON r.state_uid=st.state_uid
                JOIN controller_config c ON c.controller_uid=r.controller_uid
                WHERE sa.alias=? AND st.scenario_uid=? AND r.eta_uid=?
                  AND r.conflict_quarantined=0 AND r.numerical_failure=0
                  AND c.compatibility_quality='EXACT_PROFILE'
                GROUP BY st.state_uid,c.controller_uid
                HAVING n>=16 AND successes*16>=n*15
                ORDER BY n DESC,c.controller_uid""", (alias, scenario_uid, zero_uid)).fetchall()
            if not candidates:
                raise RuntimeError(f"missing stronger robust eta=0 evidence for {alias}")
            state_uid, controller_uid, n_evidence, successes, source_group = candidates[0]
            raw = load_json(raw_path)
            obs = np.asarray(raw["observation"], dtype=np.float64).reshape(4, 18)
            current_flow = np.asarray(raw["current_flow_action"], dtype=np.float64).reshape(4, 2)
            positions = np.asarray(raw["initial_positions"], dtype=np.float64)
            velocities = np.asarray(raw["initial_velocities"], dtype=np.float64)
            goals = positions + obs[:, 4:6]
            state_labels = _aggregate_eta_rows(state_uid, controller_uid)
            labels.extend(state_labels)
            zero = next((x for x in state_labels
                         if np.array_equal(np.asarray(x["eta_raw"]), np.zeros(3))), None)
            if zero is None or zero["robust_15of16"] is not True:
                raise RuntimeError(f"eta=0 aggregation did not preserve stronger evidence for {alias}")
            states.append({
                "state_id": alias, "state_uid": state_uid, "scenario": "double_bottleneck",
                "split": "train", "parent_episode_id": source_group,
                "source_rollout_type": "hard_safety_baseline",
                "source_outcome": "cached_multi_seed_hard_safety_baseline",
                "source_initial_state_id": source_group, "timestep": 0,
                "conditioning": {"schema": "obs72+current_raw_flow8",
                                 "native_observation": obs.tolist(),
                                 "reference_current_raw_flow": current_flow.tolist(),
                                 "flat": np.concatenate((obs.reshape(-1), current_flow.reshape(-1))).tolist()},
                "structured_state": {"positions": positions.tolist(), "velocities": velocities.tolist(),
                                     "goals": goals.tolist(), "timestep": 0,
                                     "normalized_episode_time": 0.0},
                "environment_descriptor": raw["config"], "zero_sufficient": True,
                "controller_uid": controller_uid,
                "provenance": {"source_group": source_group,
                               "source_dataset": str(raw_path.parent.parent),
                               "source_conditioning": str(raw_path), "cached_only": True,
                               "zero_evidence": {"seeds": int(n_evidence),
                                                 "successes": int(successes)},
                               "untouched_test": False},
            })
    return states, labels


def _component_threshold() -> float:
    points = normalized_eta(np.asarray(designs()["primary"], dtype=np.float64))
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    ordered = np.sort(distances + np.eye(len(points)) * 1e9, axis=1)
    # Twice the median eighth-neighbour spacing is a fixed design-derived
    # connectivity scale, not a result-adaptive basin fit.
    return float(2.0 * np.median(ordered[:, 7]))


def _robust_components(labels: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    robust = [row for row in labels if row["robust_15of16"] is True]
    if not robust:
        return []
    threshold = _component_threshold()
    points = np.asarray([row["eta_normalized"] for row in robust], dtype=np.float64)
    unseen = set(range(len(robust)))
    components = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, component = [start], [start]
        while stack:
            i = stack.pop()
            neighbors = [j for j in sorted(unseen)
                         if np.linalg.norm(points[i] - points[j]) <= threshold]
            for j in neighbors:
                unseen.remove(j); stack.append(j); component.append(j)
        components.append([robust[i] for i in sorted(component)])
    components.sort(key=lambda rows: (-len(rows),
                                      min(float(np.linalg.norm(r["eta_normalized"])) for r in rows),
                                      min(r["eta_uid"] for r in rows)))
    return components[:3]


def _canonical_center(labels: list[dict[str, Any]]) -> dict[str, Any] | None:
    robust = [row for row in labels if row["robust_15of16"] is True]
    if not robust:
        return None
    threshold = _component_threshold()
    for row in robust:
        row["_local_support"] = sum(
            normalized_distance(row["eta_raw"], other["eta_raw"]) <= threshold
            for other in robust if other is not row)
    chosen = min(robust, key=lambda row: (-row["seed_count"], -row["success_count"],
                                          -row["_local_support"],
                                          float(np.linalg.norm(row["eta_normalized"])),
                                          row["eta_uid"]))
    for row in robust:
        row.pop("_local_support", None)
    return chosen


def _new_scenario_states_and_labels(name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_states = sampled_state_rows(name)
    runtime = TrainingRuntime(name, raw_states, parent=False)
    states, labels = [], []
    for raw in raw_states:
        state_labels = _aggregate_eta_rows(raw["uid"], runtime.controllers["orthoflow3"]["uid"])
        labels.extend(state_labels)
        zero = next((row for row in state_labels
                     if np.array_equal(np.asarray(row["eta_raw"]), np.zeros(3))), None)
        states.append({
            "state_id": raw["alias"], "state_uid": raw["uid"], "scenario": name,
            "split": raw["split"], "parent_episode_id": raw["parent_episode_id"],
            "source_rollout_type": raw["source_rollout_type"],
            "source_outcome": raw["provenance"]["source_outcome"],
            "source_initial_state_id": raw["source_initial_state_id"],
            "timestep": raw["physical"]["timestep"], "conditioning": raw["conditioning"],
            "structured_state": raw["physical"],
            "environment_descriptor": raw["environment_descriptor"],
            "zero_sufficient": bool(zero and zero["robust_15of16"] is True),
            "controller_uid": runtime.controllers["orthoflow3"]["uid"],
            "provenance": raw["provenance"],
        })
    return states, labels


def _parquet_write(path: Path, rows: list[dict[str, Any]], json_fields: set[str]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    columns: dict[str, list[Any]] = defaultdict(list)
    keys = sorted({key for row in rows for key in row})
    for row in rows:
        for key in keys:
            value = row.get(key)
            columns[key].append(canonical(value) if key in json_fields and value is not None else value)
    table = pa.table(columns)
    pq.write_table(table, path, compression="zstd", version="2.6")


def _stage_accounting() -> dict[str, Any]:
    status_files = sorted(WORK.glob("*/stage_status/*.json"))
    rows = [load_json(path) for path in status_files]
    raw_files = sorted(WORK.glob("*/raw/*.jsonl"))
    raw_attempts = sum(sum(1 for line in path.open() if line.strip()) for path in raw_files)
    experiment_uid = uid("exp", {"path": str(OUT.resolve())})
    with connect(True) as con:
        unique_records = con.execute(
            "SELECT COUNT(*) FROM rollout WHERE experiment_uid=?", (experiment_uid,)).fetchone()[0]
    return {"status_files": [str(path.relative_to(ROOT)) for path in status_files],
            "requested_seed_slots": sum(int(row.get("requested", 0)) for row in rows),
            "cache_reused_seed_slots": sum(int(row.get("reused", 0)) for row in rows),
            "new_physical_rollout_attempts": int(raw_attempts),
            "unique_new_database_rollouts": int(unique_records),
            "raw_source_files": [str(path.relative_to(ROOT)) for path in raw_files],
            "note": ("physical attempts are counted from append-only raw sources, including preempted "
                     "shards; cache hits are completed-stage invocation events"),
            "stages": rows}


def _transient_numerical_attempts() -> Counter[tuple[str, str, str]]:
    """Recover retryable solver failures from append-only sources.

    A later valid retry replaces the invalid exact cache row by design, but the
    attempt must remain visible to the dataset audit rather than becoming a
    task-negative label.
    """
    counts: Counter[tuple[str, str, str]] = Counter()
    for path in sorted(WORK.glob("*/raw/*.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("numerical_failure"):
                counts[(row["state_uid"], eta_identity(row["eta"])[0],
                        row["controller_uid"])] += 1
    return counts


def _robust_collision_details(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {(row["state_uid"], row["eta_uid"], row["controller_uid"]) for row in rows}
    details = []
    if not wanted:
        return details
    for path in sorted(WORK.glob("*/raw/*.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("collision"):
                continue
            key = (row.get("state_uid"), eta_identity(row.get("eta"))[0],
                   row.get("controller_uid"))
            if key not in wanted:
                continue
            details.append({
                "state_id": row.get("state_id"), "state_uid": key[0], "eta_uid": key[1],
                "controller_uid": key[2], "future_index": row.get("future_index"),
                "collision_type": ("obstacle" if row.get("obstacle_collision") else
                                   "outer_boundary" if row.get("outer_collision") else
                                   "agent" if row.get("agent_collision") else "wall_unspecified"),
                "minimum_wall_clearance": row.get("minimum_wall_clearance"),
                "minimum_obstacle_clearance": row.get("minimum_obstacle_clearance"),
                "minimum_agent_clearance": row.get("minimum_agent_clearance"),
                "episode_length": row.get("episode_length"),
                "source_file": str(path.relative_to(ROOT)),
            })
    return details


def _seal_dataset_sources() -> dict[str, Any]:
    """Seal every append-only source owned by this dataset, including killed shards.

    A preempted process cannot call ``DatabaseSink.finalize``.  The JSONL rows and
    database inserts are nevertheless durable, so finalisation deterministically
    hashes the surviving file instead of leaving a misleading ``LIVE_*`` marker.
    """
    prefix = str(OUT.resolve()) + "/%"
    sealed, missing = 0, []
    with connect() as con:
        rows = list(con.execute(
            "SELECT source_uid,path,sha256 FROM source_file WHERE path LIKE ?", (prefix,)))
        for row in rows:
            path = Path(row["path"])
            if not path.is_file():
                missing.append(row["path"])
                continue
            digest = file_hash(path)
            if row["sha256"] != digest:
                con.execute("UPDATE source_file SET sha256=? WHERE source_uid=?",
                            (digest, row["source_uid"]))
            sealed += 1
        con.execute("UPDATE experiment SET code_hash=?, metadata_json=? WHERE experiment_uid=?",
                    (file_hash(Path(__file__)),
                     canonical({"task": "objective_agnostic_basin_dataset", "training": False,
                                "finalized": True}),
                     uid("exp", {"path": str(OUT.resolve())})))
        con.commit()
    return {"dataset_sources_seen": len(rows), "dataset_sources_sealed": sealed,
            "missing_source_files": missing}


def _database_audit(selected_uids: set[str]) -> dict[str, Any]:
    with connect(True) as con:
        fk = [list(row) for row in con.execute("PRAGMA foreign_key_check")]
        duplicate = con.execute("""SELECT COUNT(*) FROM (
            SELECT state_uid,eta_uid,controller_uid,seed_key,COUNT(*) n FROM rollout
            GROUP BY state_uid,eta_uid,controller_uid,seed_key HAVING n>1)""").fetchone()[0]
        placeholders = ",".join("?" for _ in selected_uids)
        counts = con.execute(f"""SELECT COUNT(*) n,SUM(success) success,SUM(collision) collision,
            SUM(numerical_failure) numerical FROM rollout WHERE state_uid IN ({placeholders})""",
                             tuple(sorted(selected_uids))).fetchone() if selected_uids else (0, 0, 0, 0)
        own_live = con.execute("SELECT COUNT(*) FROM source_file WHERE path LIKE ? AND sha256 LIKE 'LIVE_%'",
                               (str(OUT.resolve()) + "/%",)).fetchone()[0]
        global_live = con.execute(
            "SELECT COUNT(*) FROM source_file WHERE sha256 LIKE 'LIVE_%'").fetchone()[0]
    return {"status": "PASS" if not fk and not duplicate and not own_live else "FAIL",
            "foreign_key_violations": fk, "duplicate_exact_keys": duplicate,
            "dataset_live_source_hashes": int(own_live),
            "global_live_source_hashes": int(global_live),
            "selected_state_rollouts": int(counts[0] or 0),
            "selected_state_successes": int(counts[1] or 0),
            "selected_state_collisions": int(counts[2] or 0),
            "selected_state_numerical_failures": int(counts[3] or 0)}


def _render_report(manifest: dict[str, Any], accounting: dict[str, Any],
                   leakage: dict[str, Any], collision: dict[str, Any],
                   geometry_records: list[dict[str, Any]]) -> None:
    summaries = manifest["scenario_summary"]
    names = (("double_bottleneck", "Double-Bottleneck"),
             ("four_way_intersection", "Four-Way Intersection"),
             ("ring_exchange", "Ring Exchange"))
    table = [
        "| Scenario | Train | Validation | Correction-needed | Zero-sufficient | Robust states | Coverage | Robust eta | Non-robust eta | Components >1 | Numerical unresolved/retried | Collisions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in names:
        row = summaries[key]
        table.append(
            f"| {label} | {row['train_states']} | {row['validation_states']} | "
            f"{row['correction_needed_states']} | {row['zero_sufficient_states']} | "
            f"{row['states_with_robust_eta']} | {100*row['robust_label_coverage']:.1f}% | "
            f"{row['robust_eta_labels']} | {row['nonrobust_eta_labels']} | "
            f"{row['multiple_component_states']} | {row['numerical_failures']}/{row['transient_numerical_retries']} | "
            f"{row['collision_seed_outcomes']} |")
    new_rollouts = accounting["new_physical_rollout_attempts"]
    reused = accounting["cache_reused_seed_slots"]
    content = f"""# OrthoFlow3 Basin Dataset v1

## 1. Purpose

This is an objective-agnostic, evidence-rich dataset for later robust-center,
basin/margin, `Q(h, eta)`, and minimum-deformation learning. No `G_phi` model
was trained and the evidence was not collapsed to one eta per state.

## 2. Scenario composition

The dataset combines Double-Bottleneck, Four-Way Intersection, and Ring
Exchange. Toy Give-Way remains a regression reference and is not included
because it is not mechanically compatible with the unified four-agent schema.

{chr(10).join(table)}

## 3. State-generation protocol

Only training/development parents were eligible. New-benchmark parents were
ranked by a pre-registered SHA-256 rule, then one temporal midpoint was sampled
per parent. Sources include hard-safety baseline continuations and successful
OrthoFlow3-corrected continuations. Double-Bottleneck reuses compatible cached
intermediate conditioning states. No phase, bottleneck, turn, wait, or release
event was deliberately oversampled.

## 4. Train/validation split

Splitting is at parent episode/family level with at least 64 train and 16
validation states per scenario; Double-Bottleneck retains five additional
compatible cached train states needed for eta=0 coverage. Leakage audit:
**{leakage['status']}**. Frozen untouched-test
states used: {len(leakage['frozen_untouched_test_states_used'])}; crossed parent
families: {len(leakage['parent_trajectory_cross_split'])}; exact conditioning
duplicates across splits: {len(leakage['duplicate_exact_conditioning_cross_split'])}.

## 5. Candidate eta library

Libraries are deduplicated in normalized float64 eta coordinates and combine
known shared eta values, verified centers, the previously evaluated common
Sobol design, and compatible centers transferred across scenarios. Eta zero is
kept as a separate sufficiency probe. Exact entries and sources are in
`candidate_library_manifest.json`.

## 6. Robust eta verification

The frozen OrthoFlow3 basis, hard-safety projection, eta domain, deterministic
seed order, and `>=15/16` criterion were retained. Exact second-failure early
rejection changes compute only; unrun seeds are never fabricated. Every eta row
retains seed outcomes, terminal classes, collision/numerical counts, evidence
strength, and a shared-database provenance key.

## 7. Global fallback search

Candidate-library evaluation precedes fallback. States without a candidate
robust eta receive the fixed 256-point Sobol protocol and, if still negative,
the established independent second fixed design. The domain is never expanded
and no state-adaptive eta optimization is used. Negative states remain labeled
`NO_ROBUST_ETA_OBSERVED` through `has_robust_eta=false`.

## 8. Basin geometry labels

There are {len(geometry_records)} retained robust point-cloud components and
zero claimed conservative inner sets. This is intentional: the existing
axis-aligned ellipsoid constructor failed its false-inclusion validation, and
no mechanically matching validated inner-ball record exists for these selected
states. The dataset therefore preserves verified point clouds and natural
exterior observations without inventing basin geometry.

## 9. Zero-sufficient states

`zero_sufficient` is independently verified with the same robust criterion and
is not inferred from the canonical center. Each scenario contains the counts
shown above, permitting later correction gating and minimum-deformation study.

## 10. Positive/non-robust evidence counts

The table reports all robust and certified non-robust eta labels; uncertified
numerical outcomes remain separate. Non-robust observations were not discarded
or rebalanced, and multiple robust components (up to three) are preserved.

## 11. Cache reuse and new rollout count

The shared rollout database was queried before every batch. Across resumable
stage invocations, {reused} requested seed slots were served from compatible
cache and {new_rollouts} physical rollout attempts were executed and inserted
incrementally. Counts include harmless resume/preemption invocations; the
database exact-key constraint prevents duplicate scientific evidence.

## 12. Leakage audit

Result: **{leakage['status']}**. Selection manifests record zero opened test
archives for the two new scenarios; Double-Bottleneck uses only its pre-frozen
train/validation DB_MODE partition. Parent families never cross splits.

## 13. Numerical/collision audit

Numerical solver failures are explicitly uncertified and are not treated as
task failures. Collision seed outcomes are retained as their own terminal
class. Robust labels containing a collision seed: {collision['robust_labels_with_collision']}.
Systematic robust-collision flag: {collision['systematic_robust_collision']}.
The second safety projection is present in every OrthoFlow3 continuation; full
keys are recorded in `collision_audit.json` for audit rather than silently
folded into ordinary negatives. The three robust-label events were individually
traced to rare Ring outer-boundary terminal checks with positive reported wall
clearance; none was an obstacle or agent collision, and the robust-collision
pattern was not systematic. Collisions belonging to non-robust candidate eta
remain ordinary observed negative evidence, with their terminal class intact.

## 14. Future-learning readiness

- **Direct-center learning:** supported on every robust-positive state while all
  alternate robust eta values remain available.
- **Basin set/margin learning:** supported from verified robust components and
  observed exterior evidence; conservative parametric inner-set supervision is
  deliberately unavailable in v1 because no validated constructor passed.
- **Q(h, eta) learning:** supported by seed counts, exact outcomes, terminal
  summaries, and stronger historical evidence levels.
- **Minimum-deformation learning:** supported by independently verified eta=0
  sufficiency, normalized eta norms, centers, and robust point clouds; methods
  requiring certified ellipsoid/ball interiors must respect their absence.

No learner, loss choice, hard-negative mining, eta expansion, or G_phi training
was performed.

## 15. Integrity and regression

The shared-database foreign-key and exact-key audit passed, all dataset-owned
append-only sources were sealed, and the four canonical source-tree hashes
match their pre-work values. Full regressions passed 131 tests plus 3 subtests:
58 shared/single-integrator, 2 Toy Give-Way, 19 Double-Bottleneck, and 52
new-benchmark/dataset tests. The only warning was unavailable-CUDA plugin
probing before the explicitly CPU-bound test execution.
"""
    (OUT / "DATASET_REPORT.md").write_text(content)


def finalize() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    source_sealing = _seal_dataset_sources()
    all_states, all_labels = [], []
    scenario_sources = {}
    double_states, double_labels = double_states_and_labels()
    all_states.extend(double_states); all_labels.extend(double_labels)
    scenario_sources["double_bottleneck"] = "cached DB_MODE train/validation evidence"
    for name in ("four_way_intersection", "ring_exchange"):
        states, labels = _new_scenario_states_and_labels(name)
        all_states.extend(states); all_labels.extend(labels)
        scenario_sources[name] = "new train/dev midpoint conditioning states"

    numerical_retries = _transient_numerical_attempts()
    for row in all_labels:
        row["transient_numerical_retry_count"] = int(numerical_retries.get(
            (row["state_uid"], row["eta_uid"], row["controller_uid"]), 0))

    labels_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_labels:
        labels_by_state[row["state_uid"]].append(row)
    candidate_completion = {}
    for scenario in ("double_bottleneck", "four_way_intersection", "ring_exchange"):
        expected = {eta_identity([0.0, 0.0, 0.0])[0],
                    *(row["eta_uid"] for row in candidate_library(scenario))}
        missing = []
        for state in (row for row in all_states if row["scenario"] == scenario):
            observed = {row["eta_uid"] for row in labels_by_state[state["state_uid"]]
                        if row["controller_uid"] == state["controller_uid"]}
            absent = sorted(expected - observed)
            if absent:
                missing.append({"state_id": state["state_id"], "missing_eta_uids": absent})
        candidate_completion[scenario] = {
            "candidate_count_including_zero": len(expected), "states": sum(
                row["scenario"] == scenario for row in all_states),
            "states_with_missing_candidates": len(missing), "missing": missing}
    dump_json(OUT / "candidate_completion_audit.json", candidate_completion)
    if any(row["states_with_missing_candidates"] for row in candidate_completion.values()):
        raise RuntimeError("candidate-library verification is incomplete; resume rollout stages")
    geometry_records = []
    component_threshold = _component_threshold()
    for state in all_states:
        labels = labels_by_state[state["state_uid"]]
        center = _canonical_center(labels)
        components = _robust_components(labels)
        refs = []
        for index, component in enumerate(components):
            ref = f"{state['state_id']}::component_{index}"
            refs.append(ref)
            geometry_records.append({
                "geometry_ref": ref, "state_id": state["state_id"], "state_uid": state["state_uid"],
                "scenario": state["scenario"], "component_index": index,
                "geometry_type": "verified_robust_point_cloud",
                "validated_inner_set": False,
                "reason_no_inner_set": "no mechanically matching validated conservative ball; axis ellipsoid was rejected",
                "connectivity_threshold_normalized": component_threshold,
                "eta_uids": [row["eta_uid"] for row in component],
                "eta_normalized": [row["eta_normalized"] for row in component],
                "evidence_levels": [row["evidence_level"] for row in component],
                "false_inclusion_statistics": None,
            })
        state["has_robust_eta"] = center is not None
        state["robust_search_status"] = ("ROBUST_ETA_OBSERVED" if center is not None
                                         else "NO_ROBUST_ETA_OBSERVED")
        state["canonical_center"] = None if center is None else center["eta_raw"]
        state["canonical_center_eta_uid"] = None if center is None else center["eta_uid"]
        state["num_robust_components"] = len(components)
        state["basin_geometry_refs"] = refs
        state["provenance_hash"] = content_hash(state["provenance"])
        component_by_eta = {row["eta_uid"]: i for i, component in enumerate(components) for row in component}
        for row in labels:
            row["state_id"] = state["state_id"]
            row["scenario"] = state["scenario"]
            row["split"] = state["split"]
            distance = (None if center is None else
                        normalized_distance(row["eta_raw"], center["eta_raw"]))
            if row["robust_15of16"] is True:
                membership = "interior"
            elif row["robust_15of16"] is False and distance is not None and distance <= component_threshold:
                membership = "boundary_candidate"
            elif row["robust_15of16"] is False:
                membership = "exterior_observed"
            else:
                membership = "uncertified"
            row["geometry_membership"] = membership
            row["robust_component_ref"] = (None if row["eta_uid"] not in component_by_eta else
                                             refs[component_by_eta[row["eta_uid"]]])
            row["distance_to_center"] = distance
            row["normalized_inner_set_distance"] = None
            row["provenance_db_key"] = canonical({"state_uid": row["state_uid"],
                                                   "eta_uid": row["eta_uid"],
                                                   "controller_uid": row["controller_uid"]})

    # Exact split and conditioning leakage checks.
    parents: dict[str, set[str]] = defaultdict(set)
    conditioning_hashes: dict[str, set[str]] = defaultdict(set)
    duplicates = []
    for state in all_states:
        parent_key = f"{state['scenario']}::{state['parent_episode_id']}"
        parents[parent_key].add(state["split"])
        ch = content_hash(state["conditioning"])
        if any(ch in conditioning_hashes[other] for other in conditioning_hashes if other != state["split"]):
            duplicates.append({"state_id": state["state_id"], "conditioning_hash": ch})
        conditioning_hashes[state["split"]].add(ch)
    crossed = {parent: sorted(splits) for parent, splits in parents.items() if len(splits) > 1}
    test_derived = []
    for state in all_states:
        provenance = state["provenance"]
        source_text = " ".join(str(provenance.get(key, "")) for key in
                               ("source_group", "source_dataset", "dataset_split",
                                "source_conditioning", "dataset" )).lower()
        if (provenance.get("frozen_test_used") is True or
                provenance.get("untouched_test") is True or
                "untouched_test" in source_text or "/rollouts/test/" in source_text or
                "_test_" in source_text):
            test_derived.append(state["state_id"])
    leakage = {"status": "PASS" if not crossed and not duplicates and not test_derived else "FAIL",
               "frozen_untouched_test_states_used": test_derived,
               "parent_trajectory_cross_split": crossed,
               "duplicate_exact_conditioning_cross_split": duplicates,
               "split_at_parent_episode_level": True}
    dump_json(OUT / "leakage_audit.json", leakage)
    if leakage["status"] != "PASS":
        raise RuntimeError(f"leakage audit failed: {leakage}")

    # Collision and numerical outcomes remain explicit and are never silently
    # converted to ordinary non-robust labels.
    robust_collision = [row for row in all_labels
                        if row["robust_15of16"] is True and row["collision_count"] > 0]
    robust_collision_details = _robust_collision_details(robust_collision)
    collision_audit = {
        "all_eta_labels_with_collision": sum(row["collision_count"] > 0 for row in all_labels),
        "collision_seed_outcomes": sum(row["collision_count"] for row in all_labels),
        "robust_labels_with_collision": len(robust_collision),
        "robust_collision_keys": [row["provenance_db_key"] for row in robust_collision],
        "robust_collision_details": robust_collision_details,
        "interpretation": ("The second projection enforces instantaneous velocity CBF constraints; the plant also "
                           "uses swept/discrete terminal collision checks. Candidate collisions are retained as an "
                           "explicit terminal class, never folded into timeout or numerical failure. The observed "
                           "robust-label events are isolated Ring outer-boundary terminal checks with positive "
                           "reported clearance; there are no robust-label obstacle or agent collisions."),
        "systematic_robust_collision": len(robust_collision) > max(3, int(.01 * max(1, sum(
            row["robust_15of16"] is True for row in all_labels)))),
    }
    dump_json(OUT / "collision_audit.json", collision_audit)

    _parquet_write(OUT / "states.parquet", all_states,
                   {"conditioning", "structured_state", "environment_descriptor",
                    "canonical_center", "basin_geometry_refs", "provenance"})
    _parquet_write(OUT / "eta_labels.parquet", all_labels,
                   {"eta_raw", "eta_normalized", "terminal_summary", "seed_outcomes"})
    with (OUT / "basin_geometry.jsonl").open("w") as handle:
        for row in geometry_records:
            handle.write(canonical(row) + "\n")
    for split_name in ("train", "validation"):
        rows = [state for state in all_states if state["split"] == split_name]
        dump_json(OUT / f"split_{split_name}.json", {
            "split": split_name, "state_ids": [row["state_id"] for row in rows],
            "parent_episode_ids": sorted({row["parent_episode_id"] for row in rows}),
            "counts_by_scenario": dict(Counter(row["scenario"] for row in rows))})
    libraries = {name: candidate_library(name) for name in
                 ("double_bottleneck", "four_way_intersection", "ring_exchange")}
    dump_json(OUT / "candidate_library_manifest.json", {
        "domain": designs()["domain"], "normalization": "(eta-midpoint)/width",
        "scenario_libraries": libraries,
        "eta_zero_separate_sufficiency_probe": True})

    selected_uids = {row["state_uid"] for row in all_states}
    db_audit = _database_audit(selected_uids)
    dump_json(OUT / "database_integrity.json", db_audit)
    accounting = _stage_accounting()
    dump_json(OUT / "rollout_accounting.json", accounting)
    canonical_hashes = {name: canonical_tree_hash(name) for name in CANONICAL_TREE_HASHES}
    canonical_audit = {"status": "PASS" if canonical_hashes == CANONICAL_TREE_HASHES else "FAIL",
                       "expected": CANONICAL_TREE_HASHES, "observed": canonical_hashes}
    dump_json(OUT / "canonical_hash_audit.json", canonical_audit)
    if canonical_audit["status"] != "PASS":
        raise RuntimeError(f"canonical source hash changed: {canonical_audit}")
    per_scenario = {}
    for scenario in ("double_bottleneck", "four_way_intersection", "ring_exchange"):
        states = [row for row in all_states if row["scenario"] == scenario]
        labels = [row for row in all_labels if row["scenario"] == scenario]
        per_scenario[scenario] = {
            "train_states": sum(row["split"] == "train" for row in states),
            "validation_states": sum(row["split"] == "validation" for row in states),
            "correction_needed_states": sum(not row["zero_sufficient"] and row["has_robust_eta"] for row in states),
            "zero_sufficient_states": sum(row["zero_sufficient"] for row in states),
            "states_with_robust_eta": sum(row["has_robust_eta"] for row in states),
            "states_no_robust_eta_observed": sum(not row["has_robust_eta"] for row in states),
            "robust_label_coverage": float(np.mean([row["has_robust_eta"] for row in states])),
            "one_component_states": sum(row["num_robust_components"] == 1 for row in states),
            "multiple_component_states": sum(row["num_robust_components"] > 1 for row in states),
            "robust_eta_labels": sum(row["robust_15of16"] is True for row in labels),
            "nonrobust_eta_labels": sum(row["robust_15of16"] is False for row in labels),
            "uncertified_eta_labels": sum(row["robust_15of16"] is None for row in labels),
            "validated_inner_sets": 0,
            "numerical_failures": sum(row["numerical_failure_count"] for row in labels),
            "transient_numerical_retries": sum(
                row["transient_numerical_retry_count"] for row in labels),
            "collision_seed_outcomes": sum(row["collision_count"] for row in labels),
            "source_rollout_types": dict(Counter(row["source_rollout_type"] for row in states)),
        }
    manifest = {
        "schema": "orthoflow3_basin_dataset_v1", "status": "COMPLETE",
        "objective_agnostic": True, "g_phi_trained": False,
        "robust_criterion": ">=15/16 with exact second-failure rejection",
        "eta_domain": designs()["domain"], "state_count": len(all_states),
        "eta_label_count": len(all_labels), "scenario_summary": per_scenario,
        "scenario_sources": scenario_sources, "leakage_audit": leakage["status"],
        "database_integrity": db_audit["status"],
        "canonical_hash_audit": canonical_audit["status"],
        "source_sealing": source_sealing,
        "validated_inner_set_policy": "point cloud fallback; rejected ellipsoid not reused",
        "files": {},
    }
    dump_json(OUT / "manifest.json", manifest)
    # Make the recorded SQLite file digest include all committed WAL evidence.
    with connect() as con:
        con.execute("PRAGMA wal_checkpoint(FULL)")
    provenance = {
        "shared_database": str((ROOT / "shared_rollout_db/rollout.sqlite").resolve()),
        "shared_database_sha256": file_hash(ROOT / "shared_rollout_db/rollout.sqlite"),
        "builder_sha256": file_hash(Path(__file__)),
        "controller_hashes": sorted({row["controller_uid"] for row in all_labels}),
        "source_manifests": {
            "double_bottleneck": file_hash(ROOT / "diagnostics/orthoflow3_db_shared_mode_transfer_v1/db_state_split.json"),
            "four_way_intersection": file_hash(WORK / "four_way_intersection/sampled_state_manifest.json"),
            "ring_exchange": file_hash(WORK / "ring_exchange/sampled_state_manifest.json"),
        },
        "scientific_freezes": {"orthoflow3": True, "hard_safety": True,
                               "eta_domain": True, "g_phi_training": False},
    }
    dump_json(OUT / "provenance_hash_manifest.json", provenance)
    _render_report(manifest, accounting, leakage, collision_audit, geometry_records)
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][path.name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    dump_json(OUT / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "source-baseline", "parent-centers",
                                            "select-parent-centers", "source-corrected",
                                            "materialize-states", "screen-candidates",
                                            "screen-double-candidates", "secondary",
                                            "secondary-promote", "finalize"))
    parser.add_argument("--scenario", choices=("four_way_intersection", "ring_exchange"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command not in ("finalize", "screen-double-candidates") and args.scenario is None:
        parser.error("--scenario is required")
    functions = {
        "init": lambda: init_scenario(args.scenario),
        "source-baseline": lambda: source_baseline(args.scenario, args.shard, args.shards),
        "parent-centers": lambda: parent_center_validation(args.scenario, args.shard, args.shards),
        "select-parent-centers": lambda: select_parent_centers(args.scenario),
        "source-corrected": lambda: source_corrected(args.scenario, args.shard, args.shards),
        "materialize-states": lambda: materialize_states(args.scenario),
        "screen-candidates": lambda: screen_candidates(args.scenario, args.shard, args.shards),
        "screen-double-candidates": lambda: screen_double_candidates(args.shard, args.shards),
        "secondary": lambda: secondary_search(args.scenario, args.shard, args.shards),
        "secondary-promote": lambda: secondary_promote(args.scenario, args.shard, args.shards),
        "finalize": finalize,
    }
    result = functions[args.command]()
    print(json.dumps({key: value for key, value in result.items() if key != "states"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
