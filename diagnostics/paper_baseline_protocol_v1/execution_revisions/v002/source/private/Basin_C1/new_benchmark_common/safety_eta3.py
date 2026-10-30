"""Formal hard-safety and frozen OrthoFlow3 evaluation for new benchmarks.

The command is deliberately resumable.  Every requested batch is preflighted
against the persistent rollout database and every newly completed rollout is
committed to SQLite before the next tuple is started.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np
import clarabel
import scipy
from scipy.stats import qmc

from new_benchmark_common.final_diagnostics import load_nominal
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from shared_control.basis_families import get_basis_family
from shared_control.hard_projection import (
    CBFSolverError,
    HardProjectionConfig,
    barrier_constraints,
    project_velocity,
)
from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import (
    CertifiedHardSafetyFilter,
)
from shared_rollout_db.src.planner import preflight
from shared_rollout_db.src.rollout_db import (
    canonical,
    connect,
    eta_identity,
    initialize,
    lookup_exact,
    uid,
)


ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = ROOT / "shared_rollout_db"
DOMAIN_LOW = np.asarray([0.5, -0.5, 0.0], dtype=np.float64)
DOMAIN_HIGH = np.asarray([1.25, 0.5, 0.75], dtype=np.float64)
PRIMARY_SOBOL_SEED = 20261001
SECONDARY_SOBOL_SEED = 20261002
LOCAL_SOBOL_SEED = 20261003
FUTURE_ROOT = 2026093011
FROZEN_EVAL_SEEDS = {"four_way_intersection": 8123, "ring_exchange": 3127}
STANDARD_SEEDS = tuple(range(16))
SCREEN_SEEDS = tuple(range(4))
CONTROL_SEEDS = tuple(range(8))
LOCAL_SCREEN_SEEDS = tuple(range(8))
LOCAL_PROMOTE = tuple(range(12))
ROBUST_PROTOCOL = "canonical_15of16_seeds_0_to_15_v1"
ROBUST_STOP_REASONS = {
    "ROBUST_IMPOSSIBLE_2_FAILURES",
    "ROBUST_CONFIRMED_15_SUCCESSES",
    "FULL_16_EVALUATED",
}


SCENARIOS = {
    "four_way_intersection": {
        "dataset": ROOT / "diagnostics/four_way_intersection_stage1/base_u_v13_broad_global_dataset",
        "checkpoint": ROOT / "diagnostics/four_way_intersection_stage1/base_u_v13_broad_global_source_balanced_macflow/best.pkl",
        "output": ROOT / "diagnostics/four_way_intersection_safety_eta3",
        "kind": "four",
    },
    "ring_exchange": {
        "dataset": ROOT / "diagnostics/ring_exchange_stage1/base_u_v10_local_dataset",
        "checkpoint": ROOT / "diagnostics/ring_exchange_stage1/base_u_v10_local_macflow/best.pkl",
        "output": ROOT / "diagnostics/ring_exchange_safety_eta3",
        "kind": "ring",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_sources(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def state_payload(trajectory) -> dict[str, Any]:
    initial = trajectory.initial_state
    return {
        "positions": np.asarray(initial["positions"], dtype=np.float64).tolist(),
        "velocities": np.asarray(initial["velocities"], dtype=np.float64).tolist(),
        "goals": None if "goals" not in initial else np.asarray(initial["goals"], dtype=np.float64).tolist(),
    }


def state_token(state_uid: str) -> int:
    return int(hashlib.sha256(state_uid.encode()).hexdigest()[:8], 16)


def designs() -> dict[str, Any]:
    def global_design(seed: int) -> np.ndarray:
        unit = qmc.Sobol(3, scramble=True, seed=seed).random_base2(8)
        return qmc.scale(unit, DOMAIN_LOW, DOMAIN_HIGH)

    local_unit = qmc.Sobol(3, scramble=True, seed=LOCAL_SOBOL_SEED).random_base2(5)
    local_offsets = (local_unit - 0.5) * 0.10
    return {
        "domain": {"lower": DOMAIN_LOW.tolist(), "upper": DOMAIN_HIGH.tolist()},
        "primary_seed": PRIMARY_SOBOL_SEED,
        "secondary_seed": SECONDARY_SOBOL_SEED,
        "local_seed": LOCAL_SOBOL_SEED,
        "primary": global_design(PRIMARY_SOBOL_SEED).tolist(),
        "secondary": global_design(SECONDARY_SOBOL_SEED).tolist(),
        "local_normalized_offsets": local_offsets.tolist(),
        "local_radius_full_width": 0.05,
    }


def normalized_eta(value: np.ndarray | Iterable[float]) -> np.ndarray:
    """Frozen Double-Bottleneck domain-centred normalization."""
    eta = np.asarray(value, dtype=np.float64)
    return (eta - 0.5 * (DOMAIN_LOW + DOMAIN_HIGH)) / (DOMAIN_HIGH - DOMAIN_LOW)


class ScenarioRuntime:
    def __init__(self, name: str):
        spec = SCENARIOS[name]
        self.name, self.spec, self.kind = name, spec, spec["kind"]
        self.output = spec["output"]
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest, self.trajectories, self.selection_audit = load_nominal(
            spec["dataset"], "test", frozen_test=True
        )
        self.config_dict = dict(self.manifest["scenario_config"])
        self.agent, self.checkpoint_metadata = load_checkpoint(
            spec["checkpoint"],
            expected_environment_fingerprint=self.manifest["environment_fingerprint"],
        )
        self.checkpoint_sha = sha256_file(spec["checkpoint"])
        self.cbf = HardProjectionConfig()
        self.projector = CertifiedHardSafetyFilter(self.cbf)
        self.basis = get_basis_family("orthoflow3")
        if self.kind == "four":
            from four_way_intersection.environment import Config, FourWayIntersectionEnv
            from four_way_intersection.rollout import crossing_order_signature

            self.config = Config(**self.config_dict)
            self.make_env = lambda: FourWayIntersectionEnv(self.config)
            self.mode_fn = crossing_order_signature
            self.env_hash = sha256_sources(
                [ROOT / "four_way_intersection/environment.py", ROOT / "four_way_intersection/scenario.py"]
            )
            self.safety_hash = sha256_sources(
                [ROOT / "shared_control/hard_projection.py", ROOT / "four_way_intersection/safety.py",
                 ROOT / "diagnostics/double_bottleneck_eta3_basin/tools/exact_projection_retry.py"]
            )
        else:
            from ring_exchange.environment import LocalFrameConfig, RingExchangeEnv
            from ring_exchange.expert import circulation_signature

            self.config = LocalFrameConfig(**self.config_dict)
            self.make_env = lambda: RingExchangeEnv(self.config)
            self.mode_fn = circulation_signature
            self.env_hash = sha256_sources(
                [ROOT / "ring_exchange/environment.py", ROOT / "ring_exchange/local_frame.py"]
            )
            self.safety_hash = sha256_sources(
                [ROOT / "shared_control/hard_projection.py", ROOT / "ring_exchange/safety.py",
                 ROOT / "diagnostics/double_bottleneck_eta3_basin/tools/exact_projection_retry.py"]
            )
        self.orthoflow_hash = sha256_file(ROOT / "shared_control/basis_families.py")
        self.macflow_hash = sha256_file(ROOT / "new_benchmark_common/macflow.py")
        self.scenario_code_hash = hashlib.sha256(
            canonical({
                "environment": self.env_hash,
                "config": self.config_dict,
                "representation": self.kind,
            }).encode()
        ).hexdigest()
        self.scenario_uid = uid("scn", {"name": name, "code": self.scenario_code_hash})
        self.states: list[dict[str, Any]] = []
        for index, trajectory in enumerate(self.trajectories):
            physical = state_payload(trajectory)
            content_hash = hashlib.sha256(canonical(physical).encode()).hexdigest()
            state_uid = uid("state", {"scenario": self.scenario_uid, "content": content_hash})
            self.states.append({
                "index": index,
                "alias": trajectory.rollout_id,
                "uid": state_uid,
                "content_hash": content_hash,
                "physical": physical,
                "trajectory": trajectory,
            })
        self.controllers = {
            name_: self._controller(name_)
            for name_ in ("no_safety", "hard_safety", "orthoflow3")
        }
        self.experiment_uid = uid("exp", {"path": str(self.output.resolve())})
        self._register_identities()

    def _controller(self, chain: str) -> dict[str, Any]:
        if chain in ("no_safety", "hard_safety"):
            rng = {
                "name": "frozen_stage1_evaluation_seed_v1",
                "evaluation_seed": FROZEN_EVAL_SEEDS[self.name],
                "derivation": "fold_in(PRNGKey(evaluation_seed),frozen_test_index),then_step",
            }
        else:
            rng = {
                "name": "matched_future_index_v1",
                "future_root": FUTURE_ROOT,
                "derivation": "fold_in(fold_in(PRNGKey(root),state_token),future_index),then_step",
            }
        payload = {
            "scenario": self.name,
            "chain": chain,
            "flow_checkpoint_sha256": self.checkpoint_sha,
            "macflow_source_sha256": self.macflow_hash,
            "environment_sha256": self.env_hash,
            "orthoflow3_sha256": self.orthoflow_hash if chain == "orthoflow3" else None,
            "safety_projection_sha256": self.safety_hash if chain != "no_safety" else None,
            "safety_config": self.cbf.to_dict() if chain != "no_safety" else None,
            "horizon": self.config.max_steps,
            "dt": self.config.dt,
            "success_semantics": "collision_free_all_agents_goal_tolerance_v1",
            "conditioning": "frozen_test_initial_state_episode_eta_v1",
            "rng": rng,
            "runtime_versions": {
                "jax": jax.__version__, "numpy": np.__version__,
                "scipy": scipy.__version__, "clarabel": clarabel.__version__,
            },
        }
        return {"uid": uid("ctl", payload), "payload": payload}

    def _register_identities(self) -> None:
        initialize()
        with connect() as con:
            con.execute(
                "INSERT OR IGNORE INTO scenario(scenario_uid,name,code_config_fingerprint,metadata_json) VALUES(?,?,?,?)",
                (self.scenario_uid, self.name, self.scenario_code_hash,
                 canonical({"frozen_stage1": True, "environment_fingerprint": self.manifest["environment_fingerprint"]})),
            )
            con.execute(
                "INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,metadata_json) VALUES(?,?,?,?,?,?)",
                (self.experiment_uid, f"{self.name}_safety_eta3", str(self.output.resolve()), None,
                 sha256_file(Path(__file__)), canonical({"resumable": True, "task": "formal_safety_eta3"})),
            )
            for state in self.states:
                con.execute(
                    """INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,
                    goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)""",
                    (state["uid"], self.scenario_uid, "frozen_test", state["content_hash"],
                     canonical(state["physical"]), canonical({"goals": state["physical"].get("goals")}),
                     canonical({"rollout_id": state["alias"], "dataset_manifest": self.selection_audit["manifest_sha256"]}),
                     "CONTENT_EXACT"),
                )
                con.execute(
                    "INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)",
                    (self.scenario_uid, state["alias"], state["uid"], self.experiment_uid),
                )
            for controller in self.controllers.values():
                p = controller["payload"]
                con.execute(
                    """INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,
                    orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,
                    rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (controller["uid"], self.scenario_uid, self.checkpoint_sha, p["orthoflow3_sha256"],
                     self.safety_hash if p["safety_projection_sha256"] else None, str(self.config.max_steps),
                     str(self.config.dt), p["success_semantics"], p["conditioning"], canonical(p["rng"]),
                     canonical(p), "EXACT_PROFILE"),
                )
            con.commit()

    def reset(self, env: Any, state: dict[str, Any]) -> None:
        p = state["physical"]
        if self.kind == "four":
            env.reset(np.asarray(p["positions"]), np.asarray(p["velocities"]))
        else:
            env.reset(np.asarray(p["positions"]), velocities=np.asarray(p["velocities"]), goals=np.asarray(p["goals"]))

    def observation(self, env: Any) -> np.ndarray:
        if self.kind == "four":
            return env.observation()
        from ring_exchange.local_frame import local_observation

        return local_observation(env.positions, env.velocities, env.goals, self.config)

    def flow_world(self, env: Any, key: jax.Array) -> np.ndarray:
        action = np.asarray(sample_bounded_actions(self.agent, self.observation(env)[None], key)[0], dtype=np.float64)
        if self.kind == "ring":
            from ring_exchange.local_frame import local_actions_to_world

            action = local_actions_to_world(action, env.positions)
        # Match final_diagnostics._bound64 exactly and remove float32/JAX
        # roundoff before the physical plant's strict speed certificate.
        norm = np.linalg.norm(action, axis=-1, keepdims=True)
        return action * np.minimum(1.0, self.config.max_speed / np.maximum(norm, 1e-30))

    def safety_snapshot(self, env: Any) -> dict[str, Any]:
        if self.kind == "four":
            return env.snapshot()
        from ring_exchange.safety import polygonal_obstacle_snapshot

        return polygonal_obstacle_snapshot(env, sides=48)

    def project(self, env: Any, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        snapshot = self.safety_snapshot(env)
        result = self.projector(snapshot, target)
        diagnostics = dict(result.diagnostics)
        diagnostics["status"] = result.status
        return np.asarray(result.velocity, dtype=np.float64), diagnostics

    def rollout(self, state: dict[str, Any], eta: np.ndarray, future_index: int, chain: str,
                *, save_trace: bool = False) -> dict[str, Any]:
        env = self.make_env()
        self.reset(env, state)
        if chain in ("no_safety", "hard_safety"):
            # Exact frozen Stage-I evaluator semantics.
            root = jax.random.fold_in(
                jax.random.PRNGKey(FROZEN_EVAL_SEEDS[self.name]), int(state["index"])
            )
        else:
            root = jax.random.fold_in(jax.random.PRNGKey(FUTURE_ROOT), state_token(state["uid"]))
            root = jax.random.fold_in(root, int(future_index))
        positions = [env.positions.copy()]
        speeds = []
        progress = []
        corrections = []
        correction_ratios = []
        first_active = []
        second_active = []
        min_wall = math.inf
        min_agent = math.inf
        min_obstacle = math.inf
        termination = "timeout"
        error = None
        last_info: dict[str, Any] = {}
        try:
            for step in range(self.config.max_steps):
                flow = self.flow_world(env, jax.random.fold_in(root, step))
                safe = flow
                first_diag = None
                if chain != "no_safety":
                    safe, first_diag = self.project(env, flow)
                    delta = float(np.linalg.norm(safe - flow))
                    first_active.append(delta > self.cbf.intervention_tol)
                    corrections.append(delta)
                    correction_ratios.append(delta / max(float(np.linalg.norm(flow)), 1e-12))
                executed = safe
                if chain == "orthoflow3":
                    fields = self.basis.compute(env.positions, env.goals, safe, self.config.max_speed)
                    corrected = safe + fields.correction(eta)
                    executed, second_diag = self.project(env, corrected)
                    second_active.append(
                        float(np.linalg.norm(executed - corrected)) > self.cbf.intervention_tol
                    )
                before_error = float(np.sum(np.linalg.norm(env.goals - env.positions, axis=1)))
                _, _, done, info = env.step(executed)
                after_error = float(np.sum(np.linalg.norm(env.goals - env.positions, axis=1)))
                last_info = info
                positions.append(env.positions.copy())
                speeds.append(np.linalg.norm(executed, axis=1).tolist())
                progress.append(before_error - after_error)
                walls, pairs = env.distances()
                min_wall = min(min_wall, float(np.min(walls)))
                min_agent = min(min_agent, float(np.min(pairs)))
                if self.kind == "ring":
                    min_obstacle = min(min_obstacle, float(np.min(walls[:, 0])))
                termination = str(info.get("termination", "running"))
                if done:
                    break
        except (CBFSolverError, ValueError, FloatingPointError) as exc:
            termination = "numerical_failure"
            error = f"{type(exc).__name__}: {exc}"
        summary = env.summary()
        collision = bool(
            summary.get("wall_collision", False)
            or summary.get("obstacle_collision", False)
            or summary.get("agent_collision", False)
        )
        success = bool(summary.get("collision_free_success", False)) and termination == "success"
        timeout = termination == "timeout"
        path = np.asarray(positions)
        result = {
            "scenario": self.name,
            "state_id": state["alias"],
            "state_uid": state["uid"],
            "initial_state_index": state["index"],
            "initial_positions": state["physical"]["positions"],
            "initial_velocities": state["physical"]["velocities"],
            "goals": state["physical"].get("goals"),
            "eta": np.asarray(eta, dtype=np.float64).tolist(),
            "future_index": int(future_index),
            "future_root_seed": FROZEN_EVAL_SEEDS[self.name] if chain in ("no_safety", "hard_safety") else FUTURE_ROOT,
            "rng_namespace": state_token(state["uid"]),
            "controller_chain": chain,
            "controller_uid": self.controllers[chain]["uid"],
            "flow_checkpoint_sha256": self.checkpoint_sha,
            "environment_sha256": self.env_hash,
            "orthoflow3_sha256": self.orthoflow_hash if chain == "orthoflow3" else None,
            "safety_projection_sha256": self.safety_hash if chain != "no_safety" else None,
            "outcome": "success" if success else ("collision" if collision else termination),
            "termination": termination,
            "success": success,
            "deadlock": False,
            "timeout": timeout,
            "collision": collision,
            "numerical_failure": error is not None,
            "execution_error": error,
            "wall_collision": bool(summary.get("wall_collision", False)),
            "obstacle_collision": bool(summary.get("obstacle_collision", False)),
            "outer_collision": bool(summary.get("outer_collision", False)),
            "agent_collision": bool(summary.get("agent_collision", False)),
            "episode_length": int(summary.get("episode_steps", len(path) - 1)),
            "minimum_wall_clearance": None if not np.isfinite(min_wall) else min_wall,
            "minimum_obstacle_clearance": None if not np.isfinite(min_obstacle) else min_obstacle,
            "minimum_agent_clearance": None if not np.isfinite(min_agent) else min_agent,
            "mode_signature": self.mode_fn(path),
            "projection_active_fraction": float(np.mean(first_active)) if first_active else 0.0,
            "projection_correction_norm_mean": float(np.mean(corrections)) if corrections else 0.0,
            "projection_correction_ratio_mean": float(np.mean(correction_ratios)) if correction_ratios else 0.0,
            "second_projection_active_fraction": float(np.mean(second_active)) if second_active else 0.0,
            "terminal_goal_error": float(np.sum(np.linalg.norm(env.goals - env.positions, axis=1))),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if save_trace:
            result["trace"] = {
                "positions": path.tolist(),
                "speeds": speeds,
                "goal_error_progress": progress,
            }
        return result


class DatabaseSink:
    """Append raw provenance and immediately commit one exact rollout."""

    def __init__(self, runtime: ScenarioRuntime, stage: str):
        self.runtime = runtime
        self.stage = stage
        self.path = runtime.output / "raw" / f"{stage}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.source_uid = uid("src", {"path": str(self.path.resolve())})
        self.live_hash = "LIVE_" + hashlib.sha256(str(self.path.resolve()).encode()).hexdigest()

    def insert(self, row: dict[str, Any], *, append_raw: bool = True) -> None:
        # Recovery of a previously flushed JSONL row must not append a second
        # physical execution. The caller is responsible for reading that row
        # from this sink's own path when append_raw is false.
        if append_raw:
            line = canonical(row) + "\n"
            with self.path.open("a") as handle:
                handle.write(line)
                handle.flush()
        eta_uid, eta, eta_hex = eta_identity(row["eta"])
        seed_key = canonical({"future_index": int(row["future_index"])})
        identity = {
            "state": row["state_uid"], "eta": eta_uid,
            "controller": row["controller_uid"], "seed": seed_key,
        }
        rollout_uid = uid("roll", identity)
        raw_hash = hashlib.sha256(canonical(row).encode()).hexdigest()
        source_uid = self.source_uid
        with connect() as con:
            con.execute(
                """INSERT INTO eta(eta_uid,eta1,eta2,eta3,normalized_eta_json,canonical_hex)
                VALUES(?,?,?,?,?,?) ON CONFLICT(eta_uid) DO UPDATE SET
                normalized_eta_json=excluded.normalized_eta_json""",
                (eta_uid, *eta, canonical(normalized_eta(eta).tolist()), eta_hex),
            )
            con.execute(
                """INSERT OR IGNORE INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen)
                VALUES(?,?,?,?,?,?,?)""",
                (source_uid, self.runtime.experiment_uid, str(self.path.resolve()), self.live_hash,
                 ".jsonl", "SEED_EXACT", 0),
            )
            con.execute(
                "UPDATE source_file SET sha256=?, rows_seen=rows_seen+1 WHERE source_uid=?",
                (self.live_hash, source_uid),
            )
            existing = con.execute(
                "SELECT rollout_uid,success,deadlock,timeout,collision,numerical_failure,raw_record_hash,original_source_file FROM rollout WHERE rollout_uid=?",
                (rollout_uid,),
            ).fetchone()
            core = (
                int(row["success"]), int(row["deadlock"]), int(row["timeout"]),
                int(row["collision"]), int(row["numerical_failure"]),
            )
            if existing is not None:
                old_core = tuple(existing[key] for key in
                                 ("success", "deadlock", "timeout", "collision", "numerical_failure"))
                if bool(existing["numerical_failure"]) and not bool(row["numerical_failure"]):
                    # Numerical executions are explicitly not scientific evidence.
                    # Preserve their raw append-only source, audit the replacement,
                    # and let the valid execution occupy the exact cache key.
                    audit_path = self.runtime.output / "invalid_numeric_attempt_v1/replacements.jsonl"
                    audit_path.parent.mkdir(parents=True, exist_ok=True)
                    with audit_path.open("a") as audit:
                        audit.write(canonical({"rollout_uid": rollout_uid,
                                               "old_core": old_core,
                                               "old_raw_record_hash": existing["raw_record_hash"],
                                               "old_source": existing["original_source_file"],
                                               "new_raw_record_hash": raw_hash}) + "\n")
                    con.execute("DELETE FROM rollout_source WHERE rollout_uid=?", (rollout_uid,))
                    con.execute("DELETE FROM rollout WHERE rollout_uid=?", (rollout_uid,))
                    existing = None
                elif old_core != core:
                    raise RuntimeError(f"database conflict for {rollout_uid}")
            con.execute(
                """INSERT OR IGNORE INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,
                continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,
                min_wall_distance,min_agent_distance,outcome,experiment_uid,original_source_file,timestamp,
                compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rollout_uid, row["state_uid"], eta_uid, row["controller_uid"], seed_key,
                 seed_key, *core, row["episode_length"], row["minimum_wall_clearance"],
                 row["minimum_agent_clearance"], row["outcome"], self.runtime.experiment_uid,
                 str(self.path.resolve()), row["timestamp"], "EXACT_REUSE", raw_hash),
            )
            con.execute(
                "INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)",
                (rollout_uid, source_uid, None),
            )
            con.commit()

    def finalize(self) -> None:
        """Seal the append-only shard once, avoiding quadratic re-hashing."""
        if not self.path.exists():
            return
        digest = sha256_file(self.path)
        with connect() as con:
            con.execute("UPDATE source_file SET sha256=? WHERE source_uid=?", (digest, self.source_uid))
            con.commit()


def request(runtime: ScenarioRuntime, state: dict[str, Any], eta: Iterable[float],
            chain: str, seeds: Iterable[int]) -> dict[str, Any]:
    eta_uid = eta_identity(eta)[0]
    return {
        "scenario_uid": runtime.scenario_uid,
        "state_uid": state["uid"],
        "eta_uid": eta_uid,
        "controller_uid": runtime.controllers[chain]["uid"],
        "seed_keys": [canonical({"future_index": int(seed)}) for seed in seeds],
    }


def cached_rows(runtime: ScenarioRuntime, state: dict[str, Any], eta: Iterable[float],
                chain: str, seeds: Iterable[int]) -> tuple[dict[int, dict[str, Any]], list[int]]:
    seed_list = list(seeds)
    keys = [canonical({"future_index": int(seed)}) for seed in seed_list]
    eta_uid = eta_identity(eta)[0]
    with connect(True) as con:
        found = lookup_exact(con, state["uid"], eta_uid, runtime.controllers[chain]["uid"], keys)
    by_key = {row["seed_key"]: row for row in found["records"]}
    rows = {seed: by_key[key] for seed, key in zip(seed_list, keys) if key in by_key}
    missing_keys = set(found["missing_seeds"])
    missing = [seed for seed, key in zip(seed_list, keys) if key in missing_keys]
    return rows, missing


def _robust_decision_from_rows(rows: dict[int, Any], *,
                               allow_early_acceptance: bool = True) -> dict[str, Any] | None:
    """Return an exact 15/16 logical decision, never imputing unrun seeds."""
    observed = [rows[seed] for seed in STANDARD_SEEDS if seed in rows]
    successes = sum(int(row["success"]) for row in observed)
    failures = len(observed) - successes
    if len(observed) == len(STANDARD_SEEDS):
        reason = "FULL_16_EVALUATED"
        robust = successes >= 15
    elif failures >= 2:
        reason = "ROBUST_IMPOSSIBLE_2_FAILURES"
        robust = False
    elif successes >= 15 and allow_early_acceptance:
        reason = "ROBUST_CONFIRMED_15_SUCCESSES"
        robust = True
    else:
        return None
    return {
        "evaluated_seed_count": len(observed),
        "n_success": successes,
        "n_failure": failures,
        "stop_reason": reason,
        "robust": robust,
        "observed_seeds": [seed for seed in STANDARD_SEEDS if seed in rows],
        "unrun_seeds": [seed for seed in STANDARD_SEEDS if seed not in rows],
    }


def persist_robust_decision(runtime: ScenarioRuntime, state: dict[str, Any],
                            eta: Iterable[float], stage: str,
                            decision: dict[str, Any]) -> None:
    """Store logical classification separately from actual seed rollouts."""
    if decision["stop_reason"] not in ROBUST_STOP_REASONS:
        raise ValueError("invalid exact robust stop reason")
    eta_uid = eta_identity(eta)[0]
    controller_uid = runtime.controllers["orthoflow3"]["uid"]
    identity = {"state": state["uid"], "eta": eta_uid,
                "controller": controller_uid, "protocol": ROBUST_PROTOCOL}
    decision_uid = uid("rdec", identity)
    provenance = {
        "criterion": "successes >= 15/16",
        "exact_logic": "reject at second observed failure; accept at fifteenth observed success",
        "observed_seeds": decision["observed_seeds"],
        "unrun_seeds": decision["unrun_seeds"],
        "unrun_seeds_are_not_failures": True,
    }
    with connect() as con:
        con.execute(
            """INSERT INTO robust_logical_decision(
            decision_uid,state_uid,eta_uid,controller_uid,protocol_name,
            canonical_seed_order_json,evaluated_seed_count,n_success,n_failure,
            stop_reason,robust,stage,experiment_uid,provenance_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(state_uid,eta_uid,controller_uid,protocol_name) DO UPDATE SET
              evaluated_seed_count=excluded.evaluated_seed_count,
              n_success=excluded.n_success,n_failure=excluded.n_failure,
              stop_reason=excluded.stop_reason,robust=excluded.robust,
              stage=excluded.stage,experiment_uid=excluded.experiment_uid,
              provenance_json=excluded.provenance_json,updated_at=CURRENT_TIMESTAMP""",
            (decision_uid, state["uid"], eta_uid, controller_uid, ROBUST_PROTOCOL,
             canonical(list(STANDARD_SEEDS)), decision["evaluated_seed_count"],
             decision["n_success"], decision["n_failure"], decision["stop_reason"],
             int(decision["robust"]), stage, runtime.experiment_uid,
             canonical(provenance)),
        )
        con.commit()


def robust_evidence(runtime: ScenarioRuntime, state: dict[str, Any],
                    eta: Iterable[float], *, persist_stage: str | None = None) -> dict[str, Any]:
    rows, missing = cached_rows(runtime, state, eta, "orthoflow3", STANDARD_SEEDS)
    decision = _robust_decision_from_rows(rows)
    if decision is None:
        eta_uid = eta_identity(eta)[0]
        keys = [canonical({"future_index": int(seed)}) for seed in missing]
        invalid_seeds: list[int] = []
        if keys:
            with connect(True) as con:
                placeholders = ",".join("?" * len(keys))
                invalid_keys = {row["seed_key"] for row in con.execute(
                    f"""SELECT seed_key FROM rollout WHERE state_uid=? AND eta_uid=?
                    AND controller_uid=? AND numerical_failure=1
                    AND seed_key IN ({placeholders})""",
                    (state["uid"], eta_uid, runtime.controllers["orthoflow3"]["uid"], *keys),
                )}
            invalid_seeds = [seed for seed in missing
                             if canonical({"future_index": int(seed)}) in invalid_keys]
        if missing and len(invalid_seeds) == len(missing):
            successes = sum(int(row["success"]) for row in rows.values())
            # Numerical-invalid executions are not task failures and are not
            # imputed.  Without 15 observed successes the tuple is simply not
            # certified robust; preserve the unresolved seeds explicitly.
            return {
                "evaluated_seed_count": len(rows), "n_success": successes,
                "n_failure": len(rows) - successes,
                "stop_reason": "NUMERICAL_INCOMPLETE_NOT_CERTIFIED",
                "robust": False,
                "observed_seeds": [seed for seed in STANDARD_SEEDS if seed in rows],
                "unrun_seeds": [],
                "unresolved_seeds": missing,
                "numerical_invalid_seeds": invalid_seeds,
            }
        raise RuntimeError(f"robust classification incomplete for {state['uid']} / {eta_uid}")
    if persist_stage is not None:
        persist_robust_decision(runtime, state, eta, persist_stage, decision)
    return decision


def execute_batch(runtime: ScenarioRuntime, stage: str, jobs: list[dict[str, Any]],
                  *, save_trace: bool = False, shard_index: int = 0,
                  num_shards: int = 1, exact_robust_15of16: bool = False,
                  exact_early_acceptance: bool = True) -> dict[str, Any]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid deterministic shard")
    jobs = [job for index, job in enumerate(jobs) if index % num_shards == shard_index]
    shard_suffix = "" if num_shards == 1 else f"_shard{shard_index}of{num_shards}"
    physical_stage = stage + shard_suffix
    manifest_path = runtime.output / "plans" / f"{physical_stage}.json"
    preflight_path = runtime.output / "cache" / f"{physical_stage}.json"
    manifest = {"schema": "new_benchmark_eta3_requests_v1", "requests": [
        request(runtime, job["state"], job["eta"], job["chain"], job["seeds"])
        for job in jobs
    ]}
    json_dump(manifest_path, manifest)
    pf = preflight(manifest_path)
    json_dump(preflight_path, pf)
    sink = DatabaseSink(runtime, physical_stage)
    physical = reused = 0
    stop_counts: Counter[str] = Counter()
    for job_index, job in enumerate(jobs):
        rows, missing = cached_rows(
            runtime, job["state"], job["eta"], job["chain"], job["seeds"]
        )
        reused += len(job["seeds"]) - len(missing)
        if exact_robust_15of16:
            if job["chain"] != "orthoflow3" or tuple(job["seeds"]) != STANDARD_SEEDS:
                raise ValueError("exact 15/16 stopping requires canonical OrthoFlow3 seeds 0..15")
            decision = _robust_decision_from_rows(
                rows, allow_early_acceptance=exact_early_acceptance)
            if decision is not None:
                persist_robust_decision(runtime, job["state"], job["eta"], stage, decision)
                stop_counts[decision["stop_reason"]] += 1
                continue
        for seed in missing:
            # A numerical solver failure is not a task outcome.  Match the
            # established Double-Bottleneck protocol by allowing three full,
            # byte-identical invocations after the first invalid execution.
            for execution_attempt in range(4):
                row = runtime.rollout(
                    job["state"], np.asarray(job["eta"], dtype=np.float64), seed,
                    job["chain"], save_trace=save_trace,
                )
                row.update({"stage": stage, "physical_stage": physical_stage,
                            "job_index": job_index,
                            "execution_attempt": execution_attempt,
                            **job.get("metadata", {})})
                sink.insert(row)
                physical += 1
                if physical % 25 == 0:
                    print(json.dumps({"stage": physical_stage, "physical": physical, "reused": reused}), flush=True)
                if not row["numerical_failure"]:
                    rows[seed] = row
                    break
            if exact_robust_15of16 and not row["numerical_failure"]:
                decision = _robust_decision_from_rows(
                    rows, allow_early_acceptance=exact_early_acceptance)
                if decision is not None:
                    persist_robust_decision(runtime, job["state"], job["eta"], stage, decision)
                    stop_counts[decision["stop_reason"]] += 1
                    break
    sink.finalize()
    result = {"stage": physical_stage, "logical_stage": stage, "jobs": len(jobs), "requested": sum(len(j["seeds"]) for j in jobs),
              "reused": reused, "physical": physical, "preflight": pf["summary"],
              "exact_robust_early_stop": exact_robust_15of16,
              "exact_early_acceptance": exact_early_acceptance,
              "stop_reason_counts": dict(stop_counts)}
    json_dump(runtime.output / "stage_status" / f"{physical_stage}.json", result)
    return result


def load_raw(runtime: ScenarioRuntime) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((runtime.output / "raw").glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def formal_jobs(runtime: ScenarioRuntime) -> list[dict[str, Any]]:
    return [
        {"state": state, "eta": [0.0, 0.0, 0.0], "chain": chain, "seeds": [0],
         "metadata": {"formal_matched_episode": True}}
        for state in runtime.states for chain in ("no_safety", "hard_safety")
    ]


def summarize_formal(runtime: ScenarioRuntime) -> dict[str, Any]:
    raw = [r for r in load_raw(runtime) if r.get("formal_matched_episode") and r.get("stage") == "formal"]
    indexed = {(r["state_uid"], r["controller_chain"]): r for r in raw}
    pairs = []
    for state in runtime.states:
        no = indexed[(state["uid"], "no_safety")]
        safe = indexed[(state["uid"], "hard_safety")]
        pairs.append({"state_id": state["alias"], "state_uid": state["uid"],
                      "no_safety": no, "hard_safety": safe,
                      "conversion": f"{no['outcome']}->{safe['outcome']}"})
    def agg(chain: str) -> dict[str, Any]:
        rows = [p[chain] for p in pairs]
        return {
            "episodes": len(rows), "success": sum(r["success"] for r in rows),
            "collision": sum(r["collision"] for r in rows),
            "wall_collision": sum(r["wall_collision"] for r in rows),
            "obstacle_collision": sum(r["obstacle_collision"] for r in rows),
            "agent_collision": sum(r["agent_collision"] for r in rows),
            "timeout": sum(r["timeout"] for r in rows),
            "numerical_failure": sum(r["numerical_failure"] for r in rows),
            "mean_episode_length": float(np.mean([r["episode_length"] for r in rows])),
            "min_wall_clearance": float(min(r["minimum_wall_clearance"] for r in rows)),
            "min_agent_clearance": float(min(r["minimum_agent_clearance"] for r in rows)),
            "projection_active_fraction_mean": float(np.mean([r["projection_active_fraction"] for r in rows])),
            "correction_norm_mean": float(np.mean([r["projection_correction_norm_mean"] for r in rows])),
            "correction_flow_ratio_mean": float(np.mean([r["projection_correction_ratio_mean"] for r in rows])),
        }
    safe_failures = [p["state_uid"] for p in pairs if
                     not p["hard_safety"]["success"] and not p["hard_safety"]["collision"] and
                     p["hard_safety"]["timeout"]]
    controls = [p["state_uid"] for p in pairs if p["hard_safety"]["success"]]
    # Uniformly spaced by frozen test order, a pre-registered deterministic rule.
    if len(controls) > 24:
        controls = [controls[i] for i in np.linspace(0, len(controls) - 1, 24, dtype=int)]
    summary = {
        "no_safety": agg("no_safety"), "hard_safety": agg("hard_safety"),
        "conversion_matrix": dict(Counter(p["conversion"] for p in pairs)),
        "safe_liveness_failure_state_uids": safe_failures,
        "safe_liveness_failure_state_ids": [next(s["alias"] for s in runtime.states if s["uid"] == x) for x in safe_failures],
        "success_control_state_uids": controls,
        "success_control_selection": "up to 24 uniformly spaced in frozen test order",
        "runtime_deadlock_available": False,
        "pairs": pairs,
    }
    json_dump(runtime.output / "formal_safety_summary.json", summary)
    return summary


def state_by_uid(runtime: ScenarioRuntime, state_uid: str) -> dict[str, Any]:
    return next(state for state in runtime.states if state["uid"] == state_uid)


def success_count(runtime: ScenarioRuntime, state: dict[str, Any], eta: Iterable[float],
                  seeds: Iterable[int]) -> int:
    seed_list = list(seeds)
    rows, missing = cached_rows(runtime, state, eta, "orthoflow3", seed_list)
    if missing:
        # Persistent numerical executions remain invalid, never task failures.
        # For thresholding, count them as zero-success worst cases, which is a
        # conservative lower bound.  A truly absent tuple still aborts.
        eta_uid = eta_identity(eta)[0]
        keys = [canonical({"future_index": int(seed)}) for seed in missing]
        with connect(True) as con:
            placeholders = ",".join("?" * len(keys))
            invalid = con.execute(
                f"""SELECT seed_key FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?
                AND numerical_failure=1 AND seed_key IN ({placeholders})""",
                (state["uid"], eta_uid, runtime.controllers["orthoflow3"]["uid"], *keys),
            ).fetchall()
        if len(invalid) != len(missing):
            raise RuntimeError(f"genuinely missing cached seeds {missing}")
    return int(sum(int(row["success"]) for row in rows.values()))


def neighbor_support(points: np.ndarray, positive: set[int], index: int) -> int:
    normalized = (points - DOMAIN_LOW) / (DOMAIN_HIGH - DOMAIN_LOW)
    distances = np.linalg.norm(normalized - normalized[index], axis=1)
    nearest = [int(i) for i in np.argsort(distances) if int(i) != index][:8]
    return sum(i in positive for i in nearest)


def promote(points: np.ndarray, q4: dict[int, int]) -> list[int]:
    full = sorted(index for index, count in q4.items() if count == 4)
    partial = {index for index, count in q4.items() if count == 3}
    ranked = sorted(
        partial,
        key=lambda index: (
            -neighbor_support(points, {i for i, c in q4.items() if c >= 3}, index),
            float(np.linalg.norm(normalized_eta(points[index]))),
            index,
        ),
    )
    # Every 4/4 point is mandatory.  The 16-point cap applies to adding 3/4
    # points; if more than 16 perfect screen points exist, retain all perfects.
    if len(full) >= 16:
        return full
    return full + ranked[: 16 - len(full)]


def screen_design(runtime: ScenarioRuntime, label: str, points: np.ndarray,
                  state_uids: list[str]) -> dict[str, Any]:
    jobs = [
        {"state": state_by_uid(runtime, sid), "eta": points[index], "chain": "orthoflow3",
         "seeds": SCREEN_SEEDS, "metadata": {"design": label, "eta_index": index}}
        for sid in state_uids for index in range(len(points))
    ]
    execute_batch(runtime, f"{label}_q4", jobs)
    state_results = {}
    promotion_jobs = []
    for sid in state_uids:
        state = state_by_uid(runtime, sid)
        q4 = {index: success_count(runtime, state, points[index], SCREEN_SEEDS) for index in range(len(points))}
        promoted = promote(points, q4)
        state_results[sid] = {"q4": q4, "promoted_indices": promoted}
        for index in promoted:
            promotion_jobs.append({"state": state, "eta": points[index], "chain": "orthoflow3",
                                   "seeds": STANDARD_SEEDS,
                                   "metadata": {"design": label, "eta_index": index, "promoted": True}})
    execute_batch(runtime, f"{label}_q16", promotion_jobs,
                  exact_robust_15of16=True, exact_early_acceptance=False)
    for sid, result in state_results.items():
        state = state_by_uid(runtime, sid)
        evidence = {index: robust_evidence(runtime, state, points[index],
                                           persist_stage=f"{label}_q16")
                    for index in result["promoted_indices"]}
        result["q16"] = {index: value["n_success"] for index, value in evidence.items()}
        result["q16_evidence"] = evidence
        result["robust_indices"] = [index for index, value in evidence.items() if value["robust"]]
    json_dump(runtime.output / f"{label}_search.json", state_results)
    return state_results


def choose_center(points: np.ndarray, result: dict[str, Any]) -> int | None:
    robust = result["robust_indices"]
    if not robust:
        return None
    positive = {int(i) for i, count in result["q4"].items() if count >= 3}
    return min(
        robust,
        key=lambda index: (
            -result["q16"][index], -neighbor_support(points, positive, index),
            float(np.linalg.norm(normalized_eta(points[index]))), index,
        ),
    )


def eta_search(runtime: ScenarioRuntime) -> dict[str, Any]:
    formal = json.loads((runtime.output / "formal_safety_summary.json").read_text())
    targets = formal["safe_liveness_failure_state_uids"]
    design = designs()
    json_dump(runtime.output / "frozen_eta_design.json", design)
    primary_points = np.asarray(design["primary"])
    secondary_points = np.asarray(design["secondary"])
    primary = screen_design(runtime, "primary", primary_points, targets)
    negatives = [sid for sid, result in primary.items() if not result["robust_indices"]]
    secondary = screen_design(runtime, "secondary", secondary_points, negatives) if negatives else {}
    centers = {}
    for sid in targets:
        first = choose_center(primary_points, primary[sid])
        if first is not None:
            centers[sid] = {"design": "primary", "index": first, "eta": primary_points[first].tolist()}
            continue
        second = choose_center(secondary_points, secondary[sid])
        if second is not None:
            centers[sid] = {"design": "secondary", "index": second, "eta": secondary_points[second].tolist()}
    summary = {"targets": targets, "negative_after_primary": negatives,
               "robust_centers": centers,
               "no_robust_eta_observed": [sid for sid in targets if sid not in centers]}
    json_dump(runtime.output / "robust_existence.json", summary)
    return summary


def q4_only(runtime: ScenarioRuntime, label: str, *, shard_index: int, num_shards: int) -> dict[str, Any]:
    formal = json.loads((runtime.output / "formal_safety_summary.json").read_text())
    targets = formal["safe_liveness_failure_state_uids"]
    design = designs()
    json_dump(runtime.output / "frozen_eta_design.json", design)
    points = np.asarray(design[label])
    if label == "secondary" and (runtime.output / "primary_search.json").exists():
        primary = json.loads((runtime.output / "primary_search.json").read_text())
        targets = [sid for sid in targets if not primary[sid]["robust_indices"]]
    jobs = [
        {"state": state_by_uid(runtime, sid), "eta": points[index], "chain": "orthoflow3",
         "seeds": SCREEN_SEEDS, "metadata": {"design": label, "eta_index": index}}
        for sid in targets for index in range(len(points))
    ]
    return execute_batch(runtime, f"{label}_q4", jobs,
                         shard_index=shard_index, num_shards=num_shards)


def screen_targets(runtime: ScenarioRuntime, label: str) -> tuple[list[str], np.ndarray]:
    formal = json.loads((runtime.output / "formal_safety_summary.json").read_text())
    targets = formal["safe_liveness_failure_state_uids"]
    if label == "secondary":
        primary = json.loads((runtime.output / "primary_search.json").read_text())
        targets = [sid for sid in targets if not primary[sid]["robust_indices"]]
    return targets, np.asarray(designs()[label], dtype=np.float64)


def plan_promotions(runtime: ScenarioRuntime, label: str) -> dict[str, Any]:
    targets, points = screen_targets(runtime, label)
    planned = {}
    for sid in targets:
        state = state_by_uid(runtime, sid)
        q4 = {index: success_count(runtime, state, points[index], SCREEN_SEEDS)
              for index in range(len(points))}
        planned[sid] = {"q4": q4, "promoted_indices": promote(points, q4)}
    json_dump(runtime.output / f"{label}_promotion_plan.json", planned)
    return planned


def promotion_q16_only(runtime: ScenarioRuntime, label: str, *, shard_index: int,
                       num_shards: int) -> dict[str, Any]:
    plan = json.loads((runtime.output / f"{label}_promotion_plan.json").read_text())
    points = np.asarray(designs()[label], dtype=np.float64)
    jobs = []
    for sid, result in plan.items():
        state = state_by_uid(runtime, sid)
        for index in result["promoted_indices"]:
            jobs.append({"state": state, "eta": points[int(index)], "chain": "orthoflow3",
                         "seeds": STANDARD_SEEDS,
                         "metadata": {"design": label, "eta_index": int(index), "promoted": True}})
    return execute_batch(runtime, f"{label}_q16", jobs,
                         shard_index=shard_index, num_shards=num_shards,
                         exact_robust_15of16=True, exact_early_acceptance=False)


def finalize_screen(runtime: ScenarioRuntime, label: str) -> dict[str, Any]:
    plan = json.loads((runtime.output / f"{label}_promotion_plan.json").read_text())
    points = np.asarray(designs()[label], dtype=np.float64)
    for sid, result in plan.items():
        state = state_by_uid(runtime, sid)
        evidence = {int(index): robust_evidence(runtime, state, points[int(index)],
                                                persist_stage=f"{label}_q16")
                    for index in result["promoted_indices"]}
        result["q16"] = {index: value["n_success"] for index, value in evidence.items()}
        result["q16_evidence"] = evidence
        result["robust_indices"] = [index for index, value in evidence.items() if value["robust"]]
    json_dump(runtime.output / f"{label}_search.json", plan)
    return plan


def finalize_existence(runtime: ScenarioRuntime) -> dict[str, Any]:
    formal = json.loads((runtime.output / "formal_safety_summary.json").read_text())
    targets = formal["safe_liveness_failure_state_uids"]
    primary = json.loads((runtime.output / "primary_search.json").read_text())
    secondary_path = runtime.output / "secondary_search.json"
    secondary = json.loads(secondary_path.read_text()) if secondary_path.exists() else {}
    ppoints = np.asarray(designs()["primary"])
    spoints = np.asarray(designs()["secondary"])
    centers = {}
    for sid in targets:
        first = choose_center(ppoints, _int_key_search(primary[sid]))
        if first is not None:
            centers[sid] = {"design": "primary", "index": first, "eta": ppoints[first].tolist()}
        elif sid in secondary:
            second = choose_center(spoints, _int_key_search(secondary[sid]))
            if second is not None:
                centers[sid] = {"design": "secondary", "index": second, "eta": spoints[second].tolist()}
    result = {"targets": targets,
              "negative_after_primary": [sid for sid in targets if not primary[sid]["robust_indices"]],
              "robust_centers": centers,
              "no_robust_eta_observed": [sid for sid in targets if sid not in centers]}
    json_dump(runtime.output / "robust_existence.json", result)
    return result


def _int_key_search(result: dict[str, Any]) -> dict[str, Any]:
    result = dict(result)
    result["q4"] = {int(k): int(v) for k, v in result["q4"].items()}
    result["q16"] = {int(k): int(v) for k, v in result["q16"].items()}
    if "q16_evidence" in result:
        result["q16_evidence"] = {int(k): v for k, v in result["q16_evidence"].items()}
    result["promoted_indices"] = [int(x) for x in result["promoted_indices"]]
    result["robust_indices"] = [int(x) for x in result["robust_indices"]]
    return result


def local_and_coverage(runtime: ScenarioRuntime) -> dict[str, Any]:
    existence = json.loads((runtime.output / "robust_existence.json").read_text())
    design = json.loads((runtime.output / "frozen_eta_design.json").read_text())
    targets = existence["targets"]
    centers_by_state = existence["robust_centers"]
    offsets = np.asarray(design["local_normalized_offsets"])
    local_results = {}
    local_jobs = []
    local_points_by_state = {}
    for sid, center in centers_by_state.items():
        eta = np.asarray(center["eta"])
        unit = (eta - DOMAIN_LOW) / (DOMAIN_HIGH - DOMAIN_LOW)
        points = DOMAIN_LOW + np.clip(unit + offsets, 0.0, 1.0) * (DOMAIN_HIGH - DOMAIN_LOW)
        local_points_by_state[sid] = points
        state = state_by_uid(runtime, sid)
        for index, point in enumerate(points):
            local_jobs.append({"state": state, "eta": point, "chain": "orthoflow3",
                               "seeds": LOCAL_SCREEN_SEEDS,
                               "metadata": {"local_center_state": sid, "local_index": index}})
    execute_batch(runtime, "local_q8", local_jobs)
    promote_jobs = []
    for sid, points in local_points_by_state.items():
        state = state_by_uid(runtime, sid)
        q8 = {index: success_count(runtime, state, points[index], LOCAL_SCREEN_SEEDS)
              for index in range(len(points))}
        for index in LOCAL_PROMOTE:
            promote_jobs.append({"state": state, "eta": points[index], "chain": "orthoflow3",
                                 "seeds": STANDARD_SEEDS,
                                 "metadata": {"local_center_state": sid, "local_index": index,
                                              "local_promoted": True}})
        local_results[sid] = {"q8": q8}
    execute_batch(runtime, "local_q16", promote_jobs, exact_robust_15of16=True)
    for sid, points in local_points_by_state.items():
        state = state_by_uid(runtime, sid)
        evidence = {index: robust_evidence(runtime, state, points[index],
                                           persist_stage="local_q16")
                    for index in LOCAL_PROMOTE}
        local_results[sid]["q16"] = {index: value["n_success"] for index, value in evidence.items()}
        local_results[sid]["q16_evidence"] = evidence
        local_results[sid]["robust_fraction"] = sum(v["robust"] for v in evidence.values()) / 12.0
    json_dump(runtime.output / "local_basin.json", local_results)

    unique = []
    seen = set()
    for center in centers_by_state.values():
        eta = tuple(float(x) for x in center["eta"])
        if eta not in seen:
            seen.add(eta); unique.append(eta)
    coverage_jobs = [
        {"state": state_by_uid(runtime, sid), "eta": eta, "chain": "orthoflow3",
         "seeds": STANDARD_SEEDS, "metadata": {"cross_state_center": list(eta)}}
        for eta in unique for sid in targets
    ]
    execute_batch(runtime, "cross_state_q16", coverage_jobs, exact_robust_15of16=True)
    coverage = []
    memberships = {}
    for center_index, eta in enumerate(unique):
        robust_states = []
        for sid in targets:
            decision = robust_evidence(runtime, state_by_uid(runtime, sid), eta,
                                       persist_stage="cross_state_q16")
            if decision["robust"]:
                robust_states.append(sid)
        memberships[center_index] = set(robust_states)
        coverage.append({"center_index": center_index, "eta": list(eta),
                         "robust_states": robust_states, "coverage_count": len(robust_states),
                         "coverage_fraction": len(robust_states) / max(len(targets), 1)})
    overlaps = []
    for i, left in enumerate(targets):
        left_set = {k for k, values in memberships.items() if left in values}
        for right in targets[i + 1:]:
            right_set = {k for k, values in memberships.items() if right in values}
            union = left_set | right_set
            overlaps.append({"left": left, "right": right,
                             "jaccard": len(left_set & right_set) / len(union) if union else 0.0})
    json_dump(runtime.output / "cross_state_coverage.json", {"centers": coverage, "overlaps": overlaps})

    controls = json.loads((runtime.output / "formal_safety_summary.json").read_text())["success_control_state_uids"]
    control_jobs = [
        {"state": state_by_uid(runtime, sid), "eta": eta, "chain": "orthoflow3",
         "seeds": CONTROL_SEEDS, "metadata": {"success_control_center": list(eta)}}
        for eta in unique for sid in controls
    ]
    execute_batch(runtime, "success_controls_q8", control_jobs)
    preservation = []
    for center_index, eta in enumerate(unique):
        successes = trials = 0
        for sid in controls:
            state = state_by_uid(runtime, sid)
            successes += success_count(runtime, state, eta, CONTROL_SEEDS)
            trials += len(CONTROL_SEEDS)
        preservation.append({"center_index": center_index, "eta": list(eta),
                             "successes": successes, "trials": trials,
                             "preservation_probability": successes / max(trials, 1)})
    json_dump(runtime.output / "success_control_preservation.json", preservation)
    return {"local": local_results, "coverage": coverage, "overlaps": overlaps,
            "preservation": preservation}


def local_context(runtime: ScenarioRuntime) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    existence = json.loads((runtime.output / "robust_existence.json").read_text())
    offsets = np.asarray(designs()["local_normalized_offsets"])
    points = {}
    for sid, center in existence["robust_centers"].items():
        eta = np.asarray(center["eta"], dtype=np.float64)
        unit = (eta - DOMAIN_LOW) / (DOMAIN_HIGH - DOMAIN_LOW)
        points[sid] = DOMAIN_LOW + np.clip(unit + offsets, 0.0, 1.0) * (DOMAIN_HIGH - DOMAIN_LOW)
    return existence, points


def local_stage_only(runtime: ScenarioRuntime, q16: bool, *, shard_index: int,
                     num_shards: int) -> dict[str, Any]:
    _, point_map = local_context(runtime)
    jobs = []
    indices = LOCAL_PROMOTE if q16 else tuple(range(32))
    seeds = STANDARD_SEEDS if q16 else LOCAL_SCREEN_SEEDS
    for sid, points in point_map.items():
        state = state_by_uid(runtime, sid)
        for index in indices:
            jobs.append({"state": state, "eta": points[index], "chain": "orthoflow3",
                         "seeds": seeds,
                         "metadata": {"local_center_state": sid, "local_index": index,
                                      "local_promoted": q16}})
    return execute_batch(runtime, "local_q16" if q16 else "local_q8", jobs,
                         shard_index=shard_index, num_shards=num_shards,
                         exact_robust_15of16=q16)


def finalize_local(runtime: ScenarioRuntime) -> dict[str, Any]:
    _, point_map = local_context(runtime)
    result = {}
    for sid, points in point_map.items():
        state = state_by_uid(runtime, sid)
        q8 = {index: success_count(runtime, state, points[index], LOCAL_SCREEN_SEEDS)
              for index in range(32)}
        evidence = {index: robust_evidence(runtime, state, points[index],
                                           persist_stage="local_q16")
                    for index in LOCAL_PROMOTE}
        result[sid] = {"q8": q8,
                       "q16": {index: value["n_success"] for index, value in evidence.items()},
                       "q16_evidence": evidence,
                       "robust_fraction": sum(value["robust"] for value in evidence.values()) / 12.0}
    json_dump(runtime.output / "local_basin.json", result)
    return result


def unique_centers(runtime: ScenarioRuntime) -> tuple[list[str], list[tuple[float, float, float]]]:
    existence = json.loads((runtime.output / "robust_existence.json").read_text())
    centers = []
    seen = set()
    for center in existence["robust_centers"].values():
        eta = tuple(float(x) for x in center["eta"])
        if eta not in seen:
            seen.add(eta); centers.append(eta)
    return existence["targets"], centers


def coverage_only(runtime: ScenarioRuntime, *, shard_index: int, num_shards: int) -> dict[str, Any]:
    targets, centers = unique_centers(runtime)
    jobs = [{"state": state_by_uid(runtime, sid), "eta": eta, "chain": "orthoflow3",
             "seeds": STANDARD_SEEDS, "metadata": {"cross_state_center": list(eta)}}
            for eta in centers for sid in targets]
    return execute_batch(runtime, "cross_state_q16", jobs,
                         shard_index=shard_index, num_shards=num_shards,
                         exact_robust_15of16=True)


def finalize_coverage(runtime: ScenarioRuntime) -> dict[str, Any]:
    targets, centers = unique_centers(runtime)
    rows = []
    memberships = {}
    for index, eta in enumerate(centers):
        robust = [sid for sid in targets
                  if robust_evidence(runtime, state_by_uid(runtime, sid), eta,
                                     persist_stage="cross_state_q16")["robust"]]
        memberships[index] = set(robust)
        rows.append({"center_index": index, "eta": list(eta), "robust_states": robust,
                     "coverage_count": len(robust),
                     "coverage_fraction": len(robust) / max(len(targets), 1)})
    overlaps = []
    for left_index, left in enumerate(targets):
        left_set = {index for index, members in memberships.items() if left in members}
        for right in targets[left_index + 1:]:
            right_set = {index for index, members in memberships.items() if right in members}
            union = left_set | right_set
            overlaps.append({"left": left, "right": right,
                             "jaccard": len(left_set & right_set) / len(union) if union else 0.0})
    result = {"centers": rows, "overlaps": overlaps}
    json_dump(runtime.output / "cross_state_coverage.json", result)
    return result


def controls_only(runtime: ScenarioRuntime, *, shard_index: int, num_shards: int) -> dict[str, Any]:
    _, centers = unique_centers(runtime)
    controls = json.loads((runtime.output / "formal_safety_summary.json").read_text())["success_control_state_uids"]
    jobs = [{"state": state_by_uid(runtime, sid), "eta": eta, "chain": "orthoflow3",
             "seeds": CONTROL_SEEDS, "metadata": {"success_control_center": list(eta)}}
            for eta in centers for sid in controls]
    return execute_batch(runtime, "success_controls_q8", jobs,
                         shard_index=shard_index, num_shards=num_shards)


def finalize_controls(runtime: ScenarioRuntime) -> list[dict[str, Any]]:
    _, centers = unique_centers(runtime)
    controls = json.loads((runtime.output / "formal_safety_summary.json").read_text())["success_control_state_uids"]
    result = []
    for index, eta in enumerate(centers):
        successes = trials = numerical_incomplete = 0
        for sid in controls:
            rows, missing = cached_rows(runtime, state_by_uid(runtime, sid), eta,
                                        "orthoflow3", CONTROL_SEEDS)
            successes += sum(int(row["success"]) for row in rows.values())
            trials += len(rows)
            numerical_incomplete += len(missing)
        planned = len(controls) * len(CONTROL_SEEDS)
        result.append({"center_index": index, "eta": list(eta), "successes": successes,
                       "trials": trials, "planned_trials": planned,
                       "numerical_incomplete": numerical_incomplete,
                       "preservation_probability": successes / max(trials, 1)})
    json_dump(runtime.output / "success_control_preservation.json", result)
    return result


def mechanism_rollouts(runtime: ScenarioRuntime, *, shard_index: int,
                       num_shards: int) -> dict[str, Any]:
    existence = json.loads((runtime.output / "robust_existence.json").read_text())
    jobs = []
    for sid, center in existence["robust_centers"].items():
        state = state_by_uid(runtime, sid)
        for label, eta in (("eta_zero", (0.0, 0.0, 0.0)), ("eta_center", center["eta"])):
            jobs.append({"state": state, "eta": eta, "chain": "orthoflow3", "seeds": [16],
                         "metadata": {"mechanism_pair": True, "mechanism_arm": label,
                                      "center_source_state": sid}})
    return execute_batch(runtime, "mechanism", jobs, save_trace=True,
                         shard_index=shard_index, num_shards=num_shards)


def _interval_statistics(mask: np.ndarray) -> dict[str, float | int]:
    runs = []
    start = None
    for index, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append(index - start); start = None
    return {"count": len(runs), "total_steps": int(sum(runs)),
            "maximum_steps": int(max(runs, default=0))}


def _trace_metrics(runtime: ScenarioRuntime, row: dict[str, Any]) -> dict[str, Any]:
    trace = row["trace"]
    positions = np.asarray(trace["positions"], dtype=np.float64)
    velocities = np.diff(positions, axis=0) / runtime.config.dt
    speeds = np.linalg.norm(velocities, axis=-1)
    progress = np.asarray(trace["goal_error_progress"], dtype=np.float64)
    waiting = float(np.sum(speeds < 0.05) * runtime.config.dt / 4.0)
    low_progress = _interval_statistics(np.abs(progress) < 1e-3)
    result = {"outcome": row["outcome"], "episode_length": row["episode_length"],
              "completion_time": row["episode_length"] * runtime.config.dt,
              "mean_waiting_seconds_per_agent": waiting,
              "near_zero_progress_intervals": low_progress,
              "mode_signature": row["mode_signature"]}
    if runtime.kind == "four":
        entry = []
        for agent in range(4):
            hits = np.flatnonzero(np.linalg.norm(positions[:, agent], axis=1) <= 0.70)
            entry.append(None if not len(hits) else int(hits[0]))
        all_slow = np.all(speeds < 0.08, axis=1)
        outside_center = np.all(np.linalg.norm(positions[:-1], axis=-1) > 0.70, axis=1)
        cyclic_proxy = _interval_statistics(all_slow & outside_center)
        result.update(center_entry_steps=entry,
                      center_entry_span_steps=(None if any(x is None for x in entry) else max(entry) - min(entry)),
                      cyclic_yielding_proxy=cyclic_proxy)
    else:
        angle = np.unwrap(np.arctan2(positions[:, :, 1], positions[:, :, 0]), axis=0)
        angular_step = np.diff(angle, axis=0)
        signed = np.sign(angular_step)
        signed[np.abs(angular_step) < 1e-4] = 0
        reversals = []
        for agent in range(4):
            nz = signed[:, agent][signed[:, agent] != 0]
            reversals.append(int(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0)
        net = angle[-1] - angle[0]
        radii = np.linalg.norm(positions, axis=-1)
        head_on = np.zeros(len(velocities), dtype=bool)
        for i in range(4):
            for j in range(i + 1, 4):
                close = np.linalg.norm(positions[:-1, i] - positions[:-1, j], axis=1) < 0.65
                approaching = np.sum((positions[:-1, i] - positions[:-1, j]) *
                                     (velocities[:, i] - velocities[:, j]), axis=1) < 0
                head_on |= close & approaching
        result.update(net_angular_progress=net.tolist(),
                      circulation_signs=np.sign(net).astype(int).tolist(),
                      circulation_reversals=reversals,
                      radial_deviation_mean=float(np.mean(np.abs(radii - radii[0]))),
                      radial_deviation_max=float(np.max(np.abs(radii - radii[0]))),
                      head_on_interaction_seconds=float(np.sum(head_on) * runtime.config.dt))
    return result


def finalize_mechanism(runtime: ScenarioRuntime) -> dict[str, Any]:
    rows = [row for row in load_raw(runtime) if row.get("mechanism_pair") and
            not row.get("numerical_failure") and "trace" in row]
    latest = {(row["state_uid"], row["mechanism_arm"]): row for row in rows}
    existence = json.loads((runtime.output / "robust_existence.json").read_text())
    pairs = []
    incomplete = []
    for sid in existence["robust_centers"]:
        missing = [arm for arm in ("eta_zero", "eta_center") if (sid, arm) not in latest]
        if missing:
            incomplete.append({"state_uid": sid, "missing_valid_arms": missing,
                               "reason": "NUMERICAL_INCOMPLETE"})
            continue
        zero = _trace_metrics(runtime, latest[(sid, "eta_zero")])
        center = _trace_metrics(runtime, latest[(sid, "eta_center")])
        pairs.append({"state_uid": sid, "eta_zero": zero, "eta_center": center,
                      "completion_time_change": center["completion_time"] - zero["completion_time"],
                      "waiting_change": center["mean_waiting_seconds_per_agent"] - zero["mean_waiting_seconds_per_agent"]})
    result = {"diagnostic_seed": 16, "pairs": pairs,
              "numerical_incomplete_pairs": incomplete,
              "planned_pairs": len(existence["robust_centers"])}
    json_dump(runtime.output / "behavioral_mechanism.json", result)
    return result


def database_integrity(runtime: ScenarioRuntime) -> dict[str, Any]:
    sealed = 0
    with connect() as con:
        for path in sorted((runtime.output / "raw").glob("*.jsonl")):
            source_uid = uid("src", {"path": str(path.resolve())})
            if con.execute("SELECT 1 FROM source_file WHERE source_uid=?", (source_uid,)).fetchone():
                con.execute("UPDATE source_file SET sha256=? WHERE source_uid=?",
                            (sha256_file(path), source_uid)); sealed += 1
        con.commit()
        foreign = [dict(row) for row in con.execute("PRAGMA foreign_key_check")]
        live = con.execute("SELECT COUNT(*) FROM source_file WHERE sha256 LIKE 'LIVE_%'").fetchone()[0]
        duplicate = con.execute("""SELECT COUNT(*) FROM (
            SELECT state_uid,eta_uid,controller_uid,seed_key,COUNT(*) n FROM rollout
            GROUP BY state_uid,eta_uid,controller_uid,seed_key HAVING n>1)""").fetchone()[0]
        ctl = runtime.controllers["orthoflow3"]["uid"]
        counts = dict(con.execute("""SELECT COUNT(*) records,SUM(success) successes,
            SUM(collision) collisions,SUM(numerical_failure) numerical_failures
            FROM rollout WHERE controller_uid=? AND conflict_quarantined=0""", (ctl,)).fetchone())
    result = {"sealed_scenario_sources": sealed, "global_live_source_hashes": live,
              "foreign_key_violations": foreign, "duplicate_exact_keys": duplicate,
              "orthoflow3_controller": ctl, "controller_counts": counts,
              "status": "PASS" if not foreign and not duplicate else "FAIL"}
    json_dump(runtime.output / "database_integrity.json", result)
    return result


def early_stop_efficiency(runtime: ScenarioRuntime) -> dict[str, Any]:
    """Summarize exact logical stopping; this is implementation accounting only."""
    with connect(True) as con:
        rows = [dict(row) for row in con.execute(
            """SELECT evaluated_seed_count,n_success,n_failure,stop_reason,robust,stage
            FROM robust_logical_decision
            WHERE controller_uid=? AND protocol_name=? AND experiment_uid=?""",
            (runtime.controllers["orthoflow3"]["uid"], ROBUST_PROTOCOL,
             runtime.experiment_uid),
        )]
    unresolved: set[tuple[str, str]] = set()
    # Reconstruct the three frozen robust-membership request families and
    # explicitly count tuples that remain unclassifiable only because one or
    # more canonical seed executions were numerical-invalid.
    for label in ("primary", "secondary"):
        path = runtime.output / f"{label}_search.json"
        if not path.exists():
            continue
        search = json.loads(path.read_text())
        points = np.asarray(designs()[label], dtype=np.float64)
        for sid, result in search.items():
            for index in result["promoted_indices"]:
                evidence = robust_evidence(runtime, state_by_uid(runtime, sid),
                                           points[int(index)])
                if evidence["stop_reason"] == "NUMERICAL_INCOMPLETE_NOT_CERTIFIED":
                    unresolved.add((sid, eta_identity(points[int(index)])[0]))
    if (runtime.output / "robust_existence.json").exists():
        existence, local_points = local_context(runtime)
        for sid, points in local_points.items():
            for index in LOCAL_PROMOTE:
                evidence = robust_evidence(runtime, state_by_uid(runtime, sid), points[index])
                if evidence["stop_reason"] == "NUMERICAL_INCOMPLETE_NOT_CERTIFIED":
                    unresolved.add((sid, eta_identity(points[index])[0]))
        targets, centers = unique_centers(runtime)
        for sid in targets:
            for eta in centers:
                evidence = robust_evidence(runtime, state_by_uid(runtime, sid), eta)
                if evidence["stop_reason"] == "NUMERICAL_INCOMPLETE_NOT_CERTIFIED":
                    unresolved.add((sid, eta_identity(eta)[0]))
    actual = sum(row["evaluated_seed_count"] for row in rows)
    without = len(rows) * len(STANDARD_SEEDS)
    result = {
        "protocol": ROBUST_PROTOCOL,
        "criterion": "successes >= 15/16",
        "tuple_count": len(rows),
        "numerical_incomplete_not_classified": len(unresolved),
        "total_unique_requested_tuples_including_numerical_incomplete": len(rows) + len(unresolved),
        "fully_evaluated_to_16": sum(row["evaluated_seed_count"] == 16 for row in rows),
        "rejected_after_4": sum(
            row["stop_reason"] == "ROBUST_IMPOSSIBLE_2_FAILURES" and
            row["evaluated_seed_count"] <= 4 for row in rows),
        "rejected_later_after_second_failure": sum(
            row["stop_reason"] == "ROBUST_IMPOSSIBLE_2_FAILURES" and
            row["evaluated_seed_count"] > 4 for row in rows),
        "confirmed_at_15_successes": sum(
            row["stop_reason"] == "ROBUST_CONFIRMED_15_SUCCESSES" for row in rows),
        "canonical_seed_rollouts_evaluated": actual,
        "canonical_seed_rollouts_without_early_stop": without,
        "canonical_seed_rollouts_saved": without - actual,
        "compute_saved_fraction": (without - actual) / without if without else 0.0,
        "stop_reason_counts": dict(Counter(row["stop_reason"] for row in rows)),
        "classification_equivalence": (
            "EXACT: two observed failures make >=15/16 impossible; fifteen observed "
            "successes make >=15/16 certain. Unrun seeds are neither stored nor imputed."
        ),
        "note": "Efficiency accounting only; numerical-incomplete tuples are excluded from the saved-compute denominator and are not imputed as failures.",
    }
    json_dump(runtime.output / "early_stop_efficiency.json", result)
    return result


def finalize(runtime: ScenarioRuntime) -> dict[str, Any]:
    formal = json.loads((runtime.output / "formal_safety_summary.json").read_text())
    existence = json.loads((runtime.output / "robust_existence.json").read_text())
    local = json.loads((runtime.output / "local_basin.json").read_text())
    coverage = json.loads((runtime.output / "cross_state_coverage.json").read_text())
    preservation = json.loads((runtime.output / "success_control_preservation.json").read_text())
    n = len(existence["targets"])
    nr = len(existence["robust_centers"])
    fraction = nr / max(n, 1)
    best = max((row["coverage_fraction"] for row in coverage["centers"]), default=0.0)
    local_values = [row["robust_fraction"] for row in local.values()]
    if fraction >= 0.5 and best >= 0.25:
        classification = "3D_ROBUST_SHARED"
    elif fraction >= 0.5:
        classification = "3D_ROBUST_STATE_DEPENDENT"
    elif fraction >= 0.2:
        classification = "3D_PARTIAL"
    else:
        classification = "3D_INSUFFICIENT"
    jaccards = [row["jaccard"] for row in coverage["overlaps"]]
    efficiency = early_stop_efficiency(runtime)
    result = {
        "scenario": runtime.name,
        "formal_hard_safety": formal["hard_safety"],
        "safe_failures": n,
        "robust_states": nr,
        "robust_existence_fraction": fraction,
        "best_center_coverage": best,
        "median_center_coverage": float(np.median([r["coverage_fraction"] for r in coverage["centers"]])) if coverage["centers"] else 0.0,
        "coverage_threshold_counts": {str(threshold): sum(r["coverage_fraction"] >= threshold for r in coverage["centers"])
                                      for threshold in (0.10, 0.25, 0.50, 0.75, 1.0)},
        "median_local_robust_fraction": float(np.median(local_values)) if local_values else 0.0,
        "median_basin_jaccard": float(np.median(jaccards)) if jaccards else 0.0,
        "zero_overlap_pair_fraction": float(np.mean(np.asarray(jaccards) == 0)) if jaccards else 0.0,
        "median_success_control_preservation": float(np.median([r["preservation_probability"] for r in preservation])) if preservation else 0.0,
        "classification": classification,
        "early_stop_efficiency": efficiency,
        "hashes": {"checkpoint": runtime.checkpoint_sha, "environment": runtime.env_hash,
                   "orthoflow3": runtime.orthoflow_hash, "safety": runtime.safety_hash},
    }
    json_dump(runtime.output / "SUMMARY.json", result)
    return result


def setup(runtime: ScenarioRuntime) -> dict[str, Any]:
    frozen = {
        "scenario": runtime.name, "checkpoint": str(runtime.spec["checkpoint"]),
        "dataset": str(runtime.spec["dataset"]), "checkpoint_sha256": runtime.checkpoint_sha,
        "environment_sha256": runtime.env_hash, "orthoflow3_sha256": runtime.orthoflow_hash,
        "safety_projection_sha256": runtime.safety_hash,
        "environment_fingerprint": runtime.manifest["environment_fingerprint"],
        "horizon": runtime.config.max_steps, "dt": runtime.config.dt,
        "frozen_test_states": len(runtime.states), "evaluation_seed": FROZEN_EVAL_SEEDS[runtime.name],
        "eta_domain": {"lower": DOMAIN_LOW.tolist(), "upper": DOMAIN_HIGH.tolist()},
        "orthoflow3_scale": runtime.basis.metadata.scale,
        "standard_seed_policy": list(STANDARD_SEEDS), "future_root": FUTURE_ROOT,
        "selection_audit": runtime.selection_audit,
    }
    json_dump(runtime.output / "FROZEN_PROTOCOL.json", frozen)
    frozen_design = designs()
    json_dump(runtime.output / "frozen_eta_design.json", frozen_design)
    # Canonical eta metadata is shared globally; repair any earlier live-row
    # placeholder normalization without altering exact float64 eta identity.
    with connect() as con:
        for label in ("primary", "secondary"):
            for value in frozen_design[label]:
                eta_uid, eta, eta_hex = eta_identity(value)
                con.execute(
                    """INSERT INTO eta(eta_uid,eta1,eta2,eta3,normalized_eta_json,canonical_hex)
                    VALUES(?,?,?,?,?,?) ON CONFLICT(eta_uid) DO UPDATE SET
                    normalized_eta_json=excluded.normalized_eta_json""",
                    (eta_uid, *eta, canonical(normalized_eta(eta).tolist()), eta_hex),
                )
        con.commit()
    return frozen


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
    parser.add_argument("--stage", choices=(
        "setup", "formal", "primary-q4", "primary-plan", "primary-q16", "primary-finalize",
        "secondary-q4", "secondary-plan", "secondary-q16", "secondary-finalize",
        "existence-finalize", "search", "local-q8", "local-q16", "local-finalize",
        "coverage-q16", "coverage-finalize", "controls-q8", "controls-finalize",
        "mechanism", "mechanism-finalize",
        "db-audit",
        "local-coverage", "finalize", "all"), required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)
    runtime = ScenarioRuntime(args.scenario)
    setup(runtime)
    if args.stage in ("setup",):
        return
    if args.stage in ("formal", "all"):
        print(json.dumps(execute_batch(runtime, "formal", formal_jobs(runtime)), indent=2))
        summarize_formal(runtime)
    if args.stage == "primary-q4":
        print(json.dumps(q4_only(runtime, "primary", shard_index=args.shard_index,
                                 num_shards=args.num_shards), indent=2))
    if args.stage == "secondary-q4":
        print(json.dumps(q4_only(runtime, "secondary", shard_index=args.shard_index,
                                 num_shards=args.num_shards), indent=2))
    if args.stage in ("primary-plan", "secondary-plan"):
        print(json.dumps(plan_promotions(runtime, args.stage.split("-")[0]), indent=2))
    if args.stage in ("primary-q16", "secondary-q16"):
        print(json.dumps(promotion_q16_only(runtime, args.stage.split("-")[0],
                                            shard_index=args.shard_index,
                                            num_shards=args.num_shards), indent=2))
    if args.stage in ("primary-finalize", "secondary-finalize"):
        result = finalize_screen(runtime, args.stage.split("-")[0])
        print(json.dumps({"states": len(result),
                          "robust": sum(bool(x["robust_indices"]) for x in result.values())}, indent=2))
    if args.stage == "existence-finalize":
        print(json.dumps(finalize_existence(runtime), indent=2))
    if args.stage in ("local-q8", "local-q16"):
        print(json.dumps(local_stage_only(runtime, args.stage == "local-q16",
                                          shard_index=args.shard_index,
                                          num_shards=args.num_shards), indent=2))
    if args.stage == "local-finalize":
        print(json.dumps(finalize_local(runtime), indent=2))
    if args.stage == "coverage-q16":
        print(json.dumps(coverage_only(runtime, shard_index=args.shard_index,
                                       num_shards=args.num_shards), indent=2))
    if args.stage == "coverage-finalize":
        print(json.dumps(finalize_coverage(runtime), indent=2))
    if args.stage == "controls-q8":
        print(json.dumps(controls_only(runtime, shard_index=args.shard_index,
                                       num_shards=args.num_shards), indent=2))
    if args.stage == "controls-finalize":
        print(json.dumps(finalize_controls(runtime), indent=2))
    if args.stage == "mechanism":
        print(json.dumps(mechanism_rollouts(runtime, shard_index=args.shard_index,
                                            num_shards=args.num_shards), indent=2))
    if args.stage == "mechanism-finalize":
        print(json.dumps(finalize_mechanism(runtime), indent=2))
    if args.stage == "db-audit":
        print(json.dumps(database_integrity(runtime), indent=2))
    if args.stage in ("search", "all"):
        eta_search(runtime)
    if args.stage in ("local-coverage", "all"):
        local_and_coverage(runtime)
    if args.stage in ("finalize", "all"):
        print(json.dumps(finalize(runtime), indent=2))


if __name__ == "__main__":
    main()
