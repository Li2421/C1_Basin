"""Resumable exact Gap1 OrthoFlow3 continuations for a frozen state-eta pair.

The optional logical-robust stage uses the project's exact B15 stopping rule:
two observed failures prove non-robustness, while 15 successes prove robust
success. The full-Q16 stage evaluates every remaining canonical future seed.
Unrun seeds are always marked unobserved, never counted as failures.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import jax
import numpy as np

from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import (
    CertifiedHardSafetyFilter,
)
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from new_benchmark_common.safety_eta3 import FUTURE_ROOT, STANDARD_SEEDS, state_token
from shared_control.basis_families import get_basis_family
from shared_control.hard_projection import CBFSolverError, HardProjectionConfig
from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.rollout_db import eta_identity, uid

from .environment import BottleneckEnv
from .observation import policy_observation_competence
from .scenario import Config


DEFAULT_DESIGN = Path("datasets/gap1_eta_scaling_v2")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def bounded(action: np.ndarray, max_speed: float) -> np.ndarray:
    norm = np.linalg.norm(action, axis=1, keepdims=True)
    return action * np.minimum(1., max_speed / np.maximum(norm, 1e-30))


def rollout(state: dict, eta: np.ndarray, future_index: int, *,
            agent, config: Config, projector: CertifiedHardSafetyFilter,
            basis, identifiers: dict) -> dict:
    env = BottleneckEnv(replace(config, seed=state["episode_seed"],
                                split=state["physical_split"]))
    env.reset(np.asarray(state["positions"], dtype=np.float64),
              np.asarray(state["goals"], dtype=np.float64))
    root = jax.random.fold_in(jax.random.PRNGKey(FUTURE_ROOT),
                              state_token(state["state_uid"]))
    root = jax.random.fold_in(root, int(future_index))
    min_wall = math.inf
    min_agent = math.inf
    sum_first_correction = 0.
    sum_second_correction = 0.
    first_active_steps = 0
    second_active_steps = 0
    numerical = None
    for step in range(config.max_steps):
        observation = policy_observation_competence(env)
        flow = np.asarray(sample_bounded_actions(
            agent, observation[None], jax.random.fold_in(root, step))[0],
            dtype=np.float64)
        flow = bounded(flow, config.max_speed)
        reference = flow.copy()
        reference[np.linalg.norm(env.goals-env.positions,axis=1)
                  <= config.goal_tolerance] = 0.
        snapshot = env.snapshot()
        try:
            first = projector(snapshot, reference)
            safe = np.asarray(first.velocity, dtype=np.float64)
            correction = basis.compute(env.positions, env.goals, safe,
                                       config.max_speed).correction(eta)
            target = safe + correction
            second = projector(snapshot, target)
            executed = np.asarray(second.velocity, dtype=np.float64)
            _, done, info = env.step(executed, diagnose_stalls=False)
        except (CBFSolverError, ValueError, FloatingPointError) as exc:
            numerical = {"step": step, "type": type(exc).__name__,
                         "message": str(exc)}
            break
        first_norm = float(np.linalg.norm(safe - reference))
        second_norm = float(np.linalg.norm(executed - target))
        sum_first_correction += first_norm
        sum_second_correction += second_norm
        first_active_steps += first_norm > projector.config.intervention_tol
        second_active_steps += second_norm > projector.config.intervention_tol
        min_wall = min(min_wall, float(info["min_swept_wall_clearance"]))
        min_agent = min(min_agent, float(info["min_swept_agent_clearance"]))
        if done:
            break
    outcome = "numerical_failure" if numerical else env.termination
    if outcome == "running":
        outcome = "timeout"
    success = bool(outcome == "success" and not env.collided)
    steps = env.step_count
    remaining = np.linalg.norm(env.positions-env.goals,axis=1)
    row = {
        "cache_identity_schema": "registered_exact_uid_v1",
        "scenario": f"Gap1_N{config.num_agents}",
        "scenario_uid": identifiers["scenario_uid"],
        "state_id": state["state_id"],
        "state_uid": state["state_uid"],
        "initial_positions": state["positions"],
        "initial_velocities": state["velocities"],
        "goals": state["goals"],
        "eta": eta.tolist(),
        "eta_uid": eta_identity(eta)[0],
        "controller_uid": identifiers["controller_uid"],
        "flow_checkpoint_sha256": identifiers["checkpoint_sha256"],
        "future_root_seed": FUTURE_ROOT,
        "future_index": int(future_index),
        "success": success,
        "deadlock": False,
        "timeout": outcome == "timeout",
        "collision": bool(env.collided),
        "numerical_failure": numerical is not None,
        "numerical_error": numerical,
        "outcome": outcome,
        "termination": outcome,
        "episode_length": int(steps),
        "final_goal_fraction": float(np.mean(remaining <= config.goal_tolerance)),
        "final_mean_goal_distance": float(np.mean(remaining)),
        "minimum_wall_clearance": None if not np.isfinite(min_wall) else min_wall,
        "minimum_agent_clearance": None if not np.isfinite(min_agent) else min_agent,
        "first_projection_active_fraction": first_active_steps/max(steps,1),
        "second_projection_active_fraction": second_active_steps/max(steps,1),
        "mean_first_projection_norm": sum_first_correction/max(steps,1),
        "mean_second_projection_norm": sum_second_correction/max(steps,1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return row


def logical_decision(seeds: dict[str, dict]) -> dict | None:
    observed = [seeds[str(s)] for s in STANDARD_SEEDS if str(s) in seeds]
    valid = [r for r in observed if not r["numerical_failure"]]
    success = sum(bool(r["success"]) for r in valid)
    failure = len(valid) - success
    if len(valid) == 16:
        reason = "FULL_16_EVALUATED"
    elif failure >= 2:
        reason = "ROBUST_IMPOSSIBLE_2_FAILURES"
    elif success >= 15:
        reason = "ROBUST_CONFIRMED_15_SUCCESSES"
    else:
        return None
    return {
        "robust": success >= 15,
        "stop_reason": reason,
        "n_observed_valid": len(valid),
        "n_success": success,
        "n_failure": failure,
        "observed_seed_indices": [s for s in STANDARD_SEEDS if str(s) in seeds],
        "unrun_seed_indices": [s for s in STANDARD_SEEDS if str(s) not in seeds],
        "unrun_seeds_are_not_failures": True,
    }


def run_pair(design_dir: Path, n: int, pair_index: int, stage: str) -> dict:
    design = json.loads((design_dir/"design_manifest.json").read_text())
    pool = json.loads((design_dir/"eta_pool.json").read_text())
    states = [s for s in design["states"] if s["N"] == n]
    if n not in (2,10,50) or not 0 <= pair_index < len(states)*len(pool["eta"]):
        raise ValueError("invalid N or pair index")
    state = states[pair_index // len(pool["eta"])]
    eta_index = pair_index % len(pool["eta"])
    eta = np.asarray(pool["eta"][eta_index], dtype=np.float64)
    spec = design["scenario_info"][str(n)]
    checkpoint = Path(spec["checkpoint"])
    if file_sha(checkpoint) != spec["controller"]["flow_checkpoint_sha256"]:
        raise RuntimeError("frozen Flow checkpoint hash changed")
    frozen_sources = {
        "flow_sampler_source_sha256": Path("new_benchmark_common/macflow.py"),
        "observation_source_sha256": Path("bottleneck_family/observation.py"),
        "basis_source_sha256": Path("shared_control/basis_families.py"),
        "basis_implementation_sha256": Path("shared_control/diagnostic_corrector.py"),
        "safety_source_sha256": Path("shared_control/hard_projection.py"),
        "safety_wrapper_sha256": Path(
            "diagnostics/double_bottleneck_eta3_basin/tools/exact_projection_retry.py"),
        "environment_source_sha256": Path("bottleneck_family/environment.py"),
    }
    for field, source in frozen_sources.items():
        if file_sha(source) != spec["controller"][field]:
            raise RuntimeError(f"frozen eta code hash changed: {source}")
    source_manifest = Path(spec["source_manifest"])
    if file_sha(source_manifest) != spec["source_manifest_sha256"]:
        raise RuntimeError("source geometry manifest hash changed")
    agent, _ = load_checkpoint(
        checkpoint,
        expected_environment_fingerprint=json.loads(source_manifest.read_text())[
            "environment_fingerprint"],
    )
    config = Config(**spec["config"])
    projector = CertifiedHardSafetyFilter(HardProjectionConfig())
    basis = get_basis_family("orthoflow3")
    identifiers = {"scenario_uid": spec["scenario_uid"],
                   "controller_uid": spec["controller_uid"],
                   "checkpoint_sha256": spec["controller"]["flow_checkpoint_sha256"]}
    pair_dir = design_dir/f"rollouts/n{n}/pair_{pair_index:04d}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    result_path = pair_dir/"result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["state_uid"] != state["state_uid"] or result["eta_uid"] != eta_identity(eta)[0]:
            raise RuntimeError("resumption identity mismatch")
    else:
        result = {
            "schema": "gap1_eta_state_pair_rollouts_v1",
            "N": n, "pair_index": pair_index,
            "state_id": state["state_id"], "state_uid": state["state_uid"],
            "state_split": state["split"],
            "eta_index": eta_index, "eta": eta.tolist(),
            "eta_uid": eta_identity(eta)[0],
            "controller_uid": spec["controller_uid"],
            "flow_checkpoint_sha256": identifiers["checkpoint_sha256"],
            "seeds": {},
        }
        dump_atomic(result_path, result)
    before = set(result["seeds"])
    for seed in STANDARD_SEEDS:
        if stage == "logical_robust" and logical_decision(result["seeds"]) is not None:
            break
        key = str(seed)
        if key in result["seeds"]:
            continue
        attempts = []
        for attempt in range(4):
            row = rollout(state,eta,seed,agent=agent,config=config,
                          projector=projector,basis=basis,identifiers=identifiers)
            row["execution_attempt"] = attempt
            attempts.append(row)
            if not row["numerical_failure"]:
                break
        result["seeds"][key] = attempts[-1]
        if len(attempts) > 1:
            result.setdefault("numerical_attempts", {})[key] = attempts[:-1]
        dump_atomic(result_path, result)
        print(json.dumps({"N": n, "pair": pair_index, "seed": seed,
                          "success": row["success"], "outcome": row["outcome"],
                          "steps": row["episode_length"]}), flush=True)
    decision = logical_decision(result["seeds"])
    if stage == "logical_robust" and decision is None:
        raise RuntimeError("logical robust decision not established")
    if stage == "full_q16" and len(result["seeds"]) != 16:
        raise RuntimeError("full Q16 incomplete")
    if stage == "full_q16" and all(not row["numerical_failure"]
                                   for row in result["seeds"].values()):
        result["Q16_success_count"] = sum(row["success"] for row in result["seeds"].values())
        result["empirical_Q16"] = result["Q16_success_count"]/16
    result["logical_robust_decision"] = decision
    dump_atomic(result_path, result)
    # A preempted worker can have committed seed results before writing its
    # journal. Re-journal every observed seed when this stage has no marker;
    # the exact-key merger deduplicates rows repeated across stages.
    new_rows = [result["seeds"][key] for key in sorted(result["seeds"], key=int)]
    marker = pair_dir/f"{stage}_journal_path.txt"
    if new_rows and not marker.exists():
        experiment_uid = uid("exp", {"path": str(design_dir.resolve())})
        journal = append_journal(new_rows, experiment_uid,
                                 worker_id=f"gap1_n{n}_pair{pair_index:04d}_{stage}")
        marker.write_text(str(journal)+"\n")
    return {"state_id": state["state_id"], "eta_index": eta_index,
            "pair_index": pair_index, "observed_seeds": len(result["seeds"]),
            "new_seeds": len(new_rows), "decision": decision,
            "result": str(result_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--agents", type=int, choices=(2,10,50), required=True)
    parser.add_argument("--pair-index", type=int, required=True)
    parser.add_argument("--stage", choices=("logical_robust","full_q16"), required=True)
    args = parser.parse_args()
    print(json.dumps(run_pair(args.design,args.agents,args.pair_index,args.stage),indent=2),flush=True)


if __name__ == "__main__":
    main()
