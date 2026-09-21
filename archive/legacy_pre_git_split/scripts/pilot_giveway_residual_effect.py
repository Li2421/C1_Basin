#!/usr/bin/env python3
"""Fixed GiveWay residual-effect pilot (exploratory, not formal Stage 3--5)."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.stats import beta as beta_distribution

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from single_integrator.c1.direction_a.policy import DirectionADistribution
from single_integrator.c1.direction_a.randomness import IndexedRandomTape, SOURCES
from single_integrator.c1.direction_a.rollout import (
    DirectionARolloutError,
    FrozenDirectionAController,
    _terminal_label,
    validate_projected_action,
)
from single_integrator.c1.differentiable_rollout import bounded_nominal
from single_integrator.cbf import CBFConfig, CBFSolverError, barrier_constraints, project_velocity
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.evaluate import load_policy


SCHEMA = "giveway_residual_effect_pilot_v1"
SPLIT = "giveway_residual_effect_pilot"
ROOT_SEED = 20260921
N_PAIRS = 64
SCENARIO_ID = 52
CHECKPOINT = ROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
CHECKPOINT_CONFIG = ROOT / "results/c1_dataset_extended_h120_seed0_baselines/config.json"
SCENE_CONFIG = ROOT / "results/c1_dataset_paired_seed0_baselines/config.json"
SCENE_SUMMARY = ROOT / "results/c1_dataset_paired_seed0_baselines/mac_cbf/summary.json"
SCENE_TRACE = ROOT / "results/c1_dataset_paired_seed0_baselines/mac_cbf/rollout_0052.npz"

FROZEN_SOURCES = (
    "single_integrator/c1/direction_a/policy.py",
    "single_integrator/c1/direction_a/randomness.py",
    "single_integrator/c1/direction_a/rollout.py",
    "single_integrator/c1/differentiable_rollout.py",
    "single_integrator/cbf.py",
    "single_integrator/environment.py",
    "single_integrator/diagnostics/stalled_outcomes.py",
    "single_integrator/outcomes.py",
    "single_integrator/evaluate.py",
    "scripts/run_giveway_residual_effect_pilot.sbatch",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def archived_assets():
    for path in (CHECKPOINT, CHECKPOINT_CONFIG, SCENE_CONFIG, SCENE_SUMMARY, SCENE_TRACE):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_cfg = json.loads(CHECKPOINT_CONFIG.read_text())
    scene_cfg = json.loads(SCENE_CONFIG.read_text())
    summaries = json.loads(SCENE_SUMMARY.read_text())["rollouts"]
    deadlocks = [row for row in summaries if row.get("deadlock")]
    if not deadlocks or int(deadlocks[0]["rollout_id"]) != SCENARIO_ID:
        raise RuntimeError("archived earliest Safety deadlock is no longer rollout 52")
    selected = deadlocks[0]
    with np.load(SCENE_TRACE, allow_pickle=False) as trace:
        initial = np.asarray(trace["initial_positions"], dtype=np.float64)
        first_before = np.asarray(trace["positions_before"][0], dtype=np.float64)
    if not np.array_equal(initial, first_before):
        raise RuntimeError("archived scene trace does not begin at its recorded t=0 state")
    if not np.array_equal(initial, np.asarray(selected["initial_positions"], dtype=np.float64)):
        raise RuntimeError("archived summary and trace initial states disagree")
    expected_hash = checkpoint_cfg["checkpoint_sha256"]
    if sha256(CHECKPOINT) != expected_hash or scene_cfg["checkpoint_sha256"] != expected_hash:
        raise RuntimeError("selected checkpoint does not match archived Safety records")
    if int(scene_cfg["environment"]["max_steps"]) != 850:
        raise RuntimeError("scene source is not the original 850-step Safety evaluation")
    if not all(scene_cfg["environment"][key] for key in (
            "terminate_on_collision", "terminate_on_success", "terminate_on_deadlock")):
        raise RuntimeError("archived termination flags differ from the frozen diagnostic")
    return checkpoint_cfg, scene_cfg, selected, initial


def build_seed_schedule(path: Path) -> str:
    # This vectorized construction is algebraically identical to
    # IndexedRandomTape.key.  The public method deliberately rejects traced
    # integer indices, so the manifest generator spells out the same folds.
    split_u32 = int.from_bytes(hashlib.sha256(SPLIT.encode("utf-8")).digest()[:4], "little")
    base = jax.random.fold_in(jax.random.PRNGKey(ROOT_SEED), jnp.uint32(split_u32))
    base = jax.random.fold_in(base, jnp.uint32(SCENARIO_ID))
    pair_ids = jnp.arange(N_PAIRS, dtype=jnp.int32)
    times = jnp.arange(850, dtype=jnp.int32)
    arrays = {"pair_id": np.arange(N_PAIRS, dtype=np.int32),
              "time": np.arange(850, dtype=np.int32)}
    for source in ("flow_bc", "gate", "gaussian_residual", "environment"):
        tag = jnp.uint32(SOURCES[source])
        grid = jax.jit(jax.vmap(lambda continuation: jax.vmap(
            lambda step: jax.random.fold_in(jax.random.fold_in(
                jax.random.fold_in(base, jnp.uint32(continuation)),
                jnp.uint32(step)), tag)
        )(times)))(pair_ids)
        arrays[source + "_key"] = np.asarray(grid, dtype=np.uint32)
    evaluation_tag = jnp.uint32(SOURCES["evaluation_resampling"])
    evaluation = jax.jit(jax.vmap(lambda continuation: jax.random.fold_in(
        jax.random.fold_in(jax.random.fold_in(base, jnp.uint32(continuation)),
                           jnp.uint32(0)), evaluation_tag)))(pair_ids)
    arrays["evaluation_resampling_key"] = np.asarray(evaluation, dtype=np.uint32)
    np.savez_compressed(path, **arrays)
    return sha256(path)


def make_manifest(out: Path):
    checkpoint_cfg, scene_cfg, selected, initial = archived_assets()
    out.mkdir(parents=True, exist_ok=True)
    seed_path = out / "seed_schedule.npz"
    seed_hash = build_seed_schedule(seed_path) if not seed_path.exists() else sha256(seed_path)
    plant = Config(**scene_cfg["environment"])
    if plant.max_steps != 850 or plant.dt != 0.05:
        raise RuntimeError("frozen physical horizon mismatch")
    v_ref = float(plant.max_speed)
    sigma_min, sigma_max, sigma_init = .05*v_ref, .50*v_ref, .20*v_ref
    beta = -math.log(2.0)
    source_hashes = {name: sha256(ROOT/name) for name in FROZEN_SOURCES}
    script = Path(__file__).resolve()
    manifest = {
        "schema": SCHEMA,
        "status": "manifest_frozen_before_rollouts",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "name": "GiveWay residual-effect pilot",
            "classification": "independent exploratory diagnostic",
            "formal_stage_3_to_5": False,
            "method_pass_fail_allowed": False,
            "parameter_updates": 0,
            "outcome_adaptive_tuning": False,
        },
        "repository": {
            "root": str(ROOT),
            "git_work_tree": False,
            "commit": None,
            "provenance": "SHA256 manifest because repository root has no .git metadata",
        },
        "asset_selection": {
            "checkpoint_rule": "seed0 checkpoint referenced by the most recent archived Safety evaluation config",
            "checkpoint_config": str(CHECKPOINT_CONFIG),
            "checkpoint_config_mtime_ns": CHECKPOINT_CONFIG.stat().st_mtime_ns,
            "checkpoint_config_sha256": sha256(CHECKPOINT_CONFIG),
            "checkpoint_recorded_path": checkpoint_cfg["checkpoint"],
            "checkpoint_actual_path": str(CHECKPOINT),
            "checkpoint_sha256": sha256(CHECKPOINT),
            "scene_rule": "earliest rollout_id with Safety deadlock in the archived original 850-step seed0 evaluation",
            "scene_config": str(SCENE_CONFIG),
            "scene_config_sha256": sha256(SCENE_CONFIG),
            "scene_summary": str(SCENE_SUMMARY),
            "scene_summary_sha256": sha256(SCENE_SUMMARY),
            "scene_trace": str(SCENE_TRACE),
            "scene_trace_sha256": sha256(SCENE_TRACE),
            "archived_rollout_id": int(selected["rollout_id"]),
            "archived_outcome": selected["outcome"],
            "archived_first_deadlock_step": int(selected["first_deadlock_step"]),
            "initial_positions_t0": initial.tolist(),
        },
        "policy": {
            "name": "phi0_diag",
            "type": "constant-output special case of current joint-gate Gaussian policy",
            "mu_m_per_s": [0.0]*4,
            "p": .5,
            "alpha": 0.0,
            "v_ref_m_per_s": v_ref,
            "v_ref_rule": "minimum of the two identical per-agent physical speed limits in the archived environment",
            "sigma_min_m_per_s": sigma_min,
            "sigma_max_m_per_s": sigma_max,
            "sigma_init_m_per_s": sigma_init,
            "beta": beta,
            "gate": "one joint Bernoulli per physical timestep",
            "active_residual": "four-dimensional isotropic Gaussian",
            "residual_clipping": False,
        },
        "controller": {
            "S_safety": "u_exec = Pi_U(x)(u_FlowBC); raw residual is exactly zero",
            "A_phi0_diag": "u_safe = Pi_U(x)(u_FlowBC); u_exec = Pi_U(x)(u_safe+r)",
            "flow_parameters_frozen": True,
            "environment": asdict(plant),
            "cbf": scene_cfg["cbf"],
        },
        "randomness": {
            "root_seed": ROOT_SEED,
            "split": SPLIT,
            "scenario": SCENARIO_ID,
            "continuations": list(range(N_PAIRS)),
            "sources": SOURCES,
            "address": "fold_in(root, sha256(split)[0:4]_little, scenario, continuation, time, source_tag)",
            "paired_sources": ["flow_bc", "environment"],
            "experiment_only_sources": ["gate", "gaussian_residual"],
            "environment_is_deterministic": True,
            "environment_keys_reserved_but_not_consumed": True,
            "seed_schedule": str(seed_path),
            "seed_schedule_sha256": seed_hash,
        },
        "budget": {
            "groups": {"S_safety": N_PAIRS, "A_phi0_diag": N_PAIRS},
            "total_rollouts": 2*N_PAIRS,
            "max_actions_per_rollout": 850,
            "maximum_simulator_actions": 2*N_PAIRS*850,
            "horizon_seconds": 42.5,
            "no_optional_stopping_or_extension": True,
        },
        "metric": {
            "executed_correction_energy": "dt * sum_k ||u_exec,k-u_safe,k||_2^2",
            "same_state_and_flow_reference": True,
            "formal_J_def": False,
        },
        "source_sha256": source_hashes | {
            str(script.relative_to(ROOT)): sha256(script),
        },
        "specification_map": {
            "Flow_BC_sampling": "single_integrator/c1/differentiable_rollout.py:36",
            "joint_gate_Gaussian_sampling": "single_integrator/c1/direction_a/policy.py:84",
            "source_isolated_RNG": "single_integrator/c1/direction_a/randomness.py:27",
            "first_and_second_projection": "single_integrator/c1/direction_a/rollout.py:85",
            "Euclidean_projection": "single_integrator/cbf.py:96",
            "transition_and_monitor": "single_integrator/environment.py:170",
            "termination_priority": "single_integrator/c1/direction_a/rollout.py:52",
            "stalled_reclassification": "single_integrator/diagnostics/stalled_outcomes.py:12",
            "diagnostic_entry": "scripts/pilot_giveway_residual_effect.py",
        },
        "reproduce": {
            "prepare": "JAX_PLATFORMS=cpu .venv-c1/bin/python scripts/pilot_giveway_residual_effect.py --out results/giveway_residual_effect_pilot_v1 --prepare-only",
            "run": ".venv-c1/bin/python scripts/pilot_giveway_residual_effect.py --out results/giveway_residual_effect_pilot_v1",
        },
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        # Re-preparation may only refresh the creation timestamp when every
        # frozen experimental field is identical.
        old.pop("created_at_utc", None)
        check = dict(manifest); check.pop("created_at_utc", None)
        if old != check:
            raise RuntimeError("existing diagnostic manifest differs from requested frozen manifest")
    else:
        json_dump(manifest_path, manifest)
    return manifest


class ConstantDiagnosticPolicy:
    """Stateless phi0_diag output through the existing policy adapter."""

    beta = -math.log(2.0)

    @staticmethod
    def apply(params, h_s, s, observation):
        del params, s, observation
        count = h_s.shape[0]
        dtype = jnp.asarray(h_s).dtype
        return (jnp.zeros((count, 4), dtype=dtype),
                jnp.zeros((count,), dtype=dtype),
                jnp.full((count,), ConstantDiagnosticPolicy.beta, dtype=dtype))


def safety_action(controller, env, tape, continuation, time_index):
    """Original one-projection Safety arm with the shared Flow tape."""
    observation = env.observation()[None]
    noise = tape.flow_noise(SPLIT, SCENARIO_ID, continuation, time_index)[None]
    raw = np.asarray(controller._flow_sample(observation, noise))[0]
    nominal = np.asarray(bounded_nominal(jnp.asarray(raw), controller.plant.max_speed))
    A, lower, _ = barrier_constraints(env.snapshot(), controller.cbf)
    try:
        safe, status = project_velocity(nominal, A, lower,
                                        controller.plant.max_speed, controller.cbf)
    except (CBFSolverError, ValueError, FloatingPointError) as exc:
        raise DirectionARolloutError(f"first projection failure: {exc}") from exc
    audit = validate_projected_action(safe, A, lower, controller.plant.max_speed,
                                      status, controller.cbf)
    safe = np.asarray(safe, dtype=np.float64).reshape(4)
    return dict(raw_flow=raw, nominal=np.asarray(nominal).reshape(4), safe=safe,
                residual=np.zeros(4), applied=safe.copy(), gate=False,
                p=np.nan, sigma=np.nan, epsilon=np.zeros(4),
                first_projection=audit, second_projection=None)


def trace_arrays(records):
    numeric = (
        "positions_before", "positions_after", "raw_flow", "nominal", "safe",
        "residual", "applied", "correction", "goal_errors", "speeds",
        "epsilon", "first_min_linear_residual", "first_max_speed_excess",
        "second_min_linear_residual", "second_max_speed_excess", "p", "sigma",
        "candidate", "raw_deadlock", "task_success", "wall_collision",
        "agent_collision", "gate", "pre_action_latch", "stuck_timer",
    )
    arrays = {}
    for key in numeric:
        values = [row[key] for row in records]
        arrays[key] = np.asarray(values)
    arrays["first_projection_status"] = np.asarray(
        [row["first_projection_status"] for row in records], dtype="U32")
    arrays["second_projection_status"] = np.asarray(
        [row["second_projection_status"] for row in records], dtype="U32")
    return arrays


def rollout_one(controller, initial, tape, continuation, group, out: Path):
    env = GiveWayEnv(controller.plant)
    env.reset(initial)
    records = []
    terminal_label = None
    failure = None
    started = time.perf_counter()
    while not env.done:
        step = env.step_count
        pre_latch = env.first_deadlock_step is not None
        before = env.positions.copy()
        try:
            if group == "S_safety":
                action = safety_action(controller, env, tape, continuation, step)
            elif group == "A_phi0_diag":
                action = controller.act(env, tape, SPLIT, SCENARIO_ID,
                                        continuation, step)
            else:
                raise ValueError(group)
            _, _, done, info = env.step(np.asarray(action["applied"]).reshape(2, 2))
        except Exception as exc:  # retained as a visible failed assigned rollout
            failure = {"type": type(exc).__name__, "message": str(exc), "step": step}
            break
        correction = np.asarray(action["applied"])-np.asarray(action["safe"])
        second = action["second_projection"]
        records.append(dict(
            positions_before=before, positions_after=np.asarray(info["positions"]),
            raw_flow=np.asarray(action["raw_flow"]), nominal=np.asarray(action["nominal"]),
            safe=np.asarray(action["safe"]), residual=np.asarray(action["residual"]),
            applied=np.asarray(action["applied"]), correction=correction,
            goal_errors=np.asarray(info["goal_errors"]), speeds=np.asarray(info["speeds"]),
            epsilon=np.asarray(action["epsilon"]), p=float(action["p"]),
            sigma=float(action["sigma"]), gate=bool(action["gate"]),
            pre_action_latch=pre_latch, candidate=bool(info["candidate_deadlock"]),
            raw_deadlock=bool(info["deadlock"]), task_success=bool(info["task_success"]),
            wall_collision=bool(info["wall_collision"]),
            agent_collision=bool(info["agent_collision"]),
            stuck_timer=float(info["stuck_timer"]),
            first_projection_status=action["first_projection"].status,
            first_min_linear_residual=action["first_projection"].min_linear_residual,
            first_max_speed_excess=action["first_projection"].max_speed_excess,
            second_projection_status="not_run" if second is None else second.status,
            second_min_linear_residual=np.nan if second is None else second.min_linear_residual,
            second_max_speed_excess=np.nan if second is None else second.max_speed_excess,
        ))
        if done:
            terminal_label = _terminal_label(info, controller.plant)

    arrays = trace_arrays(records) if records else {}
    arrays["initial_positions"] = np.asarray(initial, dtype=np.float64)
    trace_path = out/group/f"rollout_{continuation:04d}.npz"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trace_path, **arrays)
    elapsed = time.perf_counter()-started
    historical = env.first_deadlock_step is not None
    eligible_stalled = False
    stalled_details = {"eligible": False}
    if failure is None:
        speeds = np.max(np.asarray([row["speeds"] for row in records]), axis=1)
        errors = np.asarray([row["goal_errors"] for row in records])
        reclassified, stalled_details = classify_timeout_trace(
            {"max_speed": speeds, "goal_errors": errors}, terminal_label,
            controller.plant.dt)
        eligible_stalled = reclassified == "stalled_deadlock"
    summary = env.summary()
    residual = np.asarray([row["residual"] for row in records]) if records else np.zeros((0, 4))
    correction = np.asarray([row["correction"] for row in records]) if records else np.zeros((0, 4))
    gates = np.asarray([row["gate"] for row in records], dtype=bool)
    residual_norm = np.linalg.norm(residual, axis=1)
    correction_norm = np.linalg.norm(correction, axis=1)
    collision = bool(summary["wall_collision"] or summary["agent_collision"])
    result = dict(
        pair_id=continuation, group=group, assigned=True,
        completed=failure is None, failure=failure,
        D_H=bool(historical or eligible_stalled), historical_strict=historical,
        eligible_stalled=eligible_stalled, stalled_details=stalled_details,
        success=bool(summary["success"]), wall_collision=bool(summary["wall_collision"]),
        agent_collision=bool(summary["agent_collision"]), collision=collision,
        timeout=terminal_label == "other_timeout",
        ordinary_timeout=terminal_label == "other_timeout" and not eligible_stalled,
        terminal_label=terminal_label, first_deadlock_step=summary["first_deadlock_step"],
        first_success_step=summary["first_success_step"], episode_steps=len(records),
        gate_activations=int(gates.sum()), gate_activation_fraction=float(gates.mean()) if len(gates) else None,
        residual_norm_mean=float(residual_norm.mean()) if len(residual_norm) else None,
        residual_norm_rms=float(np.sqrt(np.mean(residual_norm**2))) if len(residual_norm) else None,
        residual_norm_active_mean=float(residual_norm[gates].mean()) if np.any(gates) else None,
        residual_norm_max=float(residual_norm.max()) if len(residual_norm) else None,
        executed_correction_norm_mean=float(correction_norm.mean()) if len(correction_norm) else None,
        executed_correction_norm_rms=float(np.sqrt(np.mean(correction_norm**2))) if len(correction_norm) else None,
        executed_correction_norm_max=float(correction_norm.max()) if len(correction_norm) else None,
        executed_correction_energy=float(controller.plant.dt*np.sum(correction_norm**2)),
        active_projection_alias_count=int(np.sum(gates & (correction_norm <= 1e-10))),
        active_projection_alias_fraction=(float(np.mean(correction_norm[gates] <= 1e-10))
                                          if np.any(gates) else None),
        first_projection_status_counts=dict(Counter(row["first_projection_status"] for row in records)),
        second_projection_status_counts=dict(Counter(row["second_projection_status"] for row in records)),
        min_projection_linear_residual=(float(min(
            min(row["first_min_linear_residual"], row["second_min_linear_residual"])
            if np.isfinite(row["second_min_linear_residual"])
            else row["first_min_linear_residual"] for row in records)) if records else None),
        max_projection_speed_excess=(float(max(
            max(row["first_max_speed_excess"], row["second_max_speed_excess"])
            if np.isfinite(row["second_max_speed_excess"])
            else row["first_max_speed_excess"] for row in records)) if records else None),
        wall_seconds=elapsed, trace=str(trace_path), trace_sha256=sha256(trace_path),
    )
    return result


def clopper_pearson(count, total, alpha=.05):
    if total <= 0:
        return None
    lower = 0.0 if count == 0 else float(beta_distribution.ppf(alpha/2, count, total-count+1))
    upper = 1.0 if count == total else float(beta_distribution.ppf(1-alpha/2, count+1, total-count))
    return [lower, upper]


def group_summary(rows):
    complete = [row for row in rows if row["completed"]]
    result = {"assigned": len(rows), "completed": len(complete),
              "numerical_failures": len(rows)-len(complete),
              "simulator_steps": sum(row["episode_steps"] for row in rows),
              "equivalent_850_step_rollouts": sum(row["episode_steps"] for row in rows)/850.0,
              "wall_seconds": sum(row["wall_seconds"] for row in rows)}
    for field in ("D_H", "historical_strict", "eligible_stalled", "success",
                  "wall_collision", "agent_collision", "collision", "timeout",
                  "ordinary_timeout"):
        count = sum(bool(row[field]) for row in complete)
        result[field] = {"count": count, "denominator": len(complete),
                         "rate": count/len(complete) if complete else None,
                         "clopper_pearson_95": clopper_pearson(count, len(complete))}
    for field in ("gate_activation_fraction", "residual_norm_mean", "residual_norm_rms",
                  "executed_correction_norm_mean", "executed_correction_norm_rms",
                  "executed_correction_energy", "active_projection_alias_fraction"):
        values = [row[field] for row in complete if row[field] is not None]
        result[field] = {"mean": float(np.mean(values)) if values else None,
                         "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                         "min": float(np.min(values)) if values else None,
                         "max": float(np.max(values)) if values else None}
    result["terminal_label_counts"] = dict(Counter(row["terminal_label"] for row in complete))
    return result


def paired_summary(safety, experiment):
    by_s = {row["pair_id"]: row for row in safety}
    by_a = {row["pair_id"]: row for row in experiment}
    valid = [(by_s[index], by_a[index]) for index in range(N_PAIRS)
             if by_s[index]["completed"] and by_a[index]["completed"]]
    counts = Counter()
    details = []
    for old, new in valid:
        if old["D_H"] and not new["D_H"]:
            if new["success"] and not new["collision"]:
                category = "baseline_D1_to_A_D0_success_collision_free"
            else:
                category = "baseline_D1_to_A_D0_without_safe_completion"
        elif not old["D_H"] and new["D_H"]:
            category = "baseline_D0_to_A_D1"
        else:
            category = "same_deadlock_label"
        counts[category] += 1
        details.append({"pair_id": old["pair_id"], "category": category,
                        "S_D_H": old["D_H"], "A_D_H": new["D_H"],
                        "S_success": old["success"], "A_success": new["success"],
                        "A_collision": new["collision"],
                        "A_timeout": new["timeout"]})
    names = (
        "baseline_D1_to_A_D0_success_collision_free",
        "baseline_D1_to_A_D0_without_safe_completion",
        "baseline_D0_to_A_D1", "same_deadlock_label",
    )
    return {"valid_pairs": len(valid), "invalid_pairs": N_PAIRS-len(valid),
            "counts": {name: counts[name] for name in names}, "rows": details}


def verify_manifest(manifest):
    for name, expected in manifest["source_sha256"].items():
        path = ROOT/name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen source changed after manifest: {name}")
    schedule = Path(manifest["randomness"]["seed_schedule"])
    if sha256(schedule) != manifest["randomness"]["seed_schedule_sha256"]:
        raise RuntimeError("seed schedule changed after manifest")
    if sha256(CHECKPOINT) != manifest["asset_selection"]["checkpoint_sha256"]:
        raise RuntimeError("checkpoint changed after manifest")


def run(out: Path, manifest):
    record_path = out/"records.jsonl"
    report_path = out/"report.json"
    if record_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite an existing pilot")
    verify_manifest(manifest)
    _, scene_cfg, _, initial = archived_assets()
    plant = Config(**scene_cfg["environment"])
    cbf = CBFConfig(**scene_cfg["cbf"])
    baseline, provenance = load_policy(CHECKPOINT)
    if provenance["evaluation_environment"] != scene_cfg["environment"]:
        raise RuntimeError("checkpoint and selected original Safety environment disagree")
    distribution = DirectionADistribution(.05*plant.max_speed, .50*plant.max_speed)
    model = ConstantDiagnosticPolicy()
    controller = FrozenDirectionAController(
        baseline, model, {}, distribution, plant, cbf=cbf)
    # Diagnostic-only compilation wrapper; it evaluates the unchanged frozen
    # Flow function and does not change its inputs, outputs, or random draws.
    compiled_flow = jax.jit(controller._frozen_flow.baseline_sample)
    controller._flow_sample = compiled_flow
    # Compile once before timing assigned rollouts without consuming a pilot key.
    dummy_env = GiveWayEnv(plant); dummy_env.reset(initial)
    dummy_obs = dummy_env.observation()[None]
    _ = np.asarray(compiled_flow(dummy_obs, jnp.zeros((1, 4), dtype=jnp.float32)))
    tape = IndexedRandomTape(ROOT_SEED)
    rows = []
    started = time.time()
    for continuation in range(N_PAIRS):
        for group in ("S_safety", "A_phi0_diag"):
            row = rollout_one(controller, initial, tape, continuation, group, out)
            rows.append(row)
            with record_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({"completed_pair": continuation,
                          "S_D_H": rows[-2]["D_H"], "A_D_H": rows[-1]["D_H"],
                          "S_label": rows[-2]["terminal_label"],
                          "A_label": rows[-1]["terminal_label"]}), flush=True)
    safety = [row for row in rows if row["group"] == "S_safety"]
    experiment = [row for row in rows if row["group"] == "A_phi0_diag"]
    report = {
        "schema": SCHEMA,
        "status": "COMPLETE" if all(row["completed"] for row in rows) else "COMPLETE_WITH_FAILURES",
        "interpretation": {
            "formal_method_verdict": "NOT APPLICABLE",
            "gradient_or_learning_evidence": False,
            "success_definition": "monitor task_success latch, never motion alone",
            "scope": "conditional on one archived t=0 GiveWay initial state and the fixed phi0_diag",
        },
        "manifest": str(out/"manifest.json"),
        "manifest_sha256": sha256(out/"manifest.json"),
        "runtime": {
            "started_unix": started, "finished_unix": time.time(),
            "wall_seconds": time.time()-started,
            "python": platform.python_version(), "jax": jax.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "jax_x64": bool(jax.config.x64_enabled),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "groups": {"S_safety": group_summary(safety),
                   "A_phi0_diag": group_summary(experiment)},
        "paired": paired_summary(safety, experiment),
        "total": {
            "assigned_rollouts": len(rows),
            "completed_rollouts": sum(row["completed"] for row in rows),
            "simulator_steps": sum(row["episode_steps"] for row in rows),
            "equivalent_850_step_rollouts": sum(row["episode_steps"] for row in rows)/850.0,
            "trace_bytes": sum(Path(row["trace"]).stat().st_size for row in rows),
        },
        "record_file": str(record_path),
        "record_file_sha256": sha256(record_path),
        "source_sha256_at_completion": {
            name: sha256(ROOT/name) for name in manifest["source_sha256"]
        },
    }
    json_dump(report_path, report)
    print(json.dumps({"report": str(report_path), "status": report["status"],
                      "groups": report["groups"],
                      "paired_counts": report["paired"]["counts"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT/"results/giveway_residual_effect_pilot_v1")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    manifest = make_manifest(out)
    print(json.dumps({"manifest": str(out/"manifest.json"),
                      "manifest_sha256": sha256(out/"manifest.json"),
                      "prepared": True}, indent=2))
    if not args.prepare_only:
        run(out, manifest)


if __name__ == "__main__":
    main()
