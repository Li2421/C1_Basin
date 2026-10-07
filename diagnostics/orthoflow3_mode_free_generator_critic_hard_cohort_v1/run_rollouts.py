#!/usr/bin/env python3
"""Matched Q16 for frozen mode-free proposals on original difficult cohort."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(SYSROOT), str(ROOT)]

from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.historical_ingest import TOY_FLOW_SHA
from shared_rollout_db.src.rollout_db import eta_identity, canonical
from diagnostics.orthoflow3_mode_free_generator_critic_hard_cohort_v1.prepare_cache_plan import RNG

EXPERIMENT_UID = "exp_mode_free_generator_critic_hard_cohort_v1"


def tasks():
    obj = json.loads((OUT / "frozen_proposals.json").read_text())
    out = []
    for state in obj["states"]:
        cands = {"safety": [0.0, 0.0, 0.0], **state["eta"]}
        by_eta = defaultdict(list)
        for kind, eta in cands.items(): by_eta[eta_identity(eta)[0]].append((kind, eta))
        unique = [(names[0][0], names[0][1], [x[0] for x in names]) for names in by_eta.values()]
        unique.append(("mac_only", [0.0, 0.0, 0.0], ["mac_only"]))
        for kind, eta, aliases in unique:
            for future_index in range(16):
                out.append({"state": state, "kind": kind, "aliases": aliases, "eta": eta,
                            "future_index": future_index})
    return out


def label(env, error):
    if error: return "numerical_failure", str(error)
    s = env.summary()
    if s["wall_collision"] or s["agent_collision"]: return "collision", "collision"
    if s["success"]: return "success", "success"
    if s["deadlock"]: return "deadlock", "strict_deadlock"
    return "timeout", "timeout"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--max-tasks", type=int)
    ap.add_argument("--cache-preflight", type=Path)
    ap.add_argument("--feature-diagnostic-episode", type=int)
    args = ap.parse_args()
    import jax
    import jax.numpy as jnp
    from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
    from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
    from shared_control.basis_families import get_basis_family
    from single_integrator.cbf import CBFConfig, barrier_constraints
    from single_integrator.environment import Config, GiveWayEnv, bounded_nominal
    from single_integrator.evaluate import load_policy

    jax.config.update("jax_enable_x64", True)
    wide = json.loads((ROOT / "diagnostics/gphi_wide_ic_cadence_v1/frozen_benchmark_manifest.json").read_text())
    config = Config(**wide["environment"])
    cbf = CBFConfig(**wide["cbf"])
    policy, provenance = load_policy(SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl")
    if provenance["evaluation_environment"] != wide["environment"]: raise RuntimeError("Flow environment changed")
    sample = jax.jit(lambda obs, key: policy.sample_actions(obs[None], seed=key)[0])
    basis = get_basis_family("orthoflow3")
    expected_h = np.load(OUT / "cohort_features.npz")["h_raw"]
    all_tasks = tasks()
    if args.feature_diagnostic_episode is not None:
        ep = args.feature_diagnostic_episode
        state = next(t["state"] for t in all_tasks if int(t["state"]["episode_index"]) == ep)
        env = GiveWayEnv(config)
        env.reset(np.asarray(state["initial_positions"], np.float64))
        initial_key = jax.random.fold_in(jax.random.PRNGKey(42), int(state["rollout_id"]))
        flow_key = jax.random.fold_in(initial_key, 0)
        obs = np.asarray(env.observation(), np.float32)
        flow = bounded_nominal(np.asarray(sample(jnp.asarray(obs), flow_key), np.float64), config.max_speed)
        A, lower, _ = barrier_constraints(env.snapshot(), cbf)
        safe, _, _, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
        feature, _ = StartupAwareFeatureBuilder().build(env, {"u_flow": flow, "u_safe": safe}, config, cbf)
        got = np.asarray(feature, np.float32)
        delta = np.abs(got - expected_h[ep])
        print(json.dumps({"episode_index": ep, "max_abs": float(delta.max()),
                          "mean_abs": float(delta.mean()), "n_changed": int(np.count_nonzero(delta)),
                          "n_over_1e-5": int(np.sum(delta > 1e-5)),
                          "flow": flow.tolist(), "safe": safe.tolist(),
                          "platform": str(jax.default_backend())}))
        return
    assigned = [t for i, t in enumerate(all_tasks) if i % args.shards == args.shard]
    missing = None
    if args.cache_preflight:
        audit = json.loads(args.cache_preflight.read_text())
        missing = {(d["state_uid"], d["eta_uid"], d["controller_uid"], seed)
                   for d in audit["details"] for seed in d["missing_seeds"]}
        with (OUT / "candidate_cache_keys.csv").open(newline="") as f:
            lookup = {(int(r["episode_index"]), r["kind"]):
                      (r["state_uid"], r["eta_uid"], r["controller_uid"])
                      for r in csv.DictReader(f)}
    rawdir = OUT / "raw"
    rawdir.mkdir(exist_ok=True)
    path = rawdir / f"shard{args.shard}.jsonl"
    existing = [] if not path.exists() else [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    seen = {(r["episode_index"], r["kind"], r["future_index"]) for r in existing}
    pending = []
    n = 0
    steps = 0
    began = time.monotonic()
    with path.open("a") as sink:
        for t in assigned:
            state, kind, eta, index = t["state"], t["kind"], np.asarray(t["eta"], np.float64), t["future_index"]
            ep = int(state["episode_index"])
            if missing is not None:
                k = (*lookup[(ep, kind)], canonical({"future_index": index}))
                if k not in missing: continue
            if (ep, kind, index) in seen: continue
            env = GiveWayEnv(config)
            env.reset(np.asarray(state["initial_positions"], np.float64))
            jdef = 0.0
            error = None
            initial_key = jax.random.fold_in(jax.random.PRNGKey(42), int(state["rollout_id"]))
            future_key = jax.random.fold_in(jax.random.PRNGKey(42 + index), int(state["rollout_id"]))
            for step in range(config.max_steps):
                try:
                    obs = np.asarray(env.observation(), np.float32)
                    flow_key = jax.random.fold_in(initial_key if step == 0 else future_key, step)
                    flow = bounded_nominal(np.asarray(sample(jnp.asarray(obs), flow_key), np.float64), config.max_speed)
                    if kind == "mac_only":
                        executed = flow
                    else:
                        A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                        safe, _, _, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                        if step == 0:
                            feature, _ = StartupAwareFeatureBuilder().build(env, {"u_flow": flow, "u_safe": safe}, config, cbf)
                            got = np.asarray(feature, np.float32)
                            if not np.allclose(got, expected_h[ep], rtol=1e-6, atol=1e-7):
                                raise RuntimeError(f"Frozen true-t0 feature mismatch: max_abs={np.max(np.abs(got-expected_h[ep])):.9g}")
                        correction = np.zeros_like(safe) if kind == "safety" else basis.compute(env.positions, env.goals, safe, config.max_speed).correction(eta)
                        executed, _, _, _ = project_velocity_with_retry(safe + correction, A, lower, config.max_speed, cbf)
                        if np.min(A @ executed.reshape(4) - lower) < -cbf.feasibility_tol:
                            raise RuntimeError("Hard-safety projection violated")
                        jdef += config.dt * float(np.sum((executed - safe) ** 2))
                    env.step(executed)
                    steps += 1
                    if env.done: break
                except Exception as exc:
                    error = {"type": type(exc).__name__, "message": str(exc), "step": step}
                    break
            outcome, failure = label(env, error)
            rec = {"scenario": "Toy", "representation": "MAC_ONLY" if kind == "mac_only" else "P1-OrthoFlow3",
                   "controller_kind": "MAC_ONLY" if kind == "mac_only" else "ORTHOFLOW3",
                   "controller": kind, "kind": kind, "candidate_aliases": t["aliases"],
                   "state_id": state.get("source_group", f"wide_ic_frozen_{ep:04d}"), "episode_index": ep,
                   "source_group": state["source_group"], "initial_positions": state["initial_positions"],
                   "h_sha256": state["h_sha256"], "eta": eta.tolist(), "future_index": index,
                   "rollout_id": state["rollout_id"], "rng_semantics_version": RNG,
                   "flow_checkpoint_sha256": TOY_FLOW_SHA, "horizon": config.max_steps,
                   "dt": config.dt, "success_semantics_version": "window_progress_v2",
                   "conditioning_version": "flow_only_stepwise_v1" if kind == "mac_only" else "true_t0_latched_eta_v1",
                   "outcome": outcome, "failure_type": failure,
                   "success": outcome == "success", "deadlock": outcome == "deadlock",
                   "timeout": outcome == "timeout", "collision": outcome == "collision",
                   "numerical_failure": outcome == "numerical_failure", "scientific_outcome_valid": error is None,
                   "episode_steps": env.step_count, "J_def": None if kind == "mac_only" else jdef,
                   "execution_error": error}
            sink.write(json.dumps(rec, sort_keys=True) + "\n")
            sink.flush()
            pending.append(rec)
            n += 1
            if len(pending) >= 250:
                append_journal(pending, EXPERIMENT_UID, f"shard{args.shard}")
                pending.clear()
            if args.max_tasks is not None and n >= args.max_tasks: break
    if pending: append_journal(pending, EXPERIMENT_UID, f"shard{args.shard}")
    (rawdir / f"shard{args.shard}_runtime.json").write_text(json.dumps({"completed_new": n, "physical_steps": steps,
       "seconds": time.monotonic() - began, "assigned": len(assigned), "shard": args.shard}, indent=2) + "\n")
    print(json.dumps({"shard": args.shard, "new": n, "steps": steps}))


if __name__ == "__main__": main()
