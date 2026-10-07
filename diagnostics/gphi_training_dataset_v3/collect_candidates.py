"""Collect independent recovery-like states before looking at eta=0 labels.

Each candidate comes from a distinct successful corrected continuation of one
of twelve V2 pre-deadlock anchors.  Selection times are fixed fractions of the
successful source trajectory and therefore do not use oracle outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V2 = ROOT / "diagnostics/gphi_training_dataset_v2"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
sys.path.insert(0, str(SYSROOT))

from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi
from diagnostics.gphi_training_dataset_v2.build_states import clone_env, restore_full, save_full
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
from single_integrator.cbf import CBFConfig, barrier_constraints
from single_integrator.environment import Config, bounded_nominal
from single_integrator.evaluate import load_policy


ANCHORS = (
    "P_r052_m080", "P_r096_m080", "P_r106_m080", "P_r175_m080",
    "P_r182_m080", "P_r191_m080", "P_r198_m080", "Q_pair225_m080",
    "Q_pair226_m080", "Q_pair227_m040", "Q_pair228_m080", "D1_pair231",
)
FRACTIONS = (0.60, 0.68, 0.74, 0.80, 0.84, 0.88, 0.92, 0.95)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def event_name(summary: dict, done: bool) -> str:
    if not done:
        return "running"
    if summary["wall_collision"] or summary["agent_collision"]:
        return "collision"
    if summary["success"]:
        return "success"
    if summary["deadlock"]:
        return "deadlock"
    return "timeout"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--per-group", type=int, default=8)
    args = parser.parse_args()
    if args.per_group > len(FRACTIONS):
        raise ValueError("per-group exceeds frozen phase-fraction plan")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)

    for directory in (HERE / "candidate_states", HERE / "candidate_sources", HERE / "audit_refs", HERE / "raw"):
        directory.mkdir(parents=True, exist_ok=True)
    if any((HERE / "candidate_states").iterdir()) or any((HERE / "candidate_sources").iterdir()):
        raise RuntimeError("candidate output is nonempty; refusing to overwrite")

    v2_protocol = json.loads((V2 / "protocol.json").read_text())
    config = Config(**v2_protocol["environment"])
    cbf = CBFConfig()
    policy, _ = load_policy(Path(v2_protocol["checkpoint"]))
    v2_states = {row["state_id"]: row for row in read_jsonl(V2 / "state_manifest.jsonl")}
    v2_oracles = {row["state_id"]: row for row in read_jsonl(V2 / "oracle_search_results.jsonl")}
    missing = sorted(set(ANCHORS) - set(v2_states))
    if missing:
        raise RuntimeError(("missing anchors", missing))

    specs = []
    for group_index, state_id in enumerate(ANCHORS):
        anchor = v2_states[state_id]
        eta = v2_oracles[state_id]["eta_best"]
        if eta is None or eta == [0.0, 0.0, 0.0]:
            raise AssertionError((state_id, eta))
        for trajectory_index, fraction in enumerate(FRACTIONS[: args.per_group]):
            specs.append({
                "anchor_state": state_id,
                "anchor": anchor,
                "eta": eta,
                "source_seed": 95400001 + 100 * group_index + trajectory_index,
                "trajectory_index": trajectory_index,
                "phase_fraction": fraction,
            })

    envs = [restore_full(V2 / spec["anchor"]["state_file"], config) for spec in specs]
    correctors = [DiagnosticCorrector(DiagnosticPhi(*spec["eta"])) for spec in specs]
    histories = [{key: [] for key in (
        "positions_before", "positions_after", "u_flow", "u_safe", "g_raw", "u_exec",
        "event", "stuck_timer_after", "candidate_since_after", "max_stuck_timer_after",
    )} for _ in specs]
    errors = [None] * len(specs)
    key0 = np.asarray([
        np.asarray(jax.random.fold_in(jax.random.PRNGKey(spec["source_seed"]), int(spec["anchor"]["rng_namespace"])))
        for spec in specs
    ])
    single = lambda obs, key: policy.sample_actions(obs[None], seed=key)[0]
    sample = jax.jit(jax.vmap(single))
    fold = jax.jit(jax.vmap(jax.random.fold_in))
    started = time.monotonic()
    rounds = 0

    while any(not env.done and errors[i] is None for i, env in enumerate(envs)):
        observations = np.stack([env.observation() for env in envs])
        steps = np.asarray([env.step_count for env in envs], dtype=np.uint32)
        actions = np.asarray(sample(jnp.asarray(observations), fold(jnp.asarray(key0), jnp.asarray(steps))))
        for i, (spec, env) in enumerate(zip(specs, envs)):
            if env.done or errors[i] is not None:
                continue
            history = histories[i]
            before = env.positions.copy()
            flow = bounded_nominal(actions[i], config.max_speed)
            A, lower, _ = barrier_constraints(env.snapshot(), cbf)
            try:
                safe, _, _, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
                raw = correctors[i](observations[i], safe, config.max_speed)
                executed, _, _, _ = project_velocity_with_retry(safe + raw, A, lower, config.max_speed, cbf)
                _, _, done, _ = env.step(executed)
            except Exception as exc:
                errors[i] = {"type": type(exc).__name__, "message": str(exc), "step": env.step_count}
                continue
            summary = env.summary()
            history["positions_before"].append(before)
            history["positions_after"].append(env.positions.copy())
            history["u_flow"].append(flow)
            history["u_safe"].append(safe)
            history["g_raw"].append(raw)
            history["u_exec"].append(executed)
            history["event"].append(event_name(summary, done))
            history["stuck_timer_after"].append(env.stuck_timer)
            history["candidate_since_after"].append(-1 if env.candidate_since is None else env.candidate_since)
            history["max_stuck_timer_after"].append(env.max_stuck_timer)
        rounds += 1
        if rounds % 100 == 0:
            print(json.dumps({"source_round": rounds, "active": sum(not e.done for e in envs), "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)

    candidates = []
    source_outcomes = Counter()
    for index, (spec, env, history, error) in enumerate(zip(specs, envs, histories, errors)):
        summary = env.summary()
        outcome = "execution_error" if error else event_name(summary, True)
        source_outcomes[outcome] += 1
        arrays = {key: np.asarray(value) for key, value in history.items()}
        trace_id = f"v3_{spec['anchor_state']}_seed{spec['source_seed']}"
        trace_path = HERE / "candidate_sources" / f"{trace_id}.npz"
        np.savez_compressed(
            trace_path, **arrays, anchor_state=np.asarray(spec["anchor_state"]),
            eta=np.asarray(spec["eta"], dtype=np.float64), source_seed=np.asarray(spec["source_seed"]),
            anchor_rng_namespace=np.asarray(spec["anchor"]["rng_namespace"]), outcome=np.asarray(outcome),
        )
        if outcome != "success":
            continue
        length = len(history["u_exec"])
        local = max(5, min(length - 2, int(round(spec["phase_fraction"] * length))))
        replay = restore_full(V2 / spec["anchor"]["state_file"], config)
        for action in arrays["u_exec"][:local]:
            _, _, done, _ = replay.step(action)
            if done:
                raise AssertionError((trace_id, local, replay.step_count, "premature replay termination"))
        if not np.allclose(replay.positions, arrays["positions_before"][local], atol=1e-12, rtol=0):
            raise AssertionError((trace_id, local, "source replay mismatch"))
        state_id = f"ZR_{spec['anchor_state']}_s{spec['source_seed']}_p{local:03d}"
        state_path = HERE / "candidate_states" / f"{state_id}.npz"
        save_full(state_path, replay)
        ref_path = HERE / "audit_refs" / f"{state_id}.npz"
        np.savez_compressed(
            ref_path,
            u_flow=arrays["u_flow"][local], u_safe=arrays["u_safe"][local],
            g_raw=arrays["g_raw"][local], u_exec=arrays["u_exec"][local],
            positions_before=arrays["positions_before"][local], positions_after=arrays["positions_after"][local],
            event=arrays["event"][local], stuck_timer_after=arrays["stuck_timer_after"][local],
            candidate_since_after=arrays["candidate_since_after"][local],
            max_stuck_timer_after=arrays["max_stuck_timer_after"][local],
        )
        phase = "transition" if spec["phase_fraction"] < 0.70 else "late_recovery" if spec["phase_fraction"] < 0.88 else "post_clearance"
        candidates.append({
            "state_id": state_id, "category": "RECOVERY", "recovery_phase": phase,
            "candidate_rank_within_group": spec["trajectory_index"],
            "planned_phase_fraction": spec["phase_fraction"],
            "source_type": "v3_corrected_recovery", "source_path": str(trace_path.relative_to(HERE)),
            "source_sha256": sha(trace_path), "source_trajectory": trace_id,
            "source_group": spec["anchor"]["leakage_group"], "leakage_group": spec["anchor"]["leakage_group"],
            "source_seed": spec["source_seed"], "source_rng_id": spec["anchor"]["source_rng_id"],
            "source_local_step": local, "source_length": length,
            "absolute_step": replay.step_count, "anchor_state": spec["anchor_state"],
            "anchor_step": spec["anchor"]["step"], "source_eta": spec["eta"],
            "anchor_rng_namespace": spec["anchor"]["rng_namespace"],
            "rng_namespace": 300000 + index,
            "state_file": str(state_path.relative_to(HERE)), "state_sha256": sha(state_path),
            "audit_reference_file": str(ref_path.relative_to(HERE)), "audit_reference_sha256": sha(ref_path),
            "step": replay.step_count,
            "candidate_since": -1 if replay.candidate_since is None else replay.candidate_since,
            "stuck_timer": replay.stuck_timer, "max_stuck_timer": replay.max_stuck_timer,
            "ever_candidate_deadlock": replay.ever_candidate_deadlock,
            "retained_from_v2": False,
            "provisional_split": spec["anchor"]["split"],
        })

    if len(candidates) < 6:
        raise RuntimeError(("too few successful corrected sources", len(candidates), dict(source_outcomes)))
    write_jsonl(HERE / "candidate_state_manifest.jsonl", candidates)
    protocol = {
        **v2_protocol,
        "study": "gphi_training_dataset_v3_recovery_zero_coverage",
        "training_performed": False,
        "state_selection_frozen_before_oracle": True,
        "candidate_state_count": len(candidates),
        "candidate_source_group_count": len({row["source_group"] for row in candidates}),
        "candidate_phase_fractions": list(FRACTIONS[: args.per_group]),
        "oracle_seed_default": list(range(95710001, 95710065)),
        "oracle_rule": "eta=(0,0,0) success_count >= 63/64; stop after second physical failure",
        "resource_plan": {"gpu_shards": 1, "cpu_workers": 4, "reason": "another laboratory user's scheduler job was active"},
    }
    (HERE / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({
        "candidates": len(candidates), "source_groups": len({row["source_group"] for row in candidates}),
        "successful_source_trajectories": source_outcomes["success"],
        "source_outcomes": dict(source_outcomes), "device": [str(device) for device in jax.devices()],
        "elapsed_s": time.monotonic() - started,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
