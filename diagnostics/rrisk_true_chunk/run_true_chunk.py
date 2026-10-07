"""Run predeclared persistent-correction true-D_H continuations.

This diagnostic inserts a constant joint correction before the frozen second
hard projection.  It never trains or differentiates through the environment,
projection, monitor, latch, or event.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.differentiable_rollout import ResidualFlowField
from single_integrator.c1.models import ResidualCorrection
from single_integrator.c1.rollout_vi_r_cert import rollout
from single_integrator.c1.train import approved_checkpoint
from single_integrator.cbf import CBFConfig
from single_integrator.environment import Config
from single_integrator.evaluate import load_policy


HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def noise_steps(seed: int, rid: int, steps: int = 850):
    key = jax.random.fold_in(jax.random.PRNGKey(seed), rid)
    return jax.vmap(
        lambda t: jax.random.normal(
            jax.random.fold_in(key, t), (4,), dtype=jnp.float32
        )
    )(jnp.arange(steps, dtype=jnp.uint32))


def chunk_conditioned_noise(prefix_seed: int, reference_seed: int,
                            continuation_seed: int, rid: int,
                            action_step: int, old_split: int = 101):
    """Freeze the old prefix/reference state and current action latent.

    At action 100 this is exactly the old TEST-Q convention: prefix latent at
    action 100, independent continuation randomness from action 101 onward.
    At later states, the old risk trace is replayed through the intervened
    action, and independent continuation randomness begins at t+1.
    """
    if action_step < old_split - 1:
        raise ValueError("selected chunk state precedes the frozen old-Q state")
    prefix = noise_steps(prefix_seed, rid)
    reference = noise_steps(reference_seed, rid)
    continuation = noise_steps(continuation_seed, rid)
    return jnp.concatenate((
        prefix[:old_split],
        reference[old_split:action_step + 1],
        continuation[action_step + 1:],
    ), axis=0)


def setup_frozen(protocol):
    jax.config.update("jax_enable_x64", True)
    checkpoint = Path(protocol["assets"]["checkpoint"])
    expected = protocol["assets"]["checkpoint_sha256"]
    if digest(checkpoint) != expected or approved_checkpoint(checkpoint) != expected:
        raise RuntimeError("approved checkpoint hash mismatch")
    baseline, provenance = load_policy(checkpoint)
    plant = Config(**provenance["evaluation_environment"])
    if not all((plant.terminate_on_collision, plant.terminate_on_success,
                plant.terminate_on_deadlock)):
        raise RuntimeError("frozen first-event termination flags changed")
    if (plant.dt != protocol["frozen_system"]["dt_seconds"] or
            plant.max_steps != protocol["frozen_system"]["max_steps"] or
            plant.max_speed != protocol["frozen_system"]["max_speed"]):
        raise RuntimeError("frozen plant constants changed")
    model = ResidualCorrection(
        hidden_dims=tuple(baseline.config["actor_hidden_dims"]),
        layer_norm=baseline.config["actor_layer_norm"],
    )
    params = model.init(
        jax.random.PRNGKey(0), jnp.zeros((1, 4)), jnp.zeros((1, 1)),
        jnp.zeros((1, 20)),
    )
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), params)
    field = ResidualFlowField(baseline, model)
    field.baseline_sample = jax.jit(field.baseline_sample)
    field.correction = jax.jit(field.correction)
    return params, field, plant, CBFConfig()


def arm_specs(horizon: int, protocol):
    """Unique physical arms with all applicable budget-family labels."""
    families = protocol["pilot"]["budget_families"]
    norms = []
    for family in families:
        if family["name"] == "energy_0p2":
            norm = 0.2 / math.sqrt(horizon)
        else:
            norm = float(family["alpha_norm"])
        match = next((entry for entry in norms
                      if math.isclose(entry["norm"], norm,
                                      rel_tol=0.0, abs_tol=1e-12)), None)
        if match is None:
            match = {"norm": norm, "budget_families": []}
            norms.append(match)
        match["budget_families"].append(family["name"])
    arms = [dict(
        arm_id="baseline", alpha=[0.0] * 4, alpha_norm=0.0,
        axis=None, sign=0, budget_families=["baseline"],
        planned_total_energy=0.0,
    )]
    for entry in norms:
        for axis in range(4):
            for sign, word in ((1, "plus"), (-1, "minus")):
                alpha = np.zeros(4, np.float64)
                alpha[axis] = sign * entry["norm"]
                token = format(entry["norm"], ".12g").replace(".", "p")
                arms.append(dict(
                    arm_id=f"norm{token}_e{axis}_{word}",
                    alpha=alpha.tolist(), alpha_norm=float(entry["norm"]),
                    axis=axis, sign=sign,
                    budget_families=entry["budget_families"],
                    planned_total_energy=float(horizon * entry["norm"] ** 2),
                ))
    return arms


def scalar(value):
    array = np.asarray(value)
    if array.shape:
        return array.tolist()
    item = array.item()
    return bool(item) if isinstance(item, (bool, np.bool_)) else item


def outcome_summary(terms):
    keys = (
        "primary_deadlock", "historical_strict_deadlock", "stalled_deadlock",
        "terminal_safe_deadlock", "success", "wall_collision",
        "agent_collision", "original_other_timeout", "terminal_code",
        "action_count", "first_deadlock_step",
    )
    result = {key: scalar(terms[key]) for key in keys}
    result["D_H"] = bool(result["primary_deadlock"])
    result["collision"] = bool(
        result["wall_collision"] or result["agent_collision"])
    result["timeout"] = bool(
        result["original_other_timeout"] and not result["stalled_deadlock"])
    return result


def step_or_none(array, index):
    value = np.asarray(array[index])
    if not np.isfinite(value).all():
        return None
    return value.tolist()


def chunk_diagnostics(trace, action_step: int, horizon: int, alpha,
                      active_tolerance: float = 1e-7):
    alpha = np.asarray(alpha, np.float64)
    norm = float(np.linalg.norm(alpha))
    sliced = jax.device_get({
        key: trace[key][action_step:action_step + horizon]
        for key in ("before", "safe", "w", "applied", "A", "b",
                    "alive_pre", "latch_pre", "candidate", "event_code")
    })
    live = np.asarray(sliced["alive_pre"], bool)
    per_step = []
    realized_norms, realized_energy, removal_fractions = [], [], []
    active_sets = []
    max_zero_residual_error = 0.0
    for local in range(horizon):
        alive = bool(live[local])
        row = dict(action_index=action_step + local, alive_pre=alive,
                   latch_pre=bool(sliced["latch_pre"][local]),
                   candidate_after=bool(sliced["candidate"][local]),
                   event_code_after=int(sliced["event_code"][local]),
                   requested_correction=alpha.tolist())
        if not alive:
            row.update(
                position=None, safe=None, pre_pi2=None, executed=None,
                executed_correction=None, removed_by_pi2=None,
                removed_fraction=None, active_safety=[], speed_active=[])
            per_step.append(row)
            continue
        before = np.asarray(sliced["before"][local], np.float64)
        safe = np.asarray(sliced["safe"][local], np.float64)
        pre = np.asarray(sliced["w"][local], np.float64)
        applied = np.asarray(sliced["applied"][local], np.float64)
        matrix = np.asarray(sliced["A"][local], np.float64)
        lower = np.asarray(sliced["b"][local], np.float64)
        realized = applied - safe
        removed = pre - applied
        model_error = pre - safe - alpha
        max_zero_residual_error = max(
            max_zero_residual_error, float(np.linalg.norm(model_error)))
        removal_fraction = (float(np.linalg.norm(removed)) / norm
                            if norm > 0 else 0.0)
        active = np.flatnonzero(matrix @ applied - lower <= active_tolerance)
        speed = np.linalg.norm(applied.reshape(2, 2), axis=1)
        speed_active = np.flatnonzero(0.5 - speed <= active_tolerance)
        realized_norm = float(np.linalg.norm(realized))
        realized_norms.append(realized_norm)
        realized_energy.append(realized_norm ** 2)
        removal_fractions.append(removal_fraction)
        active_sets.append(tuple(int(x) for x in active))
        row.update(
            position=before.tolist(), safe=safe.tolist(), pre_pi2=pre.tolist(),
            executed=applied.tolist(), executed_correction=realized.tolist(),
            executed_correction_norm=realized_norm,
            removed_by_pi2=removed.tolist(),
            removed_fraction=removal_fraction,
            active_safety=[int(x) for x in active],
            speed_active=[int(x) for x in speed_active],
        )
        per_step.append(row)
    switches = sum(a != b for a, b in zip(active_sets, active_sets[1:]))
    return dict(
        planned_steps=horizon, live_chunk_steps=int(live.sum()),
        requested_per_step_norm=norm,
        planned_requested_energy=float(horizon * norm ** 2),
        live_requested_energy=float(live.sum() * norm ** 2),
        cumulative_executed_correction_norm=float(sum(realized_norms)),
        executed_correction_energy=float(sum(realized_energy)),
        mean_executed_correction_norm=(float(np.mean(realized_norms))
                                       if realized_norms else None),
        mean_removed_fraction=(float(np.mean(removal_fractions))
                               if removal_fractions else None),
        median_removed_fraction=(float(np.median(removal_fractions))
                                 if removal_fractions else None),
        max_removed_fraction=(float(np.max(removal_fractions))
                              if removal_fractions else None),
        active_set_switches=int(switches),
        distinct_active_sets=int(len(set(active_sets))),
        max_frozen_G_phi_output_norm=max_zero_residual_error,
        per_step=per_step,
    )


def selected_states(protocol, stage):
    selection = protocol["state_selection"]
    if stage == "pilot":
        return selection["pilot_states"]
    if stage == "reserve":
        return selection["reserve_states"]
    if stage in {"l5", "l40"}:
        return selection["pilot_states"] + selection["reserve_states"]
    raise ValueError(f"unsupported stage: {stage}")


def stage_horizons(protocol, stage):
    if stage in {"pilot", "reserve"}:
        return protocol["pilot"]["horizons_steps"]
    return [5] if stage == "l5" else [40]


def validate_assets(protocol):
    assets = protocol["assets"]
    checks = [
        (assets["checkpoint"], assets["checkpoint_sha256"]),
        (assets["old_manifest"], assets["old_manifest_sha256"]),
        (assets["old_test_q"], assets["old_test_q_sha256"]),
        (assets["start_protocol"], assets["start_protocol_sha256"]),
    ]
    checks.extend((item["path"], item["sha256"])
                  for item in assets["reference_traces"].values())
    for name, expected in checks:
        if digest(Path(name)) != expected:
            raise RuntimeError(f"frozen asset hash mismatch: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path,
                        default=HERE / "predeclared_protocol.json")
    parser.add_argument("--stage", choices=("pilot", "reserve", "l5", "l40"),
                        default="pilot")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "FROZEN_BEFORE_NEW_OUTCOMES":
        raise RuntimeError("refusing an unfrozen protocol")
    validate_assets(protocol)
    states = selected_states(protocol, args.stage)
    horizons = stage_horizons(protocol, args.stage)
    seeds = protocol["pilot"]["estimation_seeds"]
    scenario_count = sum(len(arm_specs(L, protocol)) for L in horizons) * len(states)
    rollout_count = scenario_count * len(seeds)
    if args.dry_run:
        print(json.dumps(dict(
            stage=args.stage, state_ids=[state["state_id"] for state in states],
            horizons=horizons,
            arms_by_horizon={str(L): len(arm_specs(L, protocol)) for L in horizons},
            arm_scenarios=scenario_count, continuations=rollout_count,
        ), indent=2))
        return
    if jax.default_backend() != "gpu":
        raise RuntimeError("true continuation outcomes require the Slurm GPU environment")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    records_path = args.out / "arm_records.json"
    runtime_path = args.out / "runtime.json"
    protocol_sha = digest(args.protocol)
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    completed = {row["scenario_id"] for row in records}
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text())
        if runtime["protocol_sha256"] != protocol_sha:
            raise RuntimeError("resume protocol mismatch")
    else:
        runtime = dict(
            schema="rrisk_true_chunk_runtime_v1", stage=args.stage,
            protocol=str(args.protocol.resolve()), protocol_sha256=protocol_sha,
            source_sha256=digest(Path(__file__)), backend=jax.default_backend(),
            jax_version=jax.__version__, started=time.time(), no_training=True,
            expected_arm_scenarios=scenario_count,
            expected_continuations=rollout_count,
        )
        atomic_json(runtime_path, runtime)

    params, field, plant, cbf = setup_frozen(protocol)
    start_protocol = json.loads(Path(protocol["assets"]["start_protocol"]).read_text())
    starts = np.asarray(start_protocol["starts"], np.float64)
    full_fn = jax.jit(lambda initial, draws, offsets: rollout(
        params, field, initial, draws, plant, cbf, offsets))
    frozen = protocol["frozen_system"]
    prefix_seed = int(frozen["prefix_seed"])
    reference_seed = int(frozen["reference_suffix_seed"])
    old_split = int(frozen["old_q_split_action"])
    reference_cache = {}
    for rid_text, item in protocol["assets"]["reference_traces"].items():
        with np.load(item["path"], allow_pickle=False) as trace:
            reference_cache[int(rid_text)] = {
                "before": np.asarray(trace["before"]),
                "alive_pre": np.asarray(trace["alive_pre"]),
                "latch_pre": np.asarray(trace["latch_pre"]),
            }

    for state in states:
        rid = int(state["rid"])
        action_step = int(state["action_step"])
        reference = reference_cache[rid]
        if (not bool(reference["alive_pre"][action_step]) or
                bool(reference["latch_pre"][action_step])):
            raise RuntimeError(f"invalid selected pre-event state {state['state_id']}")
        draws_by_seed = {
            seed: chunk_conditioned_noise(
                prefix_seed, reference_seed, seed, rid, action_step, old_split)
            for seed in seeds
        }
        for horizon in horizons:
            for arm in arm_specs(horizon, protocol):
                scenario_id = f"{state['state_id']}__L{horizon}__{arm['arm_id']}"
                if scenario_id in completed:
                    continue
                alpha = np.asarray(arm["alpha"], np.float64)
                offsets = np.zeros((plant.max_steps, 4), np.float64)
                offsets[action_step:action_step + horizon] = alpha
                outcomes = []
                for seed in seeds:
                    terms, trace = full_fn(
                        jnp.asarray(starts[rid]), draws_by_seed[seed],
                        jnp.asarray(offsets))
                    at_state = np.asarray(jax.device_get(trace["before"][action_step]))
                    error = float(np.max(np.abs(
                        at_state - reference["before"][action_step])))
                    if error > 2e-10:
                        raise RuntimeError(
                            f"reference replay mismatch {state['state_id']}: {error}")
                    summary = outcome_summary(jax.device_get(terms))
                    summary.update(
                        seed=seed, reference_state_max_abs_error=error,
                        chunk=chunk_diagnostics(trace, action_step, horizon, alpha),
                    )
                    outcomes.append(summary)
                deadlocks = int(sum(row["D_H"] for row in outcomes))
                records.append(dict(
                    schema="rrisk_true_chunk_arm_v1", scenario_id=scenario_id,
                    stage=args.stage, state=state, horizon_steps=horizon,
                    horizon_seconds=horizon * plant.dt, **arm,
                    continuation_seeds=seeds, continuations=len(outcomes),
                    deadlocks=deadlocks, Q_D=deadlocks / len(outcomes),
                    outcomes=outcomes,
                ))
                completed.add(scenario_id)
                atomic_json(records_path, records)
                runtime.update(
                    completed_arm_scenarios=len(records),
                    completed_continuations=sum(row["continuations"] for row in records),
                    updated=time.time())
                atomic_json(runtime_path, runtime)
                print(f"{len(records)}/{scenario_count} {scenario_id} "
                      f"D={deadlocks}/{len(outcomes)}", flush=True)
    runtime.update(completed=time.time(), elapsed_seconds=time.time()-runtime["started"])
    atomic_json(runtime_path, runtime)
    atomic_json(args.out / "complete.json", dict(
        complete=True, stage=args.stage, training_steps=0,
        protocol_sha256=protocol_sha, arm_scenarios=len(records),
        continuations=sum(row["continuations"] for row in records)))


if __name__ == "__main__":
    main()
