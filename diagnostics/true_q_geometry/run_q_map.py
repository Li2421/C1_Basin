"""Run the predeclared local full-horizon Q_D map on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from diagnostics.cl_fhcb.closed_loop import AugmentedState, DiagnosticPhi
from diagnostics.true_q_geometry.geometry_rollout import (
    diagnostic_state_features,
    rollout_from_state,
)
from single_integrator.cbf import CBFConfig
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.evaluate import load_policy


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "diagnostics/cl_fhcb_qualification/raw/stage1"


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate_assets(explicit):
    candidates = ([] if explicit is None else [explicit]) + [ROOT, ROOT.parent / "02_C1_Toy_GiveWay"]
    for candidate in candidates:
        checkpoint = candidate / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
        if checkpoint.is_file():
            return candidate.resolve()
    raise FileNotFoundError("frozen assets unavailable")


def source_record(stage_manifest, pair_id, phi_name):
    selected = [item for item in stage_manifest["records"]
                if item["pair_id"] == pair_id and item["flow_seed"] == 19073
                and item["phi_name"] == phi_name]
    if len(selected) != 1:
        raise ValueError("source trace is not unique")
    return selected[0]


def extract_state(source_path: Path, spec: dict, goals: np.ndarray) -> AugmentedState:
    with np.load(source_path) as data:
        before = np.asarray(data["positions_before"])
        after = np.asarray(data["positions_after"])
        velocities = np.asarray(data["last_velocity_before"])
        candidate = np.asarray(data["candidate_since_before"])
    step = int(spec["start_step"])
    sequence = np.concatenate((before[:1], after), axis=0)
    if step >= len(sequence):
        raise ValueError("state step is outside cached trajectory")
    lo = max(0, step - 40)
    history = np.linalg.norm(goals[None] - sequence[lo:step + 1], axis=-1)
    since = int(candidate[step])
    return AugmentedState(
        positions=sequence[step].copy(),
        last_velocity=velocities[step].copy(),
        step=step,
        error_history=history,
        history_start_step=lo,
        candidate_since=None if since < 0 else since,
        terminal="running",
    )


def save_state(path: Path, state: AugmentedState):
    np.savez_compressed(
        path,
        positions=state.positions,
        last_velocity=state.last_velocity,
        step=np.asarray(state.step),
        error_history=state.error_history,
        history_start_step=np.asarray(state.history_start_step),
        candidate_since=np.asarray(-1 if state.candidate_since is None else state.candidate_since),
    )


def sliced_cached_trace(source_path: Path, start_step: int, phi, flow_seed):
    with np.load(source_path) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    terminal = len(arrays["event"])
    sl = slice(start_step, terminal)
    keep = {}
    for name in (
        "positions_before", "positions_after", "last_velocity_before", "u_flow",
        "u_safe", "g", "w", "u_exec", "event", "candidate_since_before",
        "stuck_timer_after", "flow_key_data", "first_active", "second_active",
        "first_min_cbf_residual", "second_min_cbf_residual",
    ):
        keep[name] = arrays[name][sl]
    # Older cached qualification traces did not retain this monitor field.
    keep["window_progress_after"] = np.full((terminal - start_step, 2), np.nan)
    keep.update(
        outcome=str(keep["event"][-1]), start_step=start_step,
        terminal_step=terminal, phi=np.asarray(phi), flow_seed=flow_seed,
    )
    return keep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    protocol = json.loads((HERE / "predeclared_protocol.json").read_text())
    stage_manifest = json.loads((SOURCE_ROOT / "manifest.json").read_text())
    assets = locate_assets(args.asset_root)
    checkpoint = assets / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
    if digest(checkpoint) != protocol["frozen_checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash mismatch")
    policy, provenance = load_policy(checkpoint)
    config = Config(**provenance["evaluation_environment"])
    goals = GiveWayEnv(config).goals

    raw = HERE / "raw/q_map"
    if raw.exists() and any(raw.iterdir()):
        raise FileExistsError(f"output must be empty: {raw}")
    traces_dir = raw / "traces"
    states_dir = raw / "states"
    traces_dir.mkdir(parents=True)
    states_dir.mkdir()

    states = {}
    state_catalog = []
    source_records = {}
    for spec in protocol["state_selection"]:
        record = source_record(stage_manifest, spec["pair_id"], spec["source_phi"])
        source_path = SOURCE_ROOT / record["relative_path"]
        if digest(source_path) != record["sha256"]:
            raise RuntimeError("cached source hash mismatch")
        state = extract_state(source_path, spec, goals)
        states[spec["state_id"]] = state
        source_records[spec["state_id"]] = (record, source_path)
        state_path = states_dir / f"{spec['state_id']}.npz"
        save_state(state_path, state)
        state_catalog.append({
            **spec,
            "source_trace": record["id"],
            "source_trace_sha256": record["sha256"],
            "state_file": str(state_path.relative_to(raw)),
            "state_file_sha256": digest(state_path),
            "features": diagnostic_state_features(state, config, goals),
        })
    (raw / "state_catalog.json").write_text(json.dumps(state_catalog, indent=2) + "\n")

    probes = {name: tuple(values) for name, values in protocol["phi_probes"].items()}
    cached_seed = protocol["stage_1"]["cached_common_seed"]
    independent = protocol["stage_1"]["new_independent_flow_seeds"]
    all_seeds = [cached_seed] + independent
    proposed = protocol["stage_1"]["proposed_new_continuations"]
    estimated = protocol["stage_1"]["estimated_physical_steps_upper"]
    print(f"number of proposed rollouts: {proposed}", flush=True)
    print(f"estimated physical steps: <= {estimated}", flush=True)
    print("CPU/GPU: CPU", flush=True)
    print("exact scientific question: does true full-horizon Q_D vary across the predeclared local closed-loop phi probes?", flush=True)

    records = []
    completed_new = 0
    reused = 0
    for spec in protocol["state_selection"]:
        state_id = spec["state_id"]
        state = states[state_id]
        cached_probe = "goal_p2" if spec["source_phi"] == "goal025" else "p0"
        for probe_name, vector in probes.items():
            phi = DiagnosticPhi(*vector, name=probe_name)
            for flow_seed in all_seeds:
                is_cached = flow_seed == cached_seed and probe_name == cached_probe
                trace_id = f"{state_id}__{probe_name}__seed{flow_seed}"
                path = traces_dir / f"{trace_id}.npz"
                if is_cached:
                    _, source_path = source_records[state_id]
                    arrays = sliced_cached_trace(source_path, state.step, vector, flow_seed)
                    origin = "reused_compatible_cached_tail"
                    reused += 1
                else:
                    arrays = rollout_from_state(
                        policy, state, config, phi, flow_seed,
                        rollout_index=spec["pair_id"], cbf_config=CBFConfig(),
                    )
                    origin = "new_full_continuation"
                    completed_new += 1
                np.savez_compressed(path, **arrays)
                records.append({
                    "trace_id": trace_id,
                    "state_id": state_id,
                    "probe_name": probe_name,
                    "phi": list(vector),
                    "flow_seed": flow_seed,
                    "origin": origin,
                    "relative_path": str(path.relative_to(raw)),
                    "sha256": digest(path),
                    "start_step": int(arrays["start_step"]),
                    "terminal_step": int(arrays["terminal_step"]),
                    "continuation_steps": int(len(arrays["event"])),
                    "outcome": str(arrays["outcome"]),
                })
                if not is_cached:
                    print(
                        f"q-map rollout {completed_new}/{proposed}: state={state_id} "
                        f"phi={probe_name} seed={flow_seed} steps={len(arrays['event'])} "
                        f"outcome={arrays['outcome']}", flush=True,
                    )
    if completed_new != proposed or reused != 10:
        raise AssertionError((completed_new, reused))
    manifest = {
        "schema": "c1_true_q_local_map_raw_v1",
        "device": [str(device) for device in jax.devices()],
        "new_continuations": completed_new,
        "cached_continuations_reused": reused,
        "total_balanced_continuations": len(records),
        "actual_new_physical_steps": int(sum(x["continuation_steps"] for x in records if x["origin"] == "new_full_continuation")),
        "cached_physical_steps": int(sum(x["continuation_steps"] for x in records if x["origin"] != "new_full_continuation")),
        "checkpoint_sha256": digest(checkpoint),
        "protocol_sha256": digest(HERE / "predeclared_protocol.json"),
        "state_catalog_sha256": digest(raw / "state_catalog.json"),
        "records": records,
    }
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "new_continuations", "cached_continuations_reused", "actual_new_physical_steps"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
