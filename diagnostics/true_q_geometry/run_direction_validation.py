"""Fresh full-horizon validation of the predeclared true-Q FD direction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from diagnostics.cl_fhcb.closed_loop import AugmentedState, DiagnosticPhi
from diagnostics.true_q_geometry.geometry_rollout import rollout_from_state
from diagnostics.true_q_geometry.run_q_map import HERE, locate_assets
from single_integrator.cbf import CBFConfig
from single_integrator.environment import Config
from single_integrator.evaluate import load_policy


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_state(path: Path) -> AugmentedState:
    with np.load(path) as data:
        candidate = int(data["candidate_since"])
        return AugmentedState(
            positions=np.asarray(data["positions"]),
            last_velocity=np.asarray(data["last_velocity"]),
            step=int(data["step"]),
            error_history=np.asarray(data["error_history"]),
            history_start_step=int(data["history_start_step"]),
            candidate_since=None if candidate < 0 else candidate,
            terminal="running",
        )


def main():
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)
    selection_path = HERE / "raw/directional_selection.json"
    selection = json.loads(selection_path.read_text())
    protocol = json.loads((HERE / "predeclared_protocol.json").read_text())
    assets = locate_assets(None)
    checkpoint = assets / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
    policy, provenance = load_policy(checkpoint)
    config = Config(**provenance["evaluation_environment"])

    raw = HERE / "raw/direction_validation"
    if raw.exists() and any(raw.iterdir()):
        raise FileExistsError(f"output must be empty: {raw}")
    traces_dir = raw / "traces"
    traces_dir.mkdir(parents=True)

    seeds = selection["fresh_flow_seeds"]
    total = len(selection["states"]) * 3 * len(seeds)
    estimated = sum(
        (config.max_steps - load_state(HERE / f"raw/q_map/states/{x['state_id']}.npz").step) * 3 * len(seeds)
        for x in selection["states"]
    )
    print(f"number of proposed rollouts: {total}", flush=True)
    print(f"estimated physical steps: <= {estimated}", flush=True)
    print("CPU/GPU: CPU", flush=True)
    print("exact scientific question: does the fresh full-horizon Q_D ordering follow the central-FD true-risk direction?", flush=True)

    records = []
    actual_steps = 0
    for selected in selection["states"]:
        state_id = selected["state_id"]
        state = load_state(HERE / f"raw/q_map/states/{state_id}.npz")
        catalog = json.loads((HERE / "raw/q_map/state_catalog.json").read_text())
        pair_id = next(x["pair_id"] for x in catalog if x["state_id"] == state_id)
        for policy_name in ("phi_minus", "phi_0", "phi_plus"):
            vector = selected[policy_name]
            phi = DiagnosticPhi(*vector, name=policy_name)
            for seed in seeds:
                trace = rollout_from_state(
                    policy, state, config, phi, seed,
                    rollout_index=pair_id, cbf_config=CBFConfig(),
                )
                trace_id = f"{state_id}__{policy_name}__seed{seed}"
                path = traces_dir / f"{trace_id}.npz"
                np.savez_compressed(path, **trace)
                steps = len(trace["event"])
                actual_steps += steps
                records.append({
                    "trace_id": trace_id,
                    "state_id": state_id,
                    "policy_name": policy_name,
                    "phi": vector,
                    "flow_seed": seed,
                    "outcome": str(trace["outcome"]),
                    "continuation_steps": steps,
                    "relative_path": str(path.relative_to(raw)),
                    "sha256": digest(path),
                })
                print(
                    f"direction rollout {len(records)}/{total}: state={state_id} "
                    f"policy={policy_name} seed={seed} steps={steps} outcome={trace['outcome']}",
                    flush=True,
                )
    manifest = {
        "schema": "c1_true_q_direction_validation_raw_v1",
        "device": [str(x) for x in jax.devices()],
        "new_continuations": len(records),
        "actual_new_physical_steps": actual_steps,
        "selection_sha256": digest(selection_path),
        "checkpoint_sha256": digest(checkpoint),
        "records": records,
    }
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "new_continuations": len(records),
        "actual_new_physical_steps": actual_steps,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
