#!/usr/bin/env python3
"""Read-only verification of the exact first-step t=0 conditioning replay."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/orthoflow3_t0_basin_structure_v1"
for value in (str(SYSROOT), str(ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    from diagnostics.gphi_training_dataset_v2.finalize_dataset import restore_full
    from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
    from diagnostics.single_segment_recovery_training_v1.state_machine import FiniteHistoryView
    from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
    from single_integrator.cbf import CBFConfig, barrier_constraints
    from single_integrator.environment import Config, bounded_nominal
    from single_integrator.evaluate import load_policy

    integrity = json.load(open(ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1/integrity_audit.json"))
    config = Config(**integrity["environment"])
    cbf = CBFConfig(**integrity["cbf"])
    policy, _ = load_policy(SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl")
    sample = jax.jit(lambda obs, key: policy.sample_actions(obs[None], seed=key)[0])
    states = json.load(open(HERE / "t0_state_manifest.json"))["attempted_states"]
    features = np.load(HERE / "synthetic_qdir/conditioning_features.npz")["features"]
    rows = []
    for state in states:
        env = restore_full(Path(state["state_file"]), config)
        base = jax.random.fold_in(jax.random.PRNGKey(int(state["flow_seed"])), int(state["rng_namespace"]))
        key = jax.random.fold_in(base, int(state["absolute_step"]))
        action = np.asarray(sample(jnp.asarray(env.observation(), dtype=jnp.float32), key))
        flow = bounded_nominal(action, config.max_speed)
        A, lower, _ = barrier_constraints(env.snapshot(), cbf)
        safe, _, _, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
        feature, _ = StartupAwareFeatureBuilder().build(
            FiniteHistoryView(env), {"u_flow": flow, "u_safe": safe}, config, cbf
        )
        err = float(np.max(np.abs(np.asarray(feature) - features[int(state["feature_index"])])))
        rows.append({"state_id": state["state_id"], "max_abs_error": err, "pass": err <= 1e-10})
    out = {"rows": rows, "maximum_error": max(x["max_abs_error"] for x in rows), "all_pass": all(x["pass"] for x in rows)}
    (HERE / "feature_replay_validation.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2))
    if not out["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
