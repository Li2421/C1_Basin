"""Time one exact continuation without retaining it as a study sample."""

from __future__ import annotations

import argparse
import json
import time

import jax

from diagnostics.cl_fhcb.closed_loop import DiagnosticPhi
from diagnostics.true_q_geometry.geometry_rollout import rollout_from_state
from diagnostics.true_q_geometry.run_q_map import (
    HERE,
    SOURCE_ROOT,
    extract_state,
    locate_assets,
    source_record,
)
from single_integrator.cbf import CBFConfig
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.evaluate import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("cpu", "gpu"), required=True)
    args = parser.parse_args()
    jax.config.update("jax_platform_name", args.platform)
    jax.config.update("jax_enable_x64", True)

    protocol = json.loads((HERE / "predeclared_protocol.json").read_text())
    stage_manifest = json.loads((SOURCE_ROOT / "manifest.json").read_text())
    assets = locate_assets(None)
    checkpoint = assets / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
    policy, provenance = load_policy(checkpoint)
    config = Config(**provenance["evaluation_environment"])
    goals = GiveWayEnv(config).goals
    spec = next(x for x in protocol["state_selection"] if x["state_id"] == "D4_pair227")
    record = source_record(stage_manifest, spec["pair_id"], spec["source_phi"])
    state = extract_state(SOURCE_ROOT / record["relative_path"], spec, goals)

    started = time.perf_counter()
    trace = rollout_from_state(
        policy,
        state,
        config,
        DiagnosticPhi(0.0, 0.0, 0.0, name="benchmark_p0"),
        flow_seed=51999,
        rollout_index=spec["pair_id"],
        cbf_config=CBFConfig(),
    )
    # Synchronize any outstanding device work before stopping the clock.
    jax.block_until_ready(jax.numpy.asarray(trace["u_flow"][-1]))
    print(json.dumps({
        "platform": args.platform,
        "devices": [str(x) for x in jax.devices()],
        "elapsed_seconds": time.perf_counter() - started,
        "physical_steps": len(trace["event"]),
        "outcome": str(trace["outcome"]),
    }))


if __name__ == "__main__":
    main()
