"""Source-only H100 nominal-response upper bound, not a deployment choice.

This is deliberately eta-independent and cached once per state/controller.
H100 is at most ~14% of a complete task horizon. It tests whether the H20
aliasing comes from insufficient future lookahead rather than model fitting.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import rich_context as rc
from .intervention import OLD, OUT, ROOT, read

rc.PROTOCOL = {**rc.PROTOCOL, "horizon_steps": 100,
               "purpose": "source-only nominal future response identifiability upper bound"}
rc.OUT = OUT / "h100_nominal"


def run(scene, shard, shards):
    if scene == "toy_giveway":
        import jax
        swap = ROOT / "diagnostics/orthoflow3_controller_conditioning_probe_v1"
        pair_rows = read(swap / "pair_manifest.json")
        template = dict(next(s["physical"] for s in read(OLD / "states.json") if s["scenario"] == scene))
        base = rc.RichRuntime(scene)
        controllers = [(None, base),
                       (read(swap / "protocol.json")["flow_sha256"]["1"],
                        rc.RichRuntime(scene, read(swap / "protocol.json")["flow_paths"]["1"]))]
        states = {}
        for r in pair_rows:
            if r["state_uid"] in states:
                continue
            env = base.core.make()
            env.reset(np.asarray(r["initial_positions"]))
            flow = base.base_flow(env, jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(42), r["rollout_id"]), 0))
            physical = dict(template)
            physical.update(positions=env.positions.tolist(), velocities=env.velocities.tolist(),
                            goals=env.goals.tolist(), flow=flow.tolist())
            states[r["state_uid"]] = physical
    else:
        old = read(OLD / "states.json")
        pair_rows = read(OUT / "second_variant/pairs.json")
        states = {r["state_uid"]: old[r["state_index"]]["physical"] for r in pair_rows if r["scene"] == scene}
        first = read(OUT / "balanced_expansion/protocol.json")["profiles"][scene]
        second = read(OUT / "second_variant/protocol.json")["profiles"][scene]
        controllers = [(None, rc.RichRuntime(scene)),
                       (first["alternate_flow_sha256"], rc.RichRuntime(scene, first["alternate_path"])),
                       (second["alternate_flow_sha256"], rc.RichRuntime(scene, second["alternate_path"]))]
    jobs = [(uid, physical, sha, runtime) for uid, physical in sorted(states.items())
            for sha, runtime in controllers]
    results = []
    for i, (uid, physical, sha, runtime) in enumerate(jobs):
        if i % shards != shard:
            continue
        result = rc.cached(runtime, {"state_uid": uid, "physical": physical}, np.zeros(3))
        results.append({"state_uid": uid, "controller_sha": sha,
                        "valid": result["valid"], "error": result.get("error"),
                        "nominal": result["features"]["mean"][:10] if result["valid"] else None})
        print(json.dumps({"scene": scene, "shard": shard, "done": len(results)}), flush=True)
    path = rc.OUT / "coverage" / f"{scene}_{shard}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"scene": scene, "shard": shard, "complete": len(results),
                      "valid": sum(r["valid"] for r in results)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=("toy_giveway", "ring_exchange"), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    run(args.scene, args.shard, args.shards)
