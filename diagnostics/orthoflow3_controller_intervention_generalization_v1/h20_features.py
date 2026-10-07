"""Source-only twenty-step response check, still <3% of task horizon."""
import argparse
import json

import numpy as np

from . import rich_context as rc
from .intervention import OLD, OUT, ROOT, read

# This change is made only after the H8 source-family validation failed;
# no frozen K16 outcome enters this source-only feature experiment.
rc.PROTOCOL = {**rc.PROTOCOL, "horizon_steps": 20,
               "purpose": "source-family information sufficiency test after H8 failed"}
rc.OUT = OUT / "h20"


def run(scene, shard, shards):
    if scene == "toy_giveway":
        import jax
        swap = ROOT / "diagnostics/orthoflow3_controller_conditioning_probe_v1"
        pairs = read(swap / "pair_manifest.json")
        alternate = read(swap / "protocol.json")["flow_paths"]["1"]
        base = rc.RichRuntime(scene)
        alt = rc.RichRuntime(scene, alternate)
        states = read(OLD / "states.json")
        template = dict(next(s["physical"] for s in states if s["scenario"] == scene))
        jobs = []
        for r in pairs:
            env = base.core.make()
            env.reset(np.asarray(r["initial_positions"]))
            flow = base.base_flow(env, jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(42), r["rollout_id"]), 0))
            physical = dict(template)
            physical.update(positions=env.positions.tolist(), velocities=env.velocities.tolist(),
                            goals=env.goals.tolist(), flow=flow.tolist())
            jobs.append((r["state_uid"], r["eta"], physical))
    else:
        protocol = read(OUT / "balanced_expansion/protocol.json")
        profile = protocol["profiles"][scene]
        base = rc.RichRuntime(scene)
        alt = rc.RichRuntime(scene, profile["alternate_path"])
        states = read(OLD / "states.json")
        pairs = [r for r in read(OUT / "balanced_expansion/pairs.json") if r["scene"] == scene]
        jobs = [(r["state_uid"], r["eta"], states[r["state_index"]]["physical"]) for r in pairs]
    results = []
    for j, (uid, eta, physical) in enumerate(jobs):
        if j % shards != shard:
            continue
        state = {"state_uid": uid, "physical": physical}
        a, b = rc.cached(base, state, eta), rc.cached(alt, state, eta)
        results.append({"state_uid": uid, "valid": a["valid"] and b["valid"],
                        "base_error": a.get("error"), "alternate_error": b.get("error")})
        if len(results) % 8 == 0:
            print(json.dumps({"scene": scene, "shard": shard, "done": len(results)}), flush=True)
    path = rc.OUT / "coverage" / f"{scene}_{shard}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"scene": scene, "shard": shard, "complete": len(results),
                      "valid": sum(r["valid"] for r in results)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=("toy_giveway", "double_bottleneck", "four_way_intersection", "ring_exchange"), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    run(args.scene, args.shard, args.shards)
