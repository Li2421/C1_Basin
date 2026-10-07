"""Short function-space Flow/safety signature on a common physical trajectory.

Every controller is queried at identical base-controller states and matched
probe keys. No task rollout, success label, checkpoint ID or scene ID enters
the signature. The first actual action remains the committed base action.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import rich_context as rc
from .intervention import OLD, OUT, ROOT, read

DEST = OUT / "controller_signature"
PROTOCOL = {"steps": 9, "sample_steps": [0, 1, 2, 4, 8],
            "roots": [2026100417, 2026100499],
            "state_path": "shared canonical Flow+safety trajectory for all queried controllers",
            "actual_t0": "committed base action; alternate query at t0 is information only",
            "features_per_query": ["raw_goal", "raw_lateral", "raw_goal_std", "raw_speed",
                                   "safety_intervention", "safe_difference_from_base",
                                   "raw_pair_closing", "safe_pair_closing"],
            "summary": "mean, last, temporal std for each of eight physical features",
            "success_labels_used": False, "full_rollout": False}


def pair_closing(positions, action, speed):
    values = []
    for i in range(len(positions)):
        for j in range(i):
            d = positions[i] - positions[j]
            values.append(-float(np.dot(action[i] - action[j], d)) /
                          max(float(np.linalg.norm(d) * speed), 1e-12))
    return max(values) if values else 0.


def summary(base, queried, physical):
    import jax

    sequences = []
    for root in PROTOCOL["roots"]:
        env = base.core.reset(physical)
        one = []
        for t in range(PROTOCOL["steps"]):
            key = jax.random.fold_in(jax.random.PRNGKey(root), t)
            base_flow = (np.asarray(physical["flow"], float)
                         if t == 0 and physical["flow_committed"] else base.base_flow(env, key))
            base_safe = base.core.project(env, base_flow)
            if t in PROTOCOL["sample_steps"]:
                raw = queried.base_flow(env, key) if queried.alt_flow is None else queried.alt_flow(env, key)
                safe = queried.core.project(env, raw)
                goal = env.goals - env.positions
                goal /= np.maximum(np.linalg.norm(goal, axis=-1, keepdims=True), 1e-12)
                speed = base.core.cfg.max_speed
                forward = np.sum(raw * goal, axis=-1) / speed
                lateral = (raw[:, 1] * goal[:, 0] - raw[:, 0] * goal[:, 1]) / speed
                one.append([float(forward.mean()), float(lateral.mean()), float(forward.std()),
                            float(np.linalg.norm(raw, axis=-1).mean() / speed),
                            float(np.linalg.norm(safe - raw, axis=-1).mean() / speed),
                            float(np.linalg.norm(safe - base_safe, axis=-1).mean() / speed),
                            pair_closing(env.positions, raw, speed),
                            pair_closing(env.positions, safe, speed)])
            env.step(base_safe)
            if env.done:
                break
        if not one:
            raise ValueError("reference trajectory ended before a controller query")
        arr = np.asarray(one)
        sequences.append(np.concatenate((arr.mean(0), arr[-1], arr.std(0))))
    result = np.asarray(sequences)
    assert result.shape == (len(PROTOCOL["roots"]), 24) and np.isfinite(result).all()
    return {"mean": result.mean(0).tolist(), "probe_std": result.std(0).tolist(),
            "sampled_steps_per_root": [len(PROTOCOL["sample_steps"])] * len(PROTOCOL["roots"])}


def run(scene, shard, shards):
    if scene == "toy_giveway":
        import jax
        swap = ROOT / "diagnostics/orthoflow3_controller_conditioning_probe_v1"
        pairs = read(swap / "pair_manifest.json")
        template = dict(next(s["physical"] for s in read(OLD / "states.json") if s["scenario"] == scene))
        base = rc.RichRuntime(scene)
        alt = rc.RichRuntime(scene, read(swap / "protocol.json")["flow_paths"]["1"])
        controllers = [("base", base), ("first", alt)]
        states = {}
        for r in pairs:
            if r["state_uid"] in states: continue
            env = base.core.make();env.reset(np.asarray(r["initial_positions"]))
            flow = base.base_flow(env, jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(42), r["rollout_id"]), 0))
            physical = dict(template)
            physical.update(positions=env.positions.tolist(), velocities=env.velocities.tolist(),
                            goals=env.goals.tolist(), flow=flow.tolist())
            states[r["state_uid"]] = physical
    else:
        pair_rows = read(OUT / "second_variant/pairs.json")
        frozen = read(OLD / "states.json")
        states = {r["state_uid"]: frozen[r["state_index"]]["physical"] for r in pair_rows if r["scene"] == scene}
        first = read(OUT / "balanced_expansion/protocol.json")["profiles"][scene]
        second = read(OUT / "second_variant/protocol.json")["profiles"][scene]
        base = rc.RichRuntime(scene)
        controllers = [("base", base), ("first", rc.RichRuntime(scene, first["alternate_path"])),
                       ("second", rc.RichRuntime(scene, second["alternate_path"]))]
    jobs = [(uid, p, name, rt) for uid, p in sorted(states.items()) for name, rt in controllers]
    out = []
    for i, (uid, physical, name, rt) in enumerate(jobs):
        if i % shards != shard: continue
        try:
            features = summary(base, rt, physical)
            out.append({"scene": scene, "state_uid": uid, "controller": name,
                        "valid": True, "features": features})
        except Exception as exc:
            out.append({"scene": scene, "state_uid": uid, "controller": name,
                        "valid": False, "error": f"{type(exc).__name__}: {exc}"})
        print(json.dumps({"scene": scene, "shard": shard, "done": len(out)}), flush=True)
    path = DEST / "features" / f"{scene}_{shard}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"scene": scene, "shard": shard, "complete": len(out),
                      "valid": sum(r["valid"] for r in out)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=("toy_giveway", "double_bottleneck", "four_way_intersection", "ring_exchange"), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    run(args.scene, args.shard, args.shards)
