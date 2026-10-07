"""H20 source-only physical response features for the second controller variant."""
from __future__ import annotations

import argparse
import json

from . import h20_features as h20
from .intervention import OLD, read
from .second_variant import OUT


def run(scene, shard, shards):
    profile = read(OUT / "protocol.json")["profiles"][scene]
    pairs = [r for r in read(OUT / "pairs.json") if r["scene"] == scene]
    states = read(OLD / "states.json")
    runtime = h20.rc.RichRuntime(scene, profile["alternate_path"])
    rows = []
    for i, pair in enumerate(pairs):
        if i % shards != shard:
            continue
        state = {"state_uid": pair["state_uid"],
                 "physical": states[pair["state_index"]]["physical"]}
        result = h20.rc.cached(runtime, state, pair["eta"])
        rows.append({"state_uid": pair["state_uid"], "eta_uid": pair["eta_uid"],
                     "valid": result["valid"], "error": result.get("error")})
        if len(rows) % 8 == 0:
            print(json.dumps({"scene": scene, "shard": shard, "done": len(rows)}), flush=True)
    path = OUT / "h20_coverage" / f"{scene}_{shard}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"scene": scene, "shard": shard, "complete": len(rows),
                      "valid": sum(r["valid"] for r in rows)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=("double_bottleneck", "four_way_intersection", "ring_exchange"), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    run(args.scene, args.shard, args.shards)
