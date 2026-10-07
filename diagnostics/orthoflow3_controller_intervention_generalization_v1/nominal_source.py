"""Controller-only H8 source features for the frozen 305-state LOSO dataset."""
import argparse
import json
from pathlib import Path

from .intervention import OLD, OUT
from .rich_context import RichRuntime, cached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True,
                        choices=("toy_giveway", "double_bottleneck", "four_way_intersection", "ring_exchange"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    states = json.loads((OLD / "states.json").read_text())
    runtime = RichRuntime(args.scene)
    entries = []
    for i, state in enumerate(states):
        if state["scenario"] != args.scene or i % args.shards != args.shard:
            continue
        result = cached(runtime, state, [0., 0., 0.])
        entries.append({"state_index": i, "state_uid": state["state_uid"],
                        "split": state["split"], "valid": result["valid"],
                        "context": result["features"]["mean"][:10] if result["valid"] else None,
                        "error": result.get("error")})
        if len(entries) % 16 == 0:
            print(json.dumps({"scene": args.scene, "shard": args.shard, "done": len(entries)}), flush=True)
    path = OUT / "source_nominal" / f"{args.scene}_{args.shard}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2) + "\n")
    print(json.dumps({"scene": args.scene, "shard": args.shard, "complete": len(entries)}), flush=True)


if __name__ == "__main__":
    main()
