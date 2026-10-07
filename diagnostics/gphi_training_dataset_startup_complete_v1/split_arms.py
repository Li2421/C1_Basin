"""Deterministically partition frozen arms without splitting an arm."""

from __future__ import annotations

import argparse
import json

from common import HERE, sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", required=True)
    parser.add_argument("--shards", type=int, required=True, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    source = HERE / args.arms; arms = json.loads(source.read_text())
    paths = []
    for shard in range(args.shards):
        selected = arms[shard::args.shards]
        path = HERE / f"{args.prefix}_shard{shard}_arms.json"
        write_json(path, selected); paths.append({"shard": shard, "path": path.name, "arms": len(selected), "sha256": sha256(path)})
    write_json(HERE / f"{args.prefix}_shard_manifest.json", {"source": args.arms, "source_sha256": sha256(source), "shards": args.shards, "total_arms": len(arms), "partitions": paths})
    print(json.dumps({"total_arms": len(arms), "shard_counts": [row["arms"] for row in paths]}, indent=2))


if __name__ == "__main__":
    main()
