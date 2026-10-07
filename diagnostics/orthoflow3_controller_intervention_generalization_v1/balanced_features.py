"""Build H8 response features for the source-only balanced intervention batch."""
import argparse

from . import intervention as base
from . import balanced_expansion as expansion
from .rich_context import build_pilot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=base.SCENES, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    expansion.setup()
    build_pilot(args.scene, args.shard, args.shards)


if __name__ == "__main__":
    main()
