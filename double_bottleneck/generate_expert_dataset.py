"""Generate the Gate-A-approved, small Double-Bottleneck expert pilot.

Nothing is solved or written unless ``--gate-a-approved`` is supplied.  The
fixed pilot contains twelve initial Markov states and two opposite
first-direction coordination hypotheses per state (24 rollouts total).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

from .environment import Config
from .expert import CentralizedExpert
from .expert_dataset import (
    analyze_mode_diversity,
    assign_grouped_split,
    deduplicate_records,
    initialize_environment,
    pilot_initial_condition_specs,
    pilot_rollout_requests,
    record_from_plan,
    save_dataset,
    validate_records,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "diagnostics" / "double_bottleneck_expert_dataset"


def _source_hashes() -> dict[str, str]:
    names = (
        "double_bottleneck/environment.py",
        "double_bottleneck/scenario.py",
        "double_bottleneck/expert.py",
        "double_bottleneck/expert_dataset.py",
        "double_bottleneck/generate_expert_dataset.py",
    )
    return {
        name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest() for name in names
    }


def generate_pilot(
    output: Path,
    split_seed: int = 17,
    val_fraction: float = 0.25,
) -> dict:
    """Solve, validate, quality-gate, then serialize the 24-rollout pilot."""

    config = Config()
    specs = pilot_initial_condition_specs(config)
    requests = pilot_rollout_requests(specs)
    if len(specs) != 12 or len(requests) != 24:
        raise AssertionError("the approved pilot design must remain 12 starts and 24 rollouts")

    expert = CentralizedExpert()
    records = []
    started = time.perf_counter()
    for index, (spec, hypothesis) in enumerate(requests):
        env = initialize_environment(config, spec)
        plan = expert.plan_hypothesis(env, hypothesis)
        rollout_id = f"pilot_{index:03d}__{hypothesis.first_direction}"
        record = record_from_plan(plan, env, spec, rollout_id)
        records.append(record)
        print(
            f"[{index + 1:02d}/{len(requests)}] {spec.condition_id} "
            f"{hypothesis.first_direction}: {record.terminal_reason} "
            f"steps={record.episode_steps} pair={record.min_swept_pair_surface_distance:.5f} "
            f"wall={record.min_swept_wall_clearance:.5f}"
        )

    unique, duplicates = deduplicate_records(records)
    if duplicates:
        raise RuntimeError(
            "pilot contains exact duplicate trajectories; refusing to serialize: "
            + json.dumps(duplicates, sort_keys=True)
        )
    assigned = assign_grouped_split(unique, val_fraction=val_fraction, seed=split_seed)
    report = validate_records(assigned)
    mode_analysis = analyze_mode_diversity(assigned)

    # Gate C: every deliberately requested expert rollout must be clean.  The
    # failure traces remain useful during development, but are not silently
    # admitted to this training pilot.
    if report["rollouts"] != 24 or report["training_eligible_rollouts"] != 24:
        raise RuntimeError(f"pilot expert success gate failed: {json.dumps(report, sort_keys=True)}")
    if report["collision_rate"] != 0.0 or report["timeout_rate"] != 0.0:
        raise RuntimeError(f"pilot safety/completion gate failed: {json.dumps(report, sort_keys=True)}")
    if any(report["regime_counts"][regime] != 8 for regime in report["regime_counts"]):
        raise RuntimeError(f"pilot regime coverage is unbalanced: {report['regime_counts']}")
    if mode_analysis["identical_initial_states_with_multiple_successful_modes"] != 12:
        raise RuntimeError(
            "actual crossing events did not establish two modes for every identical pilot start: "
            + json.dumps(mode_analysis, sort_keys=True)
        )

    elapsed = time.perf_counter() - started
    generation_metadata = {
        "phase": "small_gate_a_approved_pilot",
        "gate_a_authorized": True,
        "large_dataset": False,
        "initial_states": len(specs),
        "requested_rollouts": len(requests),
        "hypotheses_per_state": ["left_to_right_first", "right_to_left_first"],
        "split_protocol": "family_grouped_regime_stratified_v1",
        "split_seed": int(split_seed),
        "val_fraction": float(val_fraction),
        "generation_wall_seconds": elapsed,
        "source_sha256": _source_hashes(),
    }
    return save_dataset(
        output,
        assigned,
        duplicates_dropped=duplicates,
        generation_metadata=generation_metadata,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument(
        "--gate-a-approved",
        action="store_true",
        help="required acknowledgement that centralized expert feasibility has passed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.gate_a_approved:
        raise SystemExit(
            "Refusing to solve or write a dataset before Gate A; pass --gate-a-approved only "
            "after master confirmation."
        )
    manifest = generate_pilot(args.output, args.split_seed, args.val_fraction)
    print(json.dumps(manifest["quality_report"], indent=2, sort_keys=True))
    print(json.dumps(manifest["mode_analysis"], indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
