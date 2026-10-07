"""Generate the small, balanced eight-mode Double-Bottleneck expert dataset.

The dataset keeps the twelve Gate-A-approved initial Markov states and solves
all eight centralized planning hypotheses for each state.  Hypothesis labels
are not accepted as evidence of diversity: serialization is allowed only when
the executed crossing events establish eight distinct successful modes for
every identical start.
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
    all_mode_rollout_requests,
    analyze_mode_diversity,
    assign_grouped_split,
    deduplicate_records,
    initialize_environment,
    pilot_initial_condition_specs,
    record_from_plan,
    save_dataset,
    validate_records,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "diagnostics" / "double_bottleneck_expert_dataset_8mode"


def _source_hashes() -> dict[str, str]:
    names = (
        "double_bottleneck/environment.py",
        "double_bottleneck/scenario.py",
        "double_bottleneck/expert.py",
        "double_bottleneck/expert_dataset.py",
        "double_bottleneck/generate_expert_dataset_8mode.py",
    )
    return {
        name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest() for name in names
    }


def generate_eight_mode_pilot(
    output: Path,
    split_seed: int = 17,
    val_fraction: float = 0.25,
) -> dict:
    """Solve, validate, quality-gate, then serialize the 96-rollout pilot."""

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset path {output}")

    config = Config()
    specs = pilot_initial_condition_specs(config)
    requests = all_mode_rollout_requests(specs)
    if len(specs) != 12 or len(requests) != 96:
        raise AssertionError("the eight-mode pilot must contain 12 starts and 96 rollouts")

    expert = CentralizedExpert()
    records = []
    started = time.perf_counter()
    for index, (spec, hypothesis) in enumerate(requests):
        env = initialize_environment(config, spec)
        plan = expert.plan_hypothesis(env, hypothesis)
        rollout_id = f"eight_mode_{index:03d}__{hypothesis.label}"
        record = record_from_plan(plan, env, spec, rollout_id)
        records.append(record)
        print(
            f"[{index + 1:02d}/{len(requests)}] {spec.condition_id} "
            f"{hypothesis.label}: {record.terminal_reason} "
            f"steps={record.episode_steps} "
            f"pair={record.min_swept_pair_surface_distance:.5f} "
            f"wall={record.min_swept_wall_clearance:.5f}",
            flush=True,
        )

    unique, duplicates = deduplicate_records(records)
    if duplicates:
        raise RuntimeError(
            "eight-mode pilot contains exact duplicate trajectories: "
            + json.dumps(duplicates, sort_keys=True)
        )
    assigned = assign_grouped_split(unique, val_fraction=val_fraction, seed=split_seed)
    report = validate_records(assigned)
    mode_analysis = analyze_mode_diversity(assigned)

    if report["rollouts"] != 96 or report["training_eligible_rollouts"] != 96:
        raise RuntimeError(
            "eight-mode expert success gate failed: " + json.dumps(report, sort_keys=True)
        )
    if any(report[key] != 0.0 for key in ("collision_rate", "timeout_rate", "deadlock_rate")):
        raise RuntimeError(
            "eight-mode safety/completion gate failed: " + json.dumps(report, sort_keys=True)
        )
    if any(report["regime_counts"][regime] != 32 for regime in report["regime_counts"]):
        raise RuntimeError(
            "eight-mode regime coverage is unbalanced: "
            + json.dumps(report["regime_counts"], sort_keys=True)
        )
    if report["split_counts"] != {"train": 72, "val": 24}:
        raise RuntimeError(
            "unexpected grouped split: " + json.dumps(report["split_counts"], sort_keys=True)
        )
    if mode_analysis["unique_successful_mode_signatures"] != 8:
        raise RuntimeError(
            "executed trajectories did not establish eight global modes: "
            + json.dumps(mode_analysis, sort_keys=True)
        )
    groups = mode_analysis["multimodal_initial_state_groups"]
    if len(groups) != 12 or any(len(group["mode_signatures"]) != 8 for group in groups):
        raise RuntimeError(
            "every identical start must realize all eight actual crossing modes: "
            + json.dumps(mode_analysis, sort_keys=True)
        )
    if set(mode_analysis["mode_counts"].values()) != {12}:
        raise RuntimeError(
            "the eight actual modes are not balanced: "
            + json.dumps(mode_analysis["mode_counts"], sort_keys=True)
        )

    generation_metadata = {
        "phase": "small_gate_a_approved_eight_mode_pilot",
        "gate_a_authorized": True,
        "large_dataset": False,
        "initial_states": len(specs),
        "requested_rollouts": len(requests),
        "hypotheses_per_state": 8,
        "actual_modes_required_per_state": 8,
        "split_protocol": "family_grouped_regime_stratified_v1",
        "split_seed": int(split_seed),
        "val_fraction": float(val_fraction),
        "generation_wall_seconds": time.perf_counter() - started,
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
            "Refusing to solve or write an eight-mode dataset before Gate A; pass "
            "--gate-a-approved only after master confirmation."
        )
    manifest = generate_eight_mode_pilot(args.output, args.split_seed, args.val_fraction)
    print(json.dumps(manifest["quality_report"], indent=2, sort_keys=True))
    print(json.dumps(manifest["mode_analysis"], indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
