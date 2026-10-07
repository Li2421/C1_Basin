"""Select only unresolved states for the preregistered 64->128->256 extension."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-n", type=int, choices=(64, 128), required=True)
    parser.add_argument("--target-n", type=int, choices=(128, 256), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (args.prior_n, args.target_n) not in {(64, 128), (128, 256)}:
        raise ValueError((args.prior_n, args.target_n))
    paired_path = HERE / "paired_success_differences.csv"
    with paired_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_state = defaultdict(list)
    for row in rows:
        if int(row["n_matched"]) != args.prior_n:
            continue
        discordant = int(row["Q0_success_Qd_fail"]) + int(row["Qd_success_Q0_fail"])
        if row["effect_status"] == "DELAY_EFFECT_AMBIGUOUS" and discordant > 0:
            by_state[row["state_id"]].append(int(row["delay_steps"]))

    if args.prior_n == 64:
        new_seeds = list(range(95910101, 95910165))
    else:
        new_seeds = list(range(95910201, 95910329))
    output = {
        "selection_rule": (
            "all and only states with at least one nonzero paired success discordance "
            "whose paired bootstrap 95% CI remains unresolved; extend every delay jointly "
            "to retain four-way matched randomness"
        ),
        "selection_uses_effect_magnitude_threshold": False,
        "prior_total_n": args.prior_n,
        "target_total_n": args.target_n,
        "state_ids": sorted(by_state),
        "unresolved_delays_by_state": {key: sorted(value) for key, value in sorted(by_state.items())},
        "delays_to_run_for_each_selected_state": [0, 1, 2, 4],
        "new_seeds": new_seeds,
        "audit_plan_sha256": sha(HERE / "audit_plan.json"),
        "paired_statistics_sha256": sha(paired_path),
    }
    path = HERE / args.output
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
