"""Freeze the cheap completion batch after applying the predeclared early stop."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from diagnostics.success_basin_deformation_decomposition.analyze import eta_key, load_inventory


HERE = Path(__file__).resolve().parent


def dump(name: str, value: object) -> None:
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    protocol = json.loads((HERE / "protocol.json").read_text())
    _, effective, _, conflicts = load_inventory()
    if conflicts:
        raise AssertionError(conflicts[:3])
    rows = defaultdict(dict)
    for row in effective:
        rows[(row["state_id"], eta_key(row["eta"]))][int(row["seed"])] = row
    for state in ("D2_pair228", "D4_pair227"):
        manifest = json.loads((HERE / "raw" / f"stage1_{state}" / "manifest.json").read_text())
        for row in manifest["records"]:
            key = (row["state_id"], eta_key(row["eta"]))
            if int(row["seed"]) in rows[key]:
                raise AssertionError(("stage1 accidentally reran tuple", key, row["seed"]))
            rows[key][int(row["seed"])] = row

    jobs = []
    decisions = []
    for candidate in protocol["candidate_pool"]:
        state = candidate["state_id"]
        eta = eta_key(candidate["eta"])
        valid = {
            seed: row for seed, row in rows[(state, eta)].items()
            if row.get("outcome") and not row.get("execution_error")
        }
        target = set(candidate["target_seeds"])
        observed = {seed: row for seed, row in valid.items() if seed in target}
        counts = Counter(row["outcome"] for row in observed.values())
        missing = sorted(target - set(observed))
        eligible = counts["success"] + len(missing) >= 63
        reason = "already complete" if not missing else ("still compatible with 63/64" if eligible else "early stop: at least two physical failures")
        decisions.append({
            "state_id": state,
            "eta": list(eta),
            "observed_target_seeds": len(observed),
            "counts": {name: counts[name] for name in ("success", "deadlock", "timeout", "collision")},
            "missing_target_seeds": missing,
            "eligible_for_completion": eligible,
            "reason": reason,
        })
        # Anchors have no missing seeds. Complete only promising candidates.
        if eligible:
            for seed in missing:
                jobs.append({"state_id": state, "eta": list(eta), "seed": seed, "candidate_priority": candidate["priority"], "cell_id": "completion_stage2"})

    dump("stage2_decisions.json", decisions)
    dump("stage2_jobs.json", jobs)
    for state in ("D2_pair228", "D4_pair227"):
        dump(f"stage2_{state}_jobs.json", [job for job in jobs if job["state_id"] == state])
    print(json.dumps({
        "stage2_new_rollouts": len(jobs),
        "per_state": {state: sum(job["state_id"] == state for job in jobs) for state in ("D2_pair228", "D4_pair227")},
        "decisions": decisions,
    }, indent=2))


if __name__ == "__main__":
    main()
