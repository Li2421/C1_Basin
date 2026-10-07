"""Freeze the small candidate pool and schedule only missing seeds."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from diagnostics.success_basin_deformation_decomposition.analyze import load_inventory, eta_key


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DIRECTIONAL = ROOT / "diagnostics/success_basin_deformation_directional_refinement"
SBMA = ROOT / "diagnostics/success_basin_multimodality"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
STATES = ("D2_pair228", "D4_pair227")
NEW_SEEDS = tuple(range(95108001, 95108033))
CANDIDATES = {
    "D2_pair228": (
        (1.0, 0.0, 0.25),
        (0.40625, -0.5, 0.0),
        (0.4375, -0.53125, 0.0),
    ),
    "D4_pair227": (
        (1.0, 0.0, 0.25),
        (0.375, -0.4375, -0.0625),
        (0.40625, -0.4375, -0.0625),
    ),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name: str, value: object) -> None:
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    _, effective, _, conflicts = load_inventory()
    if conflicts:
        raise AssertionError(conflicts[:3])
    groups = defaultdict(dict)
    for row in effective:
        groups[(row["state_id"], eta_key(row["eta"]))][row["seed"]] = row
    pool, jobs = [], []
    for state in STATES:
        for index, eta in enumerate(CANDIDATES[state]):
            existing = groups[(state, eta)]
            valid = {seed: row for seed, row in existing.items() if not row["execution_error"] and row["outcome"]}
            counts = Counter(row["outcome"] for row in valid.values())
            target = tuple(sorted(valid)) + NEW_SEEDS[: 64 - len(valid)]
            if len(target) != 64 or len(set(target)) != 64:
                raise AssertionError((state, eta, len(valid), len(target), len(set(target))))
            missing = [seed for seed in target if seed not in valid]
            first_batch = missing[:16]
            pool.append({
                "state_id": state, "eta": list(eta), "priority": index,
                "existing_valid_seeds": len(valid),
                "existing_counts": {name: counts[name] for name in ("success", "deadlock", "timeout", "collision")},
                "existing_mean_success_J_def": float(np.mean([row["stored_J_def"] for row in valid.values() if row["outcome"] == "success"])),
                "target_seeds": list(target), "missing_seeds": missing,
                "stage1_missing_seeds": first_batch,
                "selection_reason": "robust anchor" if eta == (1.0, 0.0, 0.25) else "lowest-J clean 32/32 local candidate",
            })
            for seed in first_batch:
                jobs.append({"state_id": state, "eta": list(eta), "seed": seed, "candidate_priority": index, "cell_id": "completion_stage1"})
    old = json.loads((DIRECTIONAL / "protocol.json").read_text())
    protocol = {
        "study": "success_basin_feasibility_completion",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "provisional_rule": "success_count >= 63/64",
        "states": old["states"], "environment": old["environment"],
        "checkpoint": old["checkpoint"], "eta_semantics": old["eta_semantics"],
        "J_def": old["J_def"], "candidate_pool": pool,
        "new_seed_pool": list(NEW_SEEDS),
        "early_stop": "Do not schedule remaining seeds after cumulative second failure within the fixed 64-seed target.",
        "source_hashes": {
            "corrector": sha(ROOT / "diagnostics/cl_fhcb/closed_loop.py"),
            "environment": sha(SYSROOT / "single_integrator/environment.py"),
            "projection": sha(SYSROOT / "single_integrator/cbf.py"),
            "retry": sha(SBMA / "exact_projector.py"),
            "checkpoint": sha(Path(old["checkpoint"])),
            "directional_manifest": sha(DIRECTIONAL / "manifest.json"),
        },
    }
    dump("protocol.json", protocol)
    dump("stage1_jobs.json", jobs)
    for state in STATES:
        dump(f"stage1_{state}_jobs.json", [job for job in jobs if job["state_id"] == state])
    print(json.dumps({"candidate_pool": len(pool), "stage1_new_rollouts": len(jobs), "per_state": {state: sum(j["state_id"] == state for j in jobs) for state in STATES}}, indent=2))


if __name__ == "__main__":
    main()
