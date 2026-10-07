"""Audit existing local directions and freeze the first directional expansion."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREV = ROOT / "diagnostics/success_basin_deformation_refinement"
DEFORMATION = ROOT / "diagnostics/success_basin_deformation"
SBMA = ROOT / "diagnostics/success_basin_multimodality"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
STATES = ("D1_pair231", "D2_pair228", "D4_pair227")
SEEDS = list(range(95101001, 95101017)) + list(range(95105001, 95105017))
CURRENT = {
    "D1_pair231": (0.59375, -0.3125, -0.125),
    "D2_pair228": (0.4375, -0.5625, 0.0),
    "D4_pair227": (0.375, -0.4375, -0.0625),
}


def key(values):
    return tuple(round(float(v), 12) for v in values)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name: str, value: object) -> None:
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def existing_groups():
    groups = defaultdict(dict)
    for line in (DEFORMATION / "per_rollout_j_def.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["cohort"] != "phase_a" or row["J_def"] is None:
            continue
        groups[(row["state_id"], key(row["eta"]))][row["seed"]] = {
            "seed": row["seed"], "J_def": row["J_def"],
            "outcome": row["terminal_outcome"], "steps": row["episode_length_steps"],
        }
    for path in sorted((PREV / "raw").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        for row in manifest["records"]:
            if row["J_def"] is None:
                continue
            groups[(row["state_id"], key(row["eta"]))][row["seed"]] = {
                "seed": row["seed"], "J_def": row["J_def"],
                "outcome": row["outcome"], "steps": row["steps"],
            }
    return groups


def paired_tendency(groups, state, center, neighbor, label):
    a, b = groups[(state, key(center))], groups[(state, key(neighbor))]
    seeds = sorted(set(a) & set(b))
    diff = np.asarray([b[s]["J_def"] - a[s]["J_def"] for s in seeds])
    mean = float(diff.mean())
    half = float(t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff)))
    return {
        "direction": label, "center": list(center), "neighbor": list(neighbor),
        "paired_n": len(seeds), "neighbor_minus_center_mean_J_def": mean,
        "two_sided_t_95_CI": [mean - half, mean + half],
        "neighbor_outcomes": {name: sum(row["outcome"] == name for row in b.values()) for name in ("success", "deadlock", "timeout", "collision")},
        "interpretation": "decreasing" if mean < 0 else "increasing",
    }


def phase1_points():
    return {
        "D1_pair231": {
            (0.5625, -0.3125, -0.125),
            (0.578125, -0.3125, -0.125),
            (0.59375, -0.34375, -0.125),
            (0.59375, -0.375, -0.125),
            (0.59375, -0.28125, -0.125),
            (0.59375, -0.3125, -0.21875),
            (0.59375, -0.3125, -0.25),
            (0.578125, -0.34375, -0.125),
            (0.578125, -0.3125, -0.1875),
        },
        "D2_pair228": {
            (0.40625, -0.5625, 0.0),
            (0.375, -0.5625, 0.0),
            (0.4375, -0.53125, 0.0),
            (0.4375, -0.5, 0.0),
            (0.4375, -0.59375, 0.0),
            (0.40625, -0.53125, 0.0),
            (0.40625, -0.5, 0.0),
            (0.375, -0.53125, 0.0),
            (0.375, -0.5, 0.0),
        },
        "D4_pair227": {
            (0.359375, -0.4375, -0.0625),
            (0.34375, -0.4375, -0.0625),
            (0.375, -0.40625, -0.0625),
            (0.375, -0.375, -0.0625),
            (0.375, -0.46875, -0.0625),
            (0.375, -0.4375, -0.09375),
            (0.375, -0.4375, -0.125),
            (0.359375, -0.40625, -0.0625),
            (0.375, -0.40625, -0.09375),
            (0.359375, -0.40625, -0.09375),
        },
    }


def main():
    groups = existing_groups()
    audits = {
        "D1_pair231": [
            paired_tendency(groups, "D1_pair231", CURRENT["D1_pair231"], (0.5625, -0.3125, -0.125), "eta1 decrease"),
            paired_tendency(groups, "D1_pair231", CURRENT["D1_pair231"], (0.625, -0.3125, -0.125), "eta1 increase"),
            paired_tendency(groups, "D1_pair231", CURRENT["D1_pair231"], (0.59375, -0.25, -0.125), "eta2 increase"),
            paired_tendency(groups, "D1_pair231", CURRENT["D1_pair231"], (0.59375, -0.3125, -0.1875), "eta3 decrease"),
            paired_tendency(groups, "D1_pair231", CURRENT["D1_pair231"], (0.59375, -0.3125, -0.0625), "eta3 increase"),
        ],
        "D2_pair228": [
            paired_tendency(groups, "D2_pair228", CURRENT["D2_pair228"], (0.5, -0.5625, 0.0), "eta1 increase"),
            paired_tendency(groups, "D2_pair228", CURRENT["D2_pair228"], (0.4375, -0.625, 0.0), "eta2 decrease"),
            paired_tendency(groups, "D2_pair228", CURRENT["D2_pair228"], (0.4375, -0.5625, -0.0625), "eta3 decrease"),
            paired_tendency(groups, "D2_pair228", CURRENT["D2_pair228"], (0.4375, -0.5625, 0.0625), "eta3 increase"),
        ],
        "D4_pair227": [
            paired_tendency(groups, "D4_pair227", CURRENT["D4_pair227"], (0.34375, -0.4375, -0.0625), "eta1 decrease"),
            paired_tendency(groups, "D4_pair227", CURRENT["D4_pair227"], (0.40625, -0.4375, -0.0625), "eta1 increase"),
            paired_tendency(groups, "D4_pair227", CURRENT["D4_pair227"], (0.375, -0.5, -0.0625), "eta2 decrease"),
            paired_tendency(groups, "D4_pair227", CURRENT["D4_pair227"], (0.375, -0.4375, 0.0), "eta3 increase"),
        ],
    }
    audit = {
        "computed_before_new_outcomes": True,
        "audits": audits,
        "predeclared_tendencies": {
            "D1_pair231": {
                "eta1": "J_def falls toward smaller eta1, but reliability already degrades at 0.5625; refine the success transition.",
                "eta2": "outward negative direction untested at the current eta1; prior nearby slices suggest it can reduce effective correction, so probe it.",
                "eta3": "negative direction is nearly flat between -0.125 and -0.1875; probe one further negative step but do not broad-sweep.",
            },
            "D2_pair228": {
                "eta1": "positive direction raises J_def; smaller eta1 is the unresolved boundary direction.",
                "eta2": "more-negative eta2 raises J_def, so probe less-negative eta2 outside the current local box.",
                "eta3": "both tested signs raise J_def; eta3=0 is locally supported and is not expanded.",
            },
            "D4_pair227": {
                "eta1": "smaller eta1 sharply lowers raw mean J_def but crosses into unreliable outcomes; refine the transition.",
                "eta2": "more-negative eta2 raises J_def, so expand toward less-negative eta2.",
                "eta3": "eta3< -0.0625 is untested while eta3=0 is higher; probe the negative direction.",
            },
        },
    }
    dump("local_direction_audit.json", audit)

    previous_protocol = json.loads((PREV / "protocol.json").read_text())
    points = {state: sorted(key(point) for point in values) for state, values in phase1_points().items()}
    jobs = []
    reused = {}
    for state, state_points in points.items():
        reused[state] = {}
        for eta in state_points:
            have = sorted(set(groups[(state, eta)]) & set(SEEDS))
            reused[state][str(eta)] = have
            for seed in sorted(set(SEEDS) - set(have)):
                jobs.append({"state_id": state, "eta": list(eta), "seed": seed, "cell_id": "directional_phase1"})
    protocol = {
        "study": "success_basin_deformation_directional_refinement",
        "phase": "directional_phase1",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "states": previous_protocol["states"],
        "environment": previous_protocol["environment"],
        "checkpoint": previous_protocol["checkpoint"],
        "eta_semantics": previous_protocol["eta_semantics"],
        "J_def": previous_protocol["J_def"],
        "success_criterion": previous_protocol["success_criterion"],
        "current_minima": {state: list(eta) for state, eta in CURRENT.items()},
        "phase1_points": {state: [list(point) for point in state_points] for state, state_points in points.items()},
        "target_seeds": SEEDS,
        "common_random_numbers": True,
        "reused_seed_map": reused,
        "new_rollouts": len(jobs),
        "source_hashes": {
            "corrector": sha(ROOT / "diagnostics/cl_fhcb/closed_loop.py"),
            "environment": sha(SYSROOT / "single_integrator/environment.py"),
            "projection": sha(SYSROOT / "single_integrator/cbf.py"),
            "retry": sha(SBMA / "exact_projector.py"),
            "checkpoint": sha(Path(previous_protocol["checkpoint"])),
            "previous_manifest": sha(PREV / "manifest.json"),
        },
        "stopping_rule": "After Phase 1, allow at most one additional directional step/refinement only where a successful boundary point still decreases J_def or a success transition must be resolved.",
    }
    dump("protocol.json", protocol)
    dump("phase1_jobs.json", jobs)
    for state in STATES:
        dump(f"phase1_{state}_jobs.json", [job for job in jobs if job["state_id"] == state])
    print(json.dumps({
        "new_rollouts": len(jobs),
        "per_state": {state: sum(job["state_id"] == state for job in jobs) for state in STATES},
        "points": {state: len(values) for state, values in points.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
