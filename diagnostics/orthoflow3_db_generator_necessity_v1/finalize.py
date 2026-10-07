#!/usr/bin/env python3
"""Final deterministic audit for the frozen hard-DB generator-necessity study."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

H = Path(__file__).parent
D = H.parent
PRIOR = D / "orthoflow3_db_shared_mode_transfer_v1"
SCALE = np.array([0.75, 1.0, 0.75], dtype=float)


def read_csv(name: str):
    p = H / name
    return list(csv.DictReader(p.open())) if p.exists() else []


def write_csv(name: str, rows, fields):
    with (H / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def dump(name: str, obj):
    (H / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def truth(x):
    return str(x).lower() in {"1", "true", "yes"}


def fnum(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def main():
    states = json.load(open(H / "hard_state_manifest.json"))["states"]
    state_by_id = {s["state_id"]: s for s in states}
    disc = read_csv("discrete_baselines.csv")
    modes = read_csv("transformed_mode_q64.csv")
    if len(disc) != len(states) or len(modes) != 12 * len(states):
        raise RuntimeError(("incomplete discrete results", len(states), len(disc), len(modes)))

    # Load every exact promoted candidate, if continuous search was needed.
    promoted = []
    for stage in (1, 2):
        for r in read_csv(f"q64_stage{stage}.csv"):
            r = dict(r)
            r["stage"] = stage
            promoted.append(r)
    promoted_by_state = defaultdict(list)
    for r in promoted:
        promoted_by_state[r["state_id"]].append(r)

    mode_by_state = defaultdict(list)
    for r in modes:
        mode_by_state[r["state_id"]].append(r)

    final_stage = json.load(open(H / "stage2_decision.json")) if (H / "stage2_decision.json").exists() else (
        json.load(open(H / "stage1_decision.json")) if (H / "stage1_decision.json").exists() else None
    )
    unresolved = set(final_stage["unresolved_states"]) if final_stage else set()

    comparison = []
    necessity = []
    for d in disc:
        sid = d["state_id"]
        code = max(mode_by_state[sid], key=lambda r: (fnum(r["Q64"]), -int(r["mode_id"])))
        candidates = promoted_by_state[sid]
        best_new = max(candidates, key=lambda r: (fnum(r["Q64"]), -int(r["frozen_order"]))) if candidates else None
        code_b63 = truth(d["oracle_B63"])
        new_b63 = best_new is not None and truth(best_new["B63"])
        if code_b63:
            best_source = "transformed_codebook"
            best_q = fnum(code["Q64"])
            best_b63 = True
            best_eta = np.array([fnum(code[f"eta{i}"]) for i in (1, 2, 3)])
            nearest = 0.0
            status = "CONFIRMED_B63_REUSED_CODEBOOK"
        elif new_b63:
            best_source = f"continuous_stage{best_new['stage']}"
            best_q = fnum(best_new["Q64"])
            best_b63 = True
            best_eta = np.array([fnum(best_new[f"eta{i}"]) for i in (1, 2, 3)])
            nearest = fnum(best_new["nearest_mode_distance"])
            status = "GENERATOR_NECESSITY_CONFIRMED"
        else:
            best_source = "continuous_search_no_B63"
            best_q = fnum(best_new["Q64"]) if best_new else np.nan
            best_b63 = False
            best_eta = np.array([fnum(best_new[f"eta{i}"]) for i in (1, 2, 3)]) if best_new else np.full(3, np.nan)
            nearest = fnum(best_new["nearest_mode_distance"]) if best_new else np.nan
            status = "CONTINUOUS_ORACLE_UNRESOLVED" if sid in unresolved else "SEARCH_NOT_REQUIRED_OR_NOT_RUN"
        row = dict(
            state_id=sid,
            source_group=d["source_group"],
            hard_rank=int(d["hard_rank"]),
            hard_score=fnum(d["hard_score"]),
            regime=d["regime"],
            fixed_B63=truth(d["fixed_B63"]),
            fixed_Q64=fnum(d["fixed_Q64"]),
            selector_B63=truth(d["selector_B63"]),
            selector_Q64=fnum(d["selector_Q64"]),
            codebook_oracle_B63=code_b63,
            codebook_oracle_Q64=fnum(d["oracle_Q64"]),
            feasible_modes=int(d["feasible_modes"]),
            continuous_oracle_status=status,
            continuous_oracle_B63=best_b63,
            continuous_oracle_Q64=best_q,
            continuous_best_source=best_source,
            continuous_eta1=best_eta[0],
            continuous_eta2=best_eta[1],
            continuous_eta3=best_eta[2],
            nearest_transformed_mode_distance=nearest,
            generator_necessity=(not code_b63 and new_b63),
        )
        comparison.append(row)
        if row["generator_necessity"]:
            necessity.append(row)

    fields = list(comparison[0])
    write_csv("continuous_vs_codebook.csv", comparison, fields)
    write_csv("continuous_search_results.csv", comparison, fields)
    write_csv("generator_necessity_states.csv", necessity, fields)

    # Compact executed state-candidate manifest (not one row per continuation).
    manifest_rows = []
    for stage in (1, 2):
        screen = read_csv(f"screening_stage{stage}.csv")
        promoted_keys = {(r["state_id"], r["candidate_id"]) for r in read_csv(f"q64_stage{stage}.csv")}
        for r in screen:
            manifest_rows.append(dict(
                state_id=r["state_id"], stage=stage, candidate_id=r["candidate_id"],
                candidate_kind=r["candidate_kind"], eta1=r["eta1"], eta2=r["eta2"], eta3=r["eta3"],
                screen_trials=8, promoted_to_Q64=(r["state_id"], r["candidate_id"]) in promoted_keys,
                exact_trials=64 if (r["state_id"], r["candidate_id"]) in promoted_keys else 8,
            ))
    write_csv("continuous_search_manifest.csv", manifest_rows,
              ["state_id", "stage", "candidate_id", "candidate_kind", "eta1", "eta2", "eta3", "screen_trials", "promoted_to_Q64", "exact_trials"])

    def summary(prefix):
        return {
            "states": len(disc),
            "B63_states": sum(truth(r[f"{prefix}_B63"]) for r in disc),
            "coverage": float(np.mean([truth(r[f"{prefix}_B63"]) for r in disc])),
            "mean_Q64": float(np.mean([fnum(r[f"{prefix}_Q64"]) for r in disc])),
            "success": sum(int(r[f"{prefix}_success"]) for r in disc),
            "trials": 64 * len(disc),
            "deadlock": sum(int(r[f"{prefix}_deadlock"]) for r in disc),
            "timeout": sum(int(r[f"{prefix}_timeout"]) for r in disc),
            "collision": sum(int(r[f"{prefix}_collision"]) for r in disc),
            "J_def": float(np.mean([fnum(r[f"{prefix}_J_def"]) for r in disc])),
            "episode_length": float(np.mean([fnum(r[f"{prefix}_episode_length"]) for r in disc])),
        }

    summaries = {k: summary(k) for k in ("safety", "fixed", "selector", "oracle")}
    continuous_coverage = float(np.mean([r["continuous_oracle_B63"] for r in comparison]))
    continuous_count = sum(r["continuous_oracle_B63"] for r in comparison)
    gap = continuous_coverage - summaries["oracle"]["coverage"]
    unresolved_count = sum(r["continuous_oracle_status"] == "CONTINUOUS_ORACLE_UNRESOLVED" for r in comparison)

    # Per-mode prevalence and hard-sparsity signature.
    per_mode = []
    for mode_id in range(12):
        v = [r for r in modes if int(r["mode_id"]) == mode_id]
        per_mode.append({"mode_id": mode_id, "B63_states": sum(truth(r["B63"]) for r in v),
                         "prevalence": float(np.mean([truth(r["B63"]) for r in v])),
                         "mean_Q64": float(np.mean([fnum(r["Q64"]) for r in v]))})
    write_csv("hard_mode_prevalence.csv", per_mode, ["mode_id", "B63_states", "prevalence", "mean_Q64"])

    prior = json.load(open(PRIOR / "final_decision.json"))
    rows = [
        {"cohort": "Toy_TEST", **prior["toy_vs_db"]["Toy"], "continuous_oracle_B63_fraction": ""},
        {"cohort": "ordinary_DB_TEST", **prior["toy_vs_db"]["DoubleBottleneck"], "continuous_oracle_B63_fraction": ""},
        {"cohort": "hard_DB", "best_single_B63_fraction": max(r["prevalence"] for r in per_mode),
         "union_oracle_B63_fraction": summaries["oracle"]["coverage"],
         "mean_feasible_modes": float(np.mean([int(r["feasible_modes"]) for r in disc])),
         "median_feasible_modes": float(np.median([int(r["feasible_modes"]) for r in disc])),
         "selector_B63_fraction": summaries["selector"]["coverage"],
         "selector_mean_regret": float(np.mean([fnum(r["selector_regret"]) for r in disc])),
         "continuous_oracle_B63_fraction": continuous_coverage},
    ]
    all_fields = []
    for r in rows:
        for k in r:
            if k not in all_fields:
                all_fields.append(k)
    write_csv("toy_vs_db_vs_harddb.csv", rows, all_fields)

    multiple_necessity = len(necessity) >= 2
    if gap >= 0.15 - 1e-12 and continuous_coverage >= 0.85 - 1e-12:
        classification = "CONTINUOUS_GENERATOR_STRONGLY_NEEDED"
    elif gap >= 0.05 - 1e-12 or multiple_necessity:
        classification = "CONTINUOUS_GENERATOR_PROMISING"
    elif unresolved_count:
        classification = "HARD_DB_UNDERRESOLVED"
    elif gap <= 0.05 + 1e-12 and summaries["oracle"]["coverage"] >= 0.90 - 1e-12:
        classification = "FIXED_MODE_CODEBOOK_SUFFICIENT"
    else:
        classification = "HARD_DB_UNDERRESOLVED"
    ready = classification in {"CONTINUOUS_GENERATOR_STRONGLY_NEEDED", "CONTINUOUS_GENERATOR_PROMISING"}

    hard = rows[-1]
    ordinary = rows[1]
    hard_sparser = (hard["best_single_B63_fraction"] < ordinary["best_single_B63_fraction"] and
                    hard["mean_feasible_modes"] < ordinary["mean_feasible_modes"])
    distances = [r["nearest_transformed_mode_distance"] for r in necessity]
    decision = {
        "classification": classification,
        "READY_TO_TRAIN_CONTINUOUS_GENERATOR": "YES" if ready else "NO",
        "hard_states": len(states),
        "safety": summaries["safety"],
        "fixed": summaries["fixed"],
        "selector": summaries["selector"],
        "transformed_codebook_oracle": summaries["oracle"],
        "continuous_oracle": {"B63_states": continuous_count, "states": len(states),
                              "coverage": continuous_coverage, "confirmed_lower_bound": bool(unresolved_count),
                              "unresolved_states": unresolved_count},
        "coverage_gap": gap,
        "generator_necessity_states": len(necessity),
        "necessity_nearest_mode_distance": {
            "values": distances,
            "min": min(distances) if distances else None,
            "median": float(np.median(distances)) if distances else None,
            "mean": float(np.mean(distances)) if distances else None,
            "max": max(distances) if distances else None,
        },
        "hard_DB_sparser_on_best_single_and_mean_feasible_modes": hard_sparser,
        "hard_DB_best_single_prevalence": hard["best_single_B63_fraction"],
        "hard_DB_mean_feasible_modes": hard["mean_feasible_modes"],
        "ordinary_DB_best_single_prevalence": ordinary["best_single_B63_fraction"],
        "ordinary_DB_mean_feasible_modes": ordinary["mean_feasible_modes"],
        "excluded_cancelled_records": 1128,
        "excluded_batch_manifest": "excluded_cancelled_batch_1150/EXCLUSION_MANIFEST.json",
    }
    dump("final_decision.json", decision)

    # Runtime accounting from complete replacement batch only.
    runtime_files = sorted((H / "raw").glob("*_runtime.json"))
    runtime = [json.load(open(p)) for p in runtime_files]
    runtime_stats = {
        "complete_batch_runtime_files": len(runtime_files),
        "new_continuations": sum(int(r.get("new_continuations", 0)) for r in runtime),
        "physical_steps": sum(int(r.get("physical_steps", 0)) for r in runtime),
        "sum_shard_wall_seconds": sum(float(r.get("wall_seconds", 0)) for r in runtime),
        "excluded_cancelled_records": 1128,
    }
    dump("runtime_statistics.json", runtime_stats)

    report = f"""# OrthoFlow3 DB continuous-generator necessity

Classification: **{classification}**

The outcome-blind hard cohort contains {len(states)} source-isolated Double-Bottleneck true-t0 states. The prior cancelled batch of 1,128 rollout records was physically isolated and never used.

| Method | B63 states | coverage | mean Q64 | success/trials | deadlock | timeout | collision | J_def | episode length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Safety | {summaries['safety']['B63_states']}/{len(states)} | {summaries['safety']['coverage']:.3f} | {summaries['safety']['mean_Q64']:.3f} | {summaries['safety']['success']}/{summaries['safety']['trials']} | {summaries['safety']['deadlock']} | {summaries['safety']['timeout']} | {summaries['safety']['collision']} | {summaries['safety']['J_def']:.4f} | {summaries['safety']['episode_length']:.1f} |
| Fixed transformed mode | {summaries['fixed']['B63_states']}/{len(states)} | {summaries['fixed']['coverage']:.3f} | {summaries['fixed']['mean_Q64']:.3f} | {summaries['fixed']['success']}/{summaries['fixed']['trials']} | {summaries['fixed']['deadlock']} | {summaries['fixed']['timeout']} | {summaries['fixed']['collision']} | {summaries['fixed']['J_def']:.4f} | {summaries['fixed']['episode_length']:.1f} |
| Frozen selector | {summaries['selector']['B63_states']}/{len(states)} | {summaries['selector']['coverage']:.3f} | {summaries['selector']['mean_Q64']:.3f} | {summaries['selector']['success']}/{summaries['selector']['trials']} | {summaries['selector']['deadlock']} | {summaries['selector']['timeout']} | {summaries['selector']['collision']} | {summaries['selector']['J_def']:.4f} | {summaries['selector']['episode_length']:.1f} |
| 12-mode oracle | {summaries['oracle']['B63_states']}/{len(states)} | {summaries['oracle']['coverage']:.3f} | {summaries['oracle']['mean_Q64']:.3f} | {summaries['oracle']['success']}/{summaries['oracle']['trials']} | {summaries['oracle']['deadlock']} | {summaries['oracle']['timeout']} | {summaries['oracle']['collision']} | {summaries['oracle']['J_def']:.4f} | {summaries['oracle']['episode_length']:.1f} |
| Continuous oracle | {continuous_count}/{len(states)} | {continuous_coverage:.3f} | — | — | — | — | — | — | — |

Continuous-minus-codebook coverage gap: **{100*gap:.1f} percentage points**. Generator-necessity states: **{len(necessity)}**. Continuous-search unresolved states: **{unresolved_count}**.

Nearest transformed-mode distances for generator-necessity eta: {distances if distances else 'none'}.

Hard DB best-single B63 prevalence is {hard['best_single_B63_fraction']:.3f}, versus {ordinary['best_single_B63_fraction']:.3f} on ordinary DB. Mean feasible modes/state is {hard['mean_feasible_modes']:.2f}, versus {ordinary['mean_feasible_modes']:.2f}. Hard DB is {'sparser on both measures' if hard_sparser else 'not sparser on both preregistered measures'}.

`READY_TO_TRAIN_CONTINUOUS_GENERATOR = {'YES' if ready else 'NO'}`

No generator was trained.
"""
    (H / "final_report.md").write_text(report)

    artifacts = [
        "hard_state_rule.md", "hard_state_manifest.json", "discrete_baselines.csv",
        "continuous_search_manifest.csv", "continuous_search_results.csv",
        "generator_necessity_states.csv", "continuous_vs_codebook.csv",
        "toy_vs_db_vs_harddb.csv", "final_decision.json", "final_report.md",
        "runtime_statistics.json", "excluded_cancelled_batch_1150/EXCLUSION_MANIFEST.json",
    ]
    dump("manifest.json", {"artifacts": {x: hashlib.sha256((H / x).read_bytes()).hexdigest() for x in artifacts},
                           "frozen_candidate_cloud_sha256": hashlib.sha256((H / "continuous_candidate_cloud.csv").read_bytes()).hexdigest(),
                           "states": len(states), "classification": classification})
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
