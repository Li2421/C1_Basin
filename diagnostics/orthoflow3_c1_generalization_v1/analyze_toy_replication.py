"""Analyze frozen Toy replication batches without changing any selections.

Only complete 16-seed state/candidate cells enter the report. The script may be
rerun as append-only journals arrive; partial cells are counted but not imputed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent / "toy_replication"
KINDS = [f"sample_{i}" for i in range(16)]


def exact_sign_p(wins: int, losses: int) -> float:
    from math import comb

    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, j) for j in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=HERE / "replication_analysis.json")
    args = ap.parse_args()
    frozen = json.loads((HERE / "frozen_proposals.json").read_text())
    states = {int(s["episode_index"]): s for s in frozen["states"]}
    cells: dict[tuple[int, str], dict[int, dict]] = defaultdict(dict)
    duplicated = 0
    for path in sorted((HERE / "raw").glob("shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (int(r["episode_index"]), str(r["kind"]))
            seed = int(r["future_index"])
            if seed in cells[key]:
                duplicated += 1
                if cells[key][seed]["outcome"] != r["outcome"]:
                    raise RuntimeError(f"Conflicting local record {key} seed={seed}")
            else:
                cells[key][seed] = r
    q: dict[tuple[int, str], float] = {}
    anomaly = Counter()
    for key, seed_records in cells.items():
        if len(seed_records) != 16:
            continue
        if set(seed_records) != set(range(16)):
            raise RuntimeError(f"Nonstandard seed set in {key}")
        for r in seed_records.values():
            anomaly[r["outcome"]] += 1
            if not r["scientific_outcome_valid"]:
                anomaly["invalid"] += 1
        q[key] = sum(r["success"] for r in seed_records.values()) / 16.0

    rows = []
    for ep in sorted(states):
        if not all((ep, kind) in q for kind in KINDS):
            continue
        s = states[ep]
        vals = [q[(ep, kind)] for kind in KINDS]
        picks = {
            "critic": int(s["critic_choice"]),
            "eta_only_kernel": int(s["eta_only_kernel_choice"]),
            "eta_only_mlp": int(s["eta_only_mlp_choice"]),
        }
        row = {
            "episode_index": ep,
            "source_group": s["source_group"],
            "oracle_q16": max(vals),
            "oracle_b15": max(vals) >= 15 / 16,
            "n_b15_proposals": sum(v >= 15 / 16 for v in vals),
        }
        for name, idx in picks.items():
            row[f"{name}_index"] = idx
            row[f"{name}_q16"] = vals[idx]
            row[f"{name}_b15"] = vals[idx] >= 15 / 16
        for name, kind in (("generator_mean", "generator_mean"),
                           ("fixed_common", "fixed_common"),
                           ("safety", "safety"), ("mac_only", "mac_only")):
            row[f"{name}_q16"] = q.get((ep, kind))
            row[f"{name}_b15"] = None if (ep, kind) not in q else q[(ep, kind)] >= 15 / 16
        rows.append(row)

    metrics = {}
    for name in ("oracle", "critic", "eta_only_kernel", "eta_only_mlp",
                 "generator_mean", "fixed_common", "safety", "mac_only"):
        valid = [r for r in rows if r.get(f"{name}_b15") is not None]
        metrics[name] = {
            "n_states": len(valid),
            "b15": sum(r[f"{name}_b15"] for r in valid),
            "mean_q16": float(np.mean([r[f"{name}_q16"] for r in valid])) if valid else None,
        }
        if not valid or name == "oracle":
            continue
        chosen_kind = {
            "critic": "critic_index", "eta_only_kernel": "eta_only_kernel_index",
            "eta_only_mlp": "eta_only_mlp_index",
        }.get(name)
        selected = []
        for row in valid:
            kind = f"sample_{row[chosen_kind]}" if chosen_kind else name
            selected.extend(cells[(row["episode_index"], kind)].values())
        outcomes = Counter(r["outcome"] for r in selected)
        metrics[name]["continuations"] = len(selected)
        metrics[name]["outcomes"] = dict(outcomes)
        jdef = [r["J_def"] for r in selected if r.get("J_def") is not None]
        metrics[name]["mean_J_def"] = float(np.mean(jdef)) if jdef else None
        metrics[name]["mean_episode_steps"] = float(np.mean([r["episode_steps"] for r in selected]))
    comparisons = {}
    rng = np.random.default_rng(20261002)
    for name in ("eta_only_kernel", "eta_only_mlp", "generator_mean", "fixed_common", "safety"):
        valid = [r for r in rows if r.get(f"{name}_b15") is not None]
        wins = sum(r["critic_b15"] and not r[f"{name}_b15"] for r in valid)
        losses = sum(r[f"{name}_b15"] and not r["critic_b15"] for r in valid)
        deltas = np.asarray([int(r["critic_b15"]) - int(r[f"{name}_b15"]) for r in valid])
        q_deltas = np.asarray([r["critic_q16"] - r[f"{name}_q16"] for r in valid])
        if valid:
            sampled = rng.integers(0, len(valid), size=(20000, len(valid)))
            boot = deltas[sampled].mean(axis=1)
            ci = np.quantile(boot, [0.025, 0.975]).tolist()
            q_boot = q_deltas[sampled].mean(axis=1)
            q_ci = np.quantile(q_boot, [0.025, 0.975]).tolist()
        else:
            ci = None
            q_ci = None
        comparisons[f"critic_vs_{name}"] = {
            "paired_states": len(valid), "rescue": wins, "break": losses,
            "net": wins - losses, "b15_rate_delta": float(deltas.mean()) if valid else None,
            "bootstrap_95ci": ci, "exact_sign_p": exact_sign_p(wins, losses),
            "mean_q16_delta": float(q_deltas.mean()) if valid else None,
            "mean_q16_bootstrap_95ci": q_ci,
        }
    result = {
        "cohort_size": len(states),
        "complete_proposal_states": len(rows),
        "complete_cells": len(q),
        "partial_cells": sum(0 < len(v) < 16 for v in cells.values()),
        "local_duplicate_records": duplicated,
        "outcomes_complete_cells": dict(anomaly),
        "metrics": metrics,
        "comparisons": comparisons,
        "per_state": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("cohort_size", "complete_proposal_states", "complete_cells", "partial_cells", "metrics", "comparisons")}, indent=2))


if __name__ == "__main__":
    main()
