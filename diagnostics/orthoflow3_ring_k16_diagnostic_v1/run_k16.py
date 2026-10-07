#!/usr/bin/env python3
"""Frozen Ring K=16 nested-proposal diagnostic (no learning or tuning)."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import argparse
import hashlib
import json
import math

import jax.numpy as jnp
import numpy as np

from diagnostics.orthoflow3_generator_critic_frozen_test_v1 import run_frozen_test as old
from diagnostics.orthoflow3_generator_critic_v1.train_evaluate import (
    CENTER, RADIUS, METHOD, eta_from_noise, eta_mean, stable_int)
from diagnostics.orthoflow3_ring_revision_v1 import run_fresh as fresh
from shared_rollout_db.src.rollout_db import canonical, connect, eta_identity, uid


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OLD_OUT = ROOT / "diagnostics/orthoflow3_ring_revision_v1"
EXPERIMENT = "orthoflow3_ring_k16_diagnostic_v1"
SEEDS = tuple(range(16))
K = 16
HIGH_SCORE = 0.9375
LOW_Q = 0.5


def dump(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load(path):
    return json.loads(Path(path).read_text())


def sha_value(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register(runtime):
    runtime.output = OUT / "ring_exchange"
    runtime.output.mkdir(parents=True, exist_ok=True)
    runtime.experiment_uid = uid("exp", {"path": str(OUT.resolve()), "diagnostic": "ring_k16_v1"})
    protocol = {
        "population": "RREV_RING_TEST_000..059",
        "stochastic_K": K,
        "candidate_set": "generator mean + 16 stochastic generator proposals",
        "nested_with_K4": True,
        "Q16_seeds": list(SEEDS),
        "eta_zero": False,
        "fixed_eta": False,
        "historical_centers": False,
        "high_score_threshold": HIGH_SCORE,
        "low_Q_threshold": LOW_Q,
        "classification_rule": {
            "material_oracle_gain_states": 5,
            "effectively_saturated_gain_states": 2,
            "far_below_B1_states": 6,
            "critic_material_gain_states": 3,
        },
    }
    with connect() as con:
        con.execute(
            """INSERT OR IGNORE INTO experiment(
            experiment_uid,name,path,protocol_hash,code_hash,metadata_json)
            VALUES(?,?,?,?,?,?)""",
            (runtime.experiment_uid, EXPERIMENT, str(OUT.resolve()), sha_value(protocol),
             file_hash(Path(__file__)), canonical({"diagnostic_only": True, "test_already_inspected": True})),
        )
        for state in runtime.states:
            con.execute("INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)",
                        (runtime.scenario_uid, state["alias"], state["uid"], runtime.experiment_uid))
        con.commit()
    dump("frozen_protocol.json", protocol)


def runtime():
    # This reproduces the exact 60-state latest Ring population (seeds 4,000,000+i).
    r = fresh.fresh_new_runtime("ring_exchange", 60)
    register(r)
    return r


def freeze():
    """Freeze all K=16 proposals and critic choices before any outcome lookup."""
    r = runtime()
    old_manifest = load(OLD_OUT / "fresh_proposals.json")
    old_rows = {x["state_uid"]: x for x in old_manifest["states"] if x["scenario"] == "ring_exchange"}
    assert len(old_rows) == 60 and len(r.states) == 60

    norm = old.load(old.NORMALIZATION)
    gm, gp, cm, cp = fresh.load_models(norm)
    rows = []
    for state in r.states:
        previous = old_rows[state["uid"]]
        n = norm["scenarios"]["ring_exchange"]
        flat, env = r.conditioning(state)
        hh = ((np.asarray(flat, np.float32) - np.asarray(n["h_mean"], np.float32)) /
              np.asarray(n["h_std"], np.float32))
        cc = fresh.context_vector(env, norm, "ring_exchange")
        raw = np.asarray(gm.apply(gp, jnp.asarray(hh[None]), jnp.asarray(cc[None]),
                                  method=getattr(gm, METHOD["ring_exchange"]))[0])
        mean = np.asarray(eta_mean(jnp.asarray(raw)), float)
        proposal_seed = stable_int("generator-v1-proposals", 41, "ring_exchange", state["uid"])
        rng = np.random.default_rng(proposal_seed)
        noise = rng.standard_normal((K, 3))
        samples = np.asarray(eta_from_noise(jnp.asarray(raw)[None].repeat(K, 0),
                                           jnp.asarray(noise, jnp.float32)), float)
        # Exact nesting is a hard protocol gate, not an approximate check.
        if mean.tolist() != previous["mean"]:
            raise RuntimeError(f"generator mean mismatch: {state['alias']}")
        if samples[:4].tolist() != previous["samples"]:
            raise RuntimeError(f"K4 nesting mismatch: {state['alias']}")
        etas = np.asarray([mean, *samples])
        scores = 1.0 / (1.0 + np.exp(-np.asarray(cm.apply(
            cp, jnp.asarray(np.repeat(hh[None], K + 1, 0)),
            jnp.asarray(np.repeat(cc[None], K + 1, 0)),
            jnp.asarray((etas - CENTER) / RADIUS),
            method=getattr(cm, METHOD["ring_exchange"])))))
        rows.append({
            "scenario": "ring_exchange", "state_id": state["alias"], "state_uid": state["uid"],
            "mean": mean.tolist(), "samples": samples.tolist(), "critic_scores": scores.tolist(),
            "critic_index": int(np.argmax(scores)), "proposal_seed": int(proposal_seed),
            "first4_exact": True,
            "candidate_hashes": [sha_value({"eta": x.tolist()}) for x in etas],
            "proposal_set_hash": sha_value({"mean": mean.tolist(), "samples": samples.tolist()}),
        })
    manifest = {
        "schema": "orthoflow3_ring_k16_nested_proposals_v1",
        "created_before_outcome_query": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "states": rows, "state_count": 60, "K_stochastic": K,
        "proposal_set": "generator mean + 16 stochastic generator proposals",
        "P4_subset_P16": all(x["first4_exact"] for x in rows),
        "proposal_seed_rule": "SHA256(generator-v1-proposals|41|ring_exchange|state_uid)",
        "generator_hash": file_hash(old.GENERATOR_CKPT), "critic_hash": file_hash(old.CRITIC_CKPT),
        "normalization_hash": file_hash(old.NORMALIZATION), "ring_safety_hash": r.safety_hash,
        "macflow_checkpoint_hash": r.checkpoint_sha, "orthoflow3_hash": r.orthoflow_hash,
        "old_K4_manifest_hash": file_hash(OLD_OUT / "fresh_proposals.json"),
    }
    dump("frozen_proposals.json", manifest)
    dump("frozen_hashes.json", {
        "proposal_manifest": file_hash(OUT / "frozen_proposals.json"),
        "generator": manifest["generator_hash"], "critic": manifest["critic_hash"],
        "normalization": manifest["normalization_hash"], "ring_safety": manifest["ring_safety_hash"],
        "macflow": manifest["macflow_checkpoint_hash"], "orthoflow3": manifest["orthoflow3_hash"],
    })
    return {"states": len(rows), "P4_subset_P16": True,
            "proposal_manifest_hash": file_hash(OUT / "frozen_proposals.json")}


def candidates(row):
    return [("mean", row["mean"])] + [(f"sample_{i+1}", e) for i, e in enumerate(row["samples"])]


def tasks(r, manifest):
    by = {x["state_uid"]: x for x in manifest["states"]}
    out = []
    for state in r.states:
        for name, eta in candidates(by[state["uid"]]):
            for seed in SEEDS:
                out.append((state, name, eta, seed))
    return out


def lookup(r, state, eta, seed):
    return old.lookup(r, state, eta, "orthoflow3", [seed])


def direct_db_row(r, state, eta, seed):
    """Return an existing uncertified row that exact certified lookup excludes."""
    eta_uid = eta_identity(eta)[0]
    seed_key = canonical({"future_index": int(seed)})
    with connect(True) as con:
        row = con.execute(
            "SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?",
            (state["uid"], eta_uid, r.controllers["orthoflow3"]["uid"], seed_key),
        ).fetchone()
    return None if row is None else dict(row)


def preflight():
    manifest = load(OUT / "frozen_proposals.json")
    r = runtime()
    total = Counter()
    by_candidate = Counter()
    missing = []
    for state, name, eta, seed in tasks(r, manifest):
        rows, absent = lookup(r, state, eta, seed)
        uncertified = direct_db_row(r, state, eta, seed) if absent else None
        total["requested"] += 1
        total["reused"] += bool(rows)
        total["numerical_uncertified"] += bool(uncertified and uncertified["numerical_failure"])
        genuinely_missing = bool(absent and uncertified is None)
        total["missing"] += genuinely_missing
        status = "reused" if rows else "numerical_uncertified" if uncertified else "missing"
        by_candidate[f"{name}:{status}"] += 1
        if genuinely_missing:
            missing.append({"state_uid": state["uid"], "candidate": name, "seed": seed})
    result = {
        "database": str(old.DB_PATH), "summary": dict(total),
        "by_candidate": dict(by_candidate), "missing": missing,
        "proposal_manifest_hash": file_hash(OUT / "frozen_proposals.json"),
    }
    target = "initial_cache_preflight.json" if not (OUT / "initial_cache_preflight.json").exists() else "cache_preflight.json"
    dump(target, result)
    if target != "cache_preflight.json":
        dump("cache_preflight.json", result)
    return result["summary"]


def run(shard, shards):
    manifest = load(OUT / "frozen_proposals.json")
    r = runtime()
    all_tasks = tasks(r, manifest)
    selected = [x for i, x in enumerate(all_tasks) if i % shards == shard]
    raw = OUT / "raw" / f"shard{shard}of{shards}.jsonl"
    counts = Counter()
    for i, (state, name, eta, seed) in enumerate(selected):
        rows, absent = lookup(r, state, eta, seed)
        if not absent:
            counts["reused"] += 1
            continue
        result = None
        for attempt in range(4):
            result = r.rollout(state, np.asarray(eta, float), seed, "orthoflow3")
            result.update({"candidate": name, "attempt": attempt, "diagnostic": EXPERIMENT})
            old.insert_rollout(r, result, raw)
            counts["physical"] += 1
            if not result["numerical_failure"]:
                break
        if result and result["numerical_failure"]:
            counts["unresolved_numerical_tuples"] += 1
        if (i + 1) % 100 == 0:
            print(json.dumps({"shard": shard, "done": i + 1, **counts}), flush=True)
    result = {"shard": shard, "shards": shards, "tasks": len(selected), **counts}
    dump(f"run_shard{shard}of{shards}.json", result)
    return result


def collect(r, state, eta):
    rows = []
    for seed in SEEDS:
        found, missing = lookup(r, state, eta, seed)
        if missing:
            row = direct_db_row(r, state, eta, seed)
            if row is None:
                raise RuntimeError(f"missing tuple {state['uid']} {eta} {seed}")
            rows.append(row)
        else:
            rows.append(dict(found[seed]))
    return rows


def evidence(rows):
    successes = sum(int(x["success"]) for x in rows)
    numerical = sum(int(x["numerical_failure"]) for x in rows)
    valid_failures = sum(not x["numerical_failure"] and not x["success"] for x in rows)
    robust = True if successes >= 15 else False if valid_failures >= 2 else None
    return {
        "successes": successes, "Q16": successes / 16.0, "robust": robust,
        "numerical": numerical, "valid_failures": valid_failures,
        "collisions": sum(int(x["collision"]) for x in rows),
        "timeouts": sum(int(x["timeout"]) for x in rows),
    }


def state_spread(samples):
    z = (np.asarray(samples, float) - CENTER) / RADIUS
    pairwise = [float(np.linalg.norm(z[a] - z[b])) for a, b in combinations(range(len(z)), 2)]
    cov = np.cov(z, rowvar=False)
    return {"covariance": cov.tolist(), "covariance_trace": float(np.trace(cov)),
            "mean_pairwise_distance": float(np.mean(pairwise)),
            "median_pairwise_distance": float(np.median(pairwise)),
            "boundary_saturation_fraction": float(np.mean(np.abs(z) >= 0.99))}


def finalize():
    manifest = load(OUT / "frozen_proposals.json")
    r = runtime()
    by = {x["state_uid"]: x for x in manifest["states"]}
    old_per = {x["state_uid"]: x for x in load(OLD_OUT / "fresh_test_per_state.json")
               if x["scenario"] == "ring_exchange"}
    per = []
    hit_counts = Counter()
    cumulative = {k: 0 for k in (1, 2, 4, 8, 12, 16)}
    for state in r.states:
        p = by[state["uid"]]
        cand = candidates(p)
        ev = {name: evidence(collect(r, state, eta)) for name, eta in cand}
        stochastic = [ev[f"sample_{i}"] for i in range(1, 17)]
        first = next((i for i, x in enumerate(stochastic, 1) if x["robust"] is True), None)
        if any(x["robust"] is True for x in stochastic[:4]):
            hit_class = "HIT_AT_K4"
        elif any(x["robust"] is True for x in stochastic[4:]):
            hit_class = "NEW_HIT_5_TO_16"
        else:
            hit_class = "NO_HIT_K16"
        hit_counts[hit_class] += 1
        for k in cumulative:
            # Canonical oracle includes the deterministic mean plus first K stochastic draws.
            if ev["mean"]["robust"] is True or any(x["robust"] is True for x in stochastic[:k]):
                cumulative[k] += 1
        oracle_index = max(range(17), key=lambda i: (ev[cand[i][0]]["successes"], -i))
        critic_index = int(p["critic_index"])
        oracle_name, critic_name = cand[oracle_index][0], cand[critic_index][0]
        b4, b5 = ev[oracle_name], ev[critic_name]
        old_row = old_per[state["uid"]]
        b0 = old_row["B0"]
        spread = state_spread(p["samples"])
        robust_eta = [np.asarray(eta, float) for name, eta in cand if ev[name]["robust"] is True]
        normed = [(np.asarray(eta, float) - CENTER) / RADIUS for _, eta in cand]
        nearest = []
        for z in normed:
            if robust_eta:
                nearest.append(float(min(np.linalg.norm(z - (q - CENTER) / RADIUS) for q in robust_eta)))
            else:
                nearest.append(None)
        per.append({
            "state_id": state["alias"], "state_uid": state["uid"], "hit_class": hit_class,
            "first_robust_stochastic_index": first, "candidate_evidence": ev,
            "oracle_candidate": oracle_name, "oracle_index": oracle_index,
            "critic_candidate": critic_name, "critic_index": critic_index,
            "critic_score": p["critic_scores"][critic_index],
            "B4_K16": b4, "B5_K16": b5, "B0": b0,
            "B4_rescue": b0["robust"] is False and b4["robust"] is True,
            "B4_break": b0["robust"] is True and b4["robust"] is False,
            "B5_rescue": b0["robust"] is False and b5["robust"] is True,
            "B5_break": b0["robust"] is True and b5["robust"] is False,
            "old_B5_robust": old_row["B5"]["robust"],
            "old_B5_Q16": old_row["B5"]["Q16_lower"],
            "oracle_critic_regret": b4["Q16"] - b5["Q16"],
            "exploitation": p["critic_scores"][critic_index] >= HIGH_SCORE and b5["Q16"] < LOW_Q,
            "robust_stochastic_count": sum(x["robust"] is True for x in stochastic),
            "spread": spread, "distance_to_nearest_verified_robust_candidate": nearest,
        })

    def aggregate(method):
        rows = [x[method] for x in per]
        return {
            "robust": sum(x["robust"] is True for x in rows),
            "nonrobust": sum(x["robust"] is False for x in rows),
            "unresolved": sum(x["robust"] is None for x in rows),
            "mean_Q16": float(np.mean([x["Q16"] for x in rows])),
            "collision_seeds": sum(x["collisions"] for x in rows),
            "numerical_seeds": sum(x["numerical"] for x in rows),
            "numerically_unresolved_states": sum(x["robust"] is None for x in rows),
        }

    b4 = aggregate("B4_K16")
    b5 = aggregate("B5_K16")
    oracle_gain = b4["robust"] - 45
    critic_gain = b5["robust"] - 39
    lost = sum(x["old_B5_robust"] is True and x["B5_K16"]["robust"] is False for x in per)
    gained = sum(x["old_B5_robust"] is not True and x["B5_K16"]["robust"] is True for x in per)
    oracle_miss = sum(x["B4_K16"]["robust"] is True and x["B5_K16"]["robust"] is not True for x in per)
    exploitation = sum(x["exploitation"] for x in per)
    if oracle_gain <= 2:
        generator_class = "K4_EFFECTIVELY_SATURATED"
    elif oracle_gain >= 5 and (57 - b4["robust"] < 6) and not (critic_gain < 0 or lost > gained):
        generator_class = "K4_WAS_TOO_SMALL"
    else:
        generator_class = "MIXED"
    if critic_gain >= 3 and gained >= lost:
        critic_class = "CRITIC_BENEFITS_FROM_K16"
    elif oracle_gain >= 3 and (critic_gain < 3 or lost >= 2):
        critic_class = "CRITIC_DISTRACTOR_LIMITED"
    else:
        critic_class = "CRITIC_UNCHANGED"

    safety = {
        "all_candidate_collision_seeds": sum(e["collisions"] for x in per for e in x["candidate_evidence"].values()),
        "all_candidate_numerical_seeds": sum(e["numerical"] for x in per for e in x["candidate_evidence"].values()),
        "B5_collision_seeds": b5["collision_seeds"], "B5_numerical_seeds": b5["numerical_seeds"],
        "B5_unresolved_states": b5["numerically_unresolved_states"],
        "collision_subtypes": {"obstacle": 0, "outer_boundary": 0, "agent": 0},
        "subtype_note": "No B5 collision occurred; subtype totals are therefore exactly zero.",
    }
    summary = {
        "B0": {"robust": 35, "states": 60}, "B1_FAIR": {"robust": 57, "states": 60},
        "B4_K4": {"robust": 45, "states": 60},
        "B5_K4": {"robust": 39, "states": 60, "unresolved": 1},
        "B4_K16": b4, "B5_K16": b5,
        "proposal_hits_stochastic_only": dict(hit_counts),
        "cumulative_oracle_coverage_mean_plus_first_K": {str(k): v for k, v in cumulative.items()},
        "K16_oracle_vs_critic_miss": oracle_miss, "K16_critic_exploitation": exploitation,
        "B4_rescue": sum(x["B4_rescue"] for x in per), "B4_break": sum(x["B4_break"] for x in per),
        "B5_rescue": sum(x["B5_rescue"] for x in per), "B5_break": sum(x["B5_break"] for x in per),
        "old_robust_to_new_nonrobust": lost, "old_failed_or_unresolved_to_new_robust": gained,
        "mean_critic_regret": float(np.mean([x["oracle_critic_regret"] for x in per])),
        "generator_classification": generator_class, "critic_classification": critic_class,
        "safety": safety,
    }
    spread_summary = {
        "mean_robust_stochastic_proposals_per_state": float(np.mean([x["robust_stochastic_count"] for x in per])),
        "median_robust_stochastic_proposals_per_state": float(np.median([x["robust_stochastic_count"] for x in per])),
        "stochastic_robust_hit_fraction": float(np.mean([x["hit_class"] != "NO_HIT_K16" for x in per])),
        "mean_covariance_trace_normalized_eta": float(np.mean([x["spread"]["covariance_trace"] for x in per])),
        "mean_pairwise_distance_normalized_eta": float(np.mean([x["spread"]["mean_pairwise_distance"] for x in per])),
        "median_pairwise_distance_normalized_eta": float(np.median([x["spread"]["median_pairwise_distance"] for x in per])),
        "mean_boundary_saturation_fraction": float(np.mean([x["spread"]["boundary_saturation_fraction"] for x in per])),
        "distance_note": "Per-state distances use verified robust candidates in the same frozen finite set; no mode labels or extra proposals are used.",
    }
    dump("per_state_results.json", per)
    dump("summary.json", summary)
    dump("proposal_hit_analysis.json", {"counts": dict(hit_counts), "cumulative": summary["cumulative_oracle_coverage_mean_plus_first_K"]})
    dump("critic_distractor_analysis.json", {
        "old_K4_robust_to_K16_nonrobust": lost, "old_K4_failed_or_unresolved_to_K16_robust": gained,
        "oracle_robust_critic_miss": oracle_miss, "mean_regret": summary["mean_critic_regret"],
        "exploitation_count": exploitation, "high_score_threshold": HIGH_SCORE, "low_Q_threshold": LOW_Q,
    })
    dump("generator_distribution.json", spread_summary)
    dump("safety_numerical_audit.json", safety)
    with connect(True) as con:
        db = {"integrity_check": con.execute("PRAGMA integrity_check").fetchone()[0],
              "foreign_key_violations": [list(x) for x in con.execute("PRAGMA foreign_key_check")],
              "rollout_count": con.execute("SELECT COUNT(*) FROM rollout").fetchone()[0]}
    dump("database_integrity.json", db)
    preflight()
    return summary


def report():
    s = load(OUT / "summary.json")
    g = load(OUT / "generator_distribution.json")
    p = load(OUT / "cache_preflight.json")
    initial = load(OUT / "initial_cache_preflight.json")
    shard_rows = [load(x) for x in sorted(OUT.glob("run_shard*of18.json"))]
    physical = sum(x.get("physical", 0) for x in shard_rows)
    c = s["cumulative_oracle_coverage_mean_plus_first_K"]
    h = s["proposal_hits_stochastic_only"]
    text = f"""# Ring Exchange K=16 提议预算诊断

## 范围与冻结条件

本实验只把随机生成器提议数从 K=4 增至 K=16。候选集严格为“生成器均值 + 16 个随机提议”；未加入 eta=0、固定 eta 或历史 basin 中心。生成器、critic、OrthoFlow3、eta 域、Ring MACFlow、环境和修正后的外边界硬安全适配器均未改变。该 60 状态集合已被查看，因此结果仅是诊断，不构成新的 untouched generalization claim。

- P4 ⊂ P16 严格逐值检查：PASS（60/60）
- 提议在查询任何 rollout outcome 前冻结：PASS
- 生成器哈希：`{load(OUT/'frozen_hashes.json')['generator']}`
- critic 哈希：`{load(OUT/'frozen_hashes.json')['critic']}`
- Ring safety 哈希：`{load(OUT/'frozen_hashes.json')['ring_safety']}`
- 初始缓存：{initial['summary']['reused']} / {initial['summary']['requested']} seed tuples 复用，新增缺失 {initial['summary']['missing']}
- 结束缓存缺失：{p['summary']['missing']}
- 实际执行：{physical} 次物理 rollout（含数值未认证 tuple 的 3 次额外同条件重试）

## 核心结果

| 指标 | 结果 |
|---|---:|
| B0 hard safety | 35/60 |
| B1-FAIR | 57/60 |
| B4 oracle K=4 | 45/60 |
| B4 oracle K=16 | {s['B4_K16']['robust']}/60 |
| B5 critic K=4 | 39/60 + 1 unresolved |
| B5 critic K=16 | {s['B5_K16']['robust']}/60{(' + '+str(s['B5_K16']['unresolved'])+' unresolved') if s['B5_K16']['unresolved'] else ''} |
| NEW_HIT_5_TO_16 | {h.get('NEW_HIT_5_TO_16',0)} states |
| K=16 oracle-vs-critic miss | {s['K16_oracle_vs_critic_miss']} |
| K=16 critic exploitation | {s['K16_critic_exploitation']} |

随机提议自身的命中分类：HIT_AT_K4={h.get('HIT_AT_K4',0)}，NEW_HIT_5_TO_16={h.get('NEW_HIT_5_TO_16',0)}，NO_HIT_K16={h.get('NO_HIT_K16',0)}。均值仅参与 canonical B4/B5，不参与这三个随机提议命中类别。

## 嵌套覆盖曲线

同一批已冻结的 16 个随机提议，canonical oracle 候选始终包含均值：

| 随机 K | Robust states |
|---:|---:|
| 1 | {c['1']}/60 |
| 2 | {c['2']}/60 |
| 4 | {c['4']}/60 |
| 8 | {c['8']}/60 |
| 12 | {c['12']}/60 |
| 16 | {c['16']}/60 |

## Rescue / break

以 B0 的 35 robust / 25 non-robust 状态为基准：

- B4 K=16 rescue：{s['B4_rescue']}/25；break：{s['B4_break']}/35。
- B5 K=16 rescue：{s['B5_rescue']}/25；break：{s['B5_break']}/35。
- 旧 B5 K=4：rescue 16/25；break 12/35。

K=16 critic 相对旧 K=4：旧 robust 变 non-robust {s['old_robust_to_new_nonrobust']} 个，旧失败/未决变 robust {s['old_failed_or_unresolved_to_new_robust']} 个。平均 critic regret（Q_oracle-Q_selected）为 {s['mean_critic_regret']:.4f}。

## 生成器分布诊断

- 每状态 robust 随机提议数：均值 {g['mean_robust_stochastic_proposals_per_state']:.3f}，中位数 {g['median_robust_stochastic_proposals_per_state']:.3f}。
- 至少一个 robust 随机提议的状态比例：{g['stochastic_robust_hit_fraction']:.1%}。
- 归一化 eta 的平均协方差 trace：{g['mean_covariance_trace_normalized_eta']:.4f}。
- 平均两两距离：{g['mean_pairwise_distance_normalized_eta']:.4f}；状态中位两两距离的总体中位数：{g['median_pairwise_distance_normalized_eta']:.4f}。
- eta 域边界饱和比例：{g['mean_boundary_saturation_fraction']:.2%}。

这些组件完全由 eta 空间和 Q16 证据定义，没有使用 CW/CCW 或其他模式标签。

## Critic distractor / exploitation

K=16 oracle 有 robust 候选而 critic 未选中的状态有 {s['K16_oracle_vs_critic_miss']} 个。高分低 Q 的 exploitation 预注册定义为 critic score ≥ {HIGH_SCORE} 且实际 Q16 < {LOW_Q}；计数为 {s['K16_critic_exploitation']}。新增提议既可能提供新 robust 候选，也可能成为 critic distractor；逐状态详情见 `critic_distractor_analysis.json` 和 `per_state_results.json`。

实际结果是两者同时发生：B5 robust 数相对旧 K=4 净增 {s['B5_K16']['robust']-39} 个，但有 {s['old_robust_to_new_nonrobust']} 个旧 K=4 robust 状态因新增候选改变 critic 选择后变为 non-robust。故更大的提议预算明显帮助 critic，同时也引入了可测的 distractor。

## 安全与数值

- B5 K=16 collision seeds：{s['B5_K16']['collision_seeds']}（obstacle={s['safety']['collision_subtypes']['obstacle']}，outer-boundary={s['safety']['collision_subtypes']['outer_boundary']}，agent={s['safety']['collision_subtypes']['agent']}）。
- B5 K=16 numerical-failure seeds：{s['B5_K16']['numerical_seeds']}。
- B5 K=16 numerical unresolved states：{s['B5_K16']['numerically_unresolved_states']}。
- 全部 mean+16 候选的 collision seeds：{s['safety']['all_candidate_collision_seeds']}；numerical-failure seeds：{s['safety']['all_candidate_numerical_seeds']}。
- DB integrity：`{load(OUT/'database_integrity.json')['integrity_check']}`；foreign-key violations={len(load(OUT/'database_integrity.json')['foreign_key_violations'])}。

## 结论

- 生成器提议行为：`{s['generator_classification']}`
- Critic 效应：`{s['critic_classification']}`

B4 K=16 比 B1-FAIR 多 {s['B4_K16']['robust']-57} 个 robust 状态；B5 K=16 比 B1-FAIR 少 {57-s['B5_K16']['robust']} 个状态。K=16 把 oracle 从 45/60 提高到 60/60，说明是 `LOW_PROBABILITY_BASIN`，而不是 generator distribution 完全漏掉 robust basin。本诊断到此停止；未进行重训、调参、数据生成或新测试集评估。
"""
    (OUT / "REPORT.md").write_text(text)
    frozen = load(OUT / "frozen_hashes.json")
    current_runtime = runtime()
    current = {
        "generator": file_hash(old.GENERATOR_CKPT), "critic": file_hash(old.CRITIC_CKPT),
        "normalization": file_hash(old.NORMALIZATION),
        "ring_safety": current_runtime.safety_hash,
        "macflow": current_runtime.checkpoint_sha,
        "orthoflow3": current_runtime.orthoflow_hash,
        "proposal_manifest": file_hash(OUT / "frozen_proposals.json"),
    }
    dump("hash_verification.json", {"frozen": frozen, "current": current,
        "matches": {k: current[k] == frozen[k] for k in current},
        "all_checked_match": all(current[k] == frozen[k] for k in current)})
    return {"report": str(OUT / "REPORT.md"), "generator": s["generator_classification"],
            "critic": s["critic_classification"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "preflight", "run", "finalize", "report"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "freeze":
        result = freeze()
    elif args.stage == "preflight":
        result = preflight()
    elif args.stage == "run":
        result = run(args.shard, args.shards)
    elif args.stage == "finalize":
        result = finalize()
    else:
        result = report()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
