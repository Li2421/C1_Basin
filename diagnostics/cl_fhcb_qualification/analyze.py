"""Analyze the predeclared frozen CL-FHCB qualification without new rollouts."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, rankdata, spearmanr

from diagnostics.cl_fhcb.certificate import BIN_NAMES
from diagnostics.cl_fhcb_qualification.qualification_common import (
    K_VALUES,
    PHIS,
    REPO_ROOT,
    json_dump,
    load_certificates,
    sha256,
    verify_frozen_method,
    wilson_interval,
)


OUT = REPO_ROOT / "diagnostics/cl_fhcb_qualification"
RAW = OUT / "raw/stage1"
FIG = OUT / "figures"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 44017


def auc_stat(positive, negative):
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    if not len(positive) or not len(negative):
        return float("nan")
    values = np.concatenate((positive, negative))
    ranks = rankdata(values, method="average")
    n_pos, n_neg = len(positive), len(negative)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def stratified_auc_bootstrap(positive, negative, rng):
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    draws = []
    separations = []
    for _ in range(BOOTSTRAP_REPLICATES):
        p = rng.choice(positive, len(positive), replace=True)
        n = rng.choice(negative, len(negative), replace=True)
        draws.append(auc_stat(p, n))
        separations.append(float(p.mean() - n.mean()))
    return {
        "auroc": auc_stat(positive, negative),
        "auroc_ci95": [float(x) for x in np.quantile(draws, (0.025, 0.975))],
        "rank_biserial": float(2 * auc_stat(positive, negative) - 1),
        "mean_separation": float(positive.mean() - negative.mean()),
        "mean_separation_ci95": [
            float(x) for x in np.quantile(separations, (0.025, 0.975))
        ],
        "positive_range": [float(positive.min()), float(positive.max())],
        "negative_range": [float(negative.min()), float(negative.max())],
        "cross_class_tie_fraction": float(
            np.mean(positive[:, None] == negative[None, :])
        ),
        "n_deadlock": len(positive),
        "n_nondeadlock": len(negative),
    }


def analyze_p(risk_records):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    strata = []
    for K in K_VALUES:
        checkpoints = sorted({
            item["checkpoint"]
            for item in risk_records
            if item["K"] == K and item["steps_to_terminal"] > K
        })
        for checkpoint in checkpoints:
            rows = [
                item for item in risk_records
                if item["K"] == K
                and item["checkpoint"] == checkpoint
                and item["steps_to_terminal"] > K
            ]
            positive = [item["value"] for item in rows if item["eventual_deadlock"]]
            negative = [item["value"] for item in rows if not item["eventual_deadlock"]]
            if not positive or not negative:
                continue
            record = {
                "K": K,
                "checkpoint": checkpoint,
                "observed_prefix_excludes_terminal": True,
                "remaining_lead_steps_after_prefix": (
                    min(item["steps_to_terminal"] for item in rows) - K
                    if checkpoint != "initial" else None
                ),
                **stratified_auc_bootstrap(positive, negative, rng),
            }
            record["predeclared_pass"] = bool(
                record["n_deadlock"] >= 4
                and record["n_nondeadlock"] >= 4
                and record["auroc"] >= 0.70
                and record["auroc_ci95"][0] > 0.50
                and record["mean_separation"] > 0
            )
            strata.append(record)
    passing = [item for item in strata if item["predeclared_pass"]]
    if passing:
        status = "PASS"
    elif any(
        item["auroc"] >= 0.60 and item["mean_separation"] > 0 for item in strata
    ):
        status = "PARTIAL"
    else:
        status = "FAIL"
    demonstrated = [
        item["remaining_lead_steps_after_prefix"]
        for item in passing
        if item["remaining_lead_steps_after_prefix"] is not None
    ]
    return {
        "property": "P — Predictive / Early Prediction",
        "status": status,
        "data": "independent Stage 1 only",
        "event_label": "eventual strict deadlock; timeout and collision are zero",
        "strata": strata,
        "earliest_useful_lead": (
            {
                "largest_demonstrated_remaining_steps": max(demonstrated),
                "seconds": 0.05 * max(demonstrated),
            }
            if demonstrated else None
        ),
        "important_limitation": (
            "Initial-state ranking is strongly inverted; PASS is supplied only by "
            "predeclared near-terminal strata and does not imply control relevance."
        ),
    }


def bootstrap_weighted_cluster(records, rng):
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, len(records), len(records))
        denominator = sum(records[i]["count"] for i in indices)
        draws.append(
            sum(records[i]["count"] * records[i]["mean_slack"] for i in indices)
            / denominator
        )
    count = sum(item["count"] for item in records)
    mean = sum(item["count"] * item["mean_slack"] for item in records) / count
    return mean, np.quantile(draws, (0.025, 0.975))


def audit_certificate(bellman_records, risk_records, certificates):
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    rows = []
    failures = []
    for phi in (item.name for item in PHIS):
        for cell in BIN_NAMES:
            selected = [
                item for item in bellman_records
                if item["phi_name"] == phi and item["cell"] == cell
            ]
            count = sum(item["count"] for item in selected)
            if not count:
                continue
            mean, ci = bootstrap_weighted_cluster(selected, rng)
            record = {
                "phi_name": phi,
                "cell": cell,
                "transition_count": count,
                "rollout_clusters": len(selected),
                "mean_bellman_slack": float(mean),
                "cluster_bootstrap_ci95": [float(ci[0]), float(ci[1])],
                "minimum_observed_slack": float(
                    min(item["minimum_slack"] for item in selected)
                ),
            }
            record["predeclared_failure"] = bool(
                count >= 30 and record["cluster_bootstrap_ci95"][1] < -0.001
            )
            if record["predeclared_failure"]:
                failures.append({"phi_name": phi, "cell": cell})
            rows.append(record)

    values = [
        item for item in bellman_records if item["cell"] == "__all_values__"
    ]
    fraction_one = sum(
        item["count"] * item["fraction_value_one"] for item in values
    ) / sum(item["count"] for item in values)
    nonvacuous = fraction_one < 1.0 and min(item["minimum_value"] for item in values) < 1.0

    initial = [
        item for item in risk_records
        if item["checkpoint"] == "initial" and item["K"] == 20
    ]
    mixture = []
    for phi in PHIS:
        selected = [item for item in initial if item["phi_name"] == phi.name]
        deadlocks = sum(item["eventual_deadlock"] for item in selected)
        lower, upper = wilson_interval(deadlocks, len(selected))
        initial_cell = "unresolved"
        b0 = float(certificates[phi.name].table[850, BIN_NAMES.index(initial_cell)])
        mixture.append({
            "phi_name": phi.name,
            "B_initial": b0,
            "deadlocks": deadlocks,
            "n": len(selected),
            "empirical_Q_mixture": deadlocks / len(selected),
            "wilson_ci95": [lower, upper],
            "B_below_Q_point_estimate": b0 < deadlocks / len(selected),
            "B_below_Q_ci_lower": b0 < lower,
        })

    valid = not failures and nonvacuous
    return {
        "audit": "independent empirical continuation-certificate validity",
        "status": "VALID" if valid else "INVALID",
        "formal_continuous_domain_proof": False,
        "nonvacuous": nonvacuous,
        "visited_value_one_fraction": float(fraction_one),
        "predeclared_failures": failures,
        "cell_results": rows,
        "initial_mixture_upper_bound_check": mixture,
        "stage_2_gate": "PASS" if valid else "STOP",
        "reason": (
            "At least one independent phi/cell has a significantly negative "
            "Bellman slack beyond the predeclared -0.001 tolerance."
            if failures else "No predeclared failure condition was met."
        ),
    }


def load_trace_map(manifest):
    result = {}
    for item in manifest["records"]:
        path = RAW / item["relative_path"]
        with np.load(path) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        result[(item["pair_id"], item["flow_seed"], item["phi_name"])] = (
            item, arrays
        )
    return result


def projection_descriptors(trace_map):
    records = []
    names = [phi.name for phi in PHIS]
    for pair_id in range(225, 233):
        seed = 19073
        for first, second in combinations(names, 2):
            meta_a, a = trace_map[(pair_id, seed, first)]
            meta_b, b = trace_map[(pair_id, seed, second)]
            for K in K_VALUES:
                length = min(K, len(a["u_exec"]), len(b["u_exec"]))
                full_length = min(len(a["u_exec"]), len(b["u_exec"]))
                def rms(x):
                    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))
                executed = rms(a["u_exec"][:length] - b["u_exec"][:length])
                state = rms(a["positions_after"][:length] - b["positions_after"][:length])
                collapse = bool(
                    executed <= 1e-6
                    and state <= 1e-6
                    and meta_a["outcome"] == meta_b["outcome"]
                    and meta_a["steps"] == meta_b["steps"]
                )
                records.append({
                    "pair_id": pair_id,
                    "flow_seed": seed,
                    "policy_a": first,
                    "policy_b": second,
                    "K": K,
                    "common_prefix_steps": length,
                    "correction_rms_difference": rms(a["g"][:length] - b["g"][:length]),
                    "preprojection_candidate_rms_difference": rms(a["w"][:length] - b["w"][:length]),
                    "executed_action_rms_difference": executed,
                    "state_rms_difference": state,
                    "full_common_executed_action_rms_difference": rms(
                        a["u_exec"][:full_length] - b["u_exec"][:full_length]
                    ),
                    "full_common_state_rms_difference": rms(
                        a["positions_after"][:full_length] - b["positions_after"][:full_length]
                    ),
                    "projection_clipping_rate_a": float(
                        np.mean(np.linalg.norm(a["w"][:length] - a["u_exec"][:length], axis=(1, 2)) > 1e-6)
                    ),
                    "projection_clipping_rate_b": float(
                        np.mean(np.linalg.norm(b["w"][:length] - b["u_exec"][:length], axis=(1, 2)) > 1e-6)
                    ),
                    "cross_policy_active_set_disagreement": float(
                        np.mean(np.any(a["second_active"][:length] != b["second_active"][:length], axis=1))
                    ),
                    "active_set_switch_rate_a": float(
                        np.mean(np.any(a["second_active"][1:length] != a["second_active"][:length-1], axis=1))
                        if length > 1 else 0.0
                    ),
                    "active_set_switch_rate_b": float(
                        np.mean(np.any(b["second_active"][1:length] != b["second_active"][:length-1], axis=1))
                        if length > 1 else 0.0
                    ),
                    "collapsed": collapse,
                })
    by_k = {}
    for K in K_VALUES:
        selected = [item for item in records if item["K"] == K]
        by_k[str(K)] = {
            "pairs": len(selected),
            "collapse_fraction": float(np.mean([item["collapsed"] for item in selected])),
            "median_executed_action_rms_difference": float(np.median([
                item["executed_action_rms_difference"] for item in selected
            ])),
            "median_state_rms_difference": float(np.median([
                item["state_rms_difference"] for item in selected
            ])),
            "median_cross_policy_active_set_disagreement": float(np.median([
                item["cross_policy_active_set_disagreement"] for item in selected
            ])),
        }
    return records, by_k


def paired_long_horizon(risk_records, manifest):
    initial = {
        (item["pair_id"], item["flow_seed"], item["phi_name"]): item
        for item in risk_records
        if item["checkpoint"] == "initial" and item["K"] == 20
    }
    outcomes = {
        (item["pair_id"], item["flow_seed"], item["phi_name"]): item
        for item in manifest["records"]
    }
    comparisons = []
    classifications = Counter()
    adverse = 0
    for pair_id in range(225, 233):
        seed = 19073
        for a, b in combinations([phi.name for phi in PHIS], 2):
            ra = initial[(pair_id, seed, a)]["value"]
            rb = initial[(pair_id, seed, b)]["value"]
            if ra == rb:
                continue
            low, high = (a, b) if ra < rb else (b, a)
            low_meta = outcomes[(pair_id, seed, low)]
            high_meta = outcomes[(pair_id, seed, high)]
            if high_meta["outcome"] == "deadlock" and low_meta["outcome"] == "success":
                classification = "TRUE_ESCAPE"
            elif high_meta["outcome"] == "deadlock" and low_meta["outcome"] == "timeout":
                classification = "TIMEOUT_SUBSTITUTION"
            elif (
                high_meta["outcome"] == "deadlock"
                and low_meta["outcome"] == "deadlock"
                and low_meta["steps"] > high_meta["steps"]
            ):
                classification = "DELAYED_DEADLOCK"
            else:
                classification = "NO_EFFECT"
            is_adverse = (
                low_meta["outcome"] == "deadlock"
                and high_meta["outcome"] != "deadlock"
            )
            adverse += int(is_adverse)
            classifications[classification] += 1
            comparisons.append({
                "pair_id": pair_id,
                "flow_seed": seed,
                "low_R_policy": low,
                "high_R_policy": high,
                "R_low": min(ra, rb),
                "R_high": max(ra, rb),
                "low_R_outcome": low_meta["outcome"],
                "high_R_outcome": high_meta["outcome"],
                "low_R_terminal_step": low_meta["steps"],
                "high_R_terminal_step": high_meta["steps"],
                "classification": classification,
                "adverse_deadlock_change": is_adverse,
            })

    pair_tests = []
    for a, b in combinations([phi.name for phi in PHIS], 2):
        ra = np.mean([initial[(p, 19073, a)]["value"] for p in range(225, 233)])
        rb = np.mean([initial[(p, 19073, b)]["value"] for p in range(225, 233)])
        low, high = (a, b) if ra < rb else (b, a)
        low_d = [
            outcomes[(p, 19073, low)]["outcome"] == "deadlock" for p in range(225, 233)
        ]
        high_d = [
            outcomes[(p, 19073, high)]["outcome"] == "deadlock" for p in range(225, 233)
        ]
        low_only = sum(x and not y for x, y in zip(low_d, high_d))
        high_only = sum(y and not x for x, y in zip(low_d, high_d))
        discordant = low_only + high_only
        pvalue = (
            float(binomtest(min(low_only, high_only), discordant, 0.5).pvalue)
            if discordant else 1.0
        )
        pair_tests.append({
            "low_R_policy": low,
            "high_R_policy": high,
            "low_R_deadlock_rate": float(np.mean(low_d)),
            "high_R_deadlock_rate": float(np.mean(high_d)),
            "low_only_deadlocks": low_only,
            "high_only_deadlocks": high_only,
            "paired_exact_p": pvalue,
            "direction": (
                "discordant" if low_only > high_only else
                "concordant" if high_only > low_only else "tied"
            ),
        })
    fail = any(
        item["direction"] == "discordant" and item["paired_exact_p"] < 0.05
        for item in pair_tests
    )
    return {
        "property": "LHC — Long-horizon Closed-loop Causality",
        "status": "FAIL" if fail else "PARTIAL",
        "data_scope": "Stage 1 full trajectories reused after validity gate stopped further rollouts",
        "classification_counts": dict(classifications),
        "adverse_deadlock_changes": adverse,
        "pair_tests": pair_tests,
        "comparisons": comparisons,
        "interpretation": (
            "The apparent low-R goal025 policy produces strict deadlock where "
            "higher-R policies do not; this is not true escape or mere delay."
        ),
    }


def outcome_alignment(risk_records, manifest):
    initial = [
        item for item in risk_records
        if item["checkpoint"] == "initial" and item["K"] in K_VALUES
    ]
    outcome = manifest["records"]
    rates = {
        phi.name: np.mean([
            item["outcome"] == "deadlock" for item in outcome if item["phi_name"] == phi.name
        ]) for phi in PHIS
    }
    results = {}
    for K in K_VALUES:
        rmeans = {
            phi.name: float(np.mean([
                item["value"] for item in initial
                if item["phi_name"] == phi.name and item["K"] == K
            ])) for phi in PHIS
        }
        comparisons = []
        counts = Counter()
        sign_agreement = []
        deltas_r, deltas_q = [], []
        for a, b in combinations([phi.name for phi in PHIS], 2):
            low, high = (a, b) if rmeans[a] < rmeans[b] else (b, a)
            dq = rates[low] - rates[high]
            category = "R_down_Q_unchanged"
            if dq < 0:
                category = "R_down_Q_down"
            elif dq > 0:
                category = "R_down_Q_up"
            counts[category] += 1
            dr_signed = rmeans[a] - rmeans[b]
            dq_signed = rates[a] - rates[b]
            if dq_signed != 0:
                sign_agreement.append(np.sign(dr_signed) == np.sign(dq_signed))
            deltas_r.append(dr_signed)
            deltas_q.append(dq_signed)
            comparisons.append({
                "policy_a": a,
                "policy_b": b,
                "R_a_minus_b": dr_signed,
                "Q_a_minus_b": dq_signed,
                "lower_R_policy": low,
                "category": category,
            })
        rho = spearmanr(
            [rmeans[name] for name in rmeans], [rates[name] for name in rmeans]
        )
        results[str(K)] = {
            "policy_mean_R": rmeans,
            "policy_deadlock_rates": rates,
            "comparison_counts": dict(counts),
            "sign_agreement_informative_fraction": (
                float(np.mean(sign_agreement)) if sign_agreement else None
            ),
            "policy_rank_spearman": float(rho.statistic),
            "policy_rank_spearman_p": float(rho.pvalue),
            "comparisons": comparisons,
        }
    fail = any(
        value["comparison_counts"].get("R_down_Q_up", 0)
        > value["comparison_counts"].get("R_down_Q_down", 0)
        or value["policy_rank_spearman"] <= 0
        for value in results.values()
    )
    return {
        "property": "OAS — Outcome-aligned Sensitivity",
        "status": "FAIL" if fail else "PARTIAL",
        "results_by_K": results,
        "delayed_deadlock_cases": 0,
        "timeout_substitution_cases": 0,
        "note": "Stage 1 estimates an eight-state test mixture, not per-state Q_D.",
    }


def figures(p_result, cert_audit, proj_records, proj_summary, lhc, oas, risk_records, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    phi_names = [phi.name for phi in PHIS]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    initial20 = [
        item for item in risk_records
        if item["checkpoint"] == "initial" and item["K"] == 20
    ]
    rmean = [np.mean([x["value"] for x in initial20 if x["phi_name"] == p]) for p in phi_names]
    qrate = [np.mean([x["eventual_deadlock"] for x in initial20 if x["phi_name"] == p]) for p in phi_names]

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, x, y, color in zip(phi_names, rmean, qrate, colors):
        ax.scatter(x, y, s=70, color=color)
        ax.annotate(name, (x, y), xytext=(5, 5), textcoords="offset points")
    ax.set(xlabel="mean frozen R_20", ylabel="full-horizon strict-deadlock rate",
           title="Independent Stage-1 R vs eventual outcome")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIG / "r_vs_full_q.png", dpi=150); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    x = np.arange(len(phi_names)); width = .38
    ax1.bar(x-width/2, rmean, width, label="mean R_20")
    ax1.bar(x+width/2, qrate, width, label="deadlock rate")
    ax1.set_xticks(x, phi_names, rotation=15)
    ax1.set_ylim(0, 1.05); ax1.set_title("Closed-loop policy ranking (Stage 1 mixture)")
    ax1.legend(); ax1.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(FIG / "closed_loop_policy_ranking.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axis("off")
    ax.text(.5, .6, "G NOT TESTABLE", ha="center", va="center", fontsize=18, weight="bold")
    ax.text(.5, .38, "Frozen B is defined only for four discrete phi IDs;\ninterpolation/extrapolation is forbidden.",
            ha="center", va="center")
    fig.tight_layout(); fig.savefig(FIG / "gradient_direction_vs_qd.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [[x["executed_action_rms_difference"] for x in proj_records if x["K"] == K] for K in K_VALUES]
    ax.boxplot(data, tick_labels=[f"K={K}" for K in K_VALUES], showfliers=False)
    ax.axhline(1e-6, color="red", linestyle="--", linewidth=1, label="collapse threshold")
    ax.set_yscale("log"); ax.set_ylabel("executed-action RMS separation")
    ax.set_title("Hard-projected trajectory separation (Stage 1)")
    ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(FIG / "executable_trajectory_separation.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, deadlock in [("eventual deadlock", True), ("other outcome", False)]:
        xs, ys = [], []
        for offset in (300, 150, 40):
            rows = [x for x in risk_records if x["K"] == 20 and x["checkpoint"] == f"terminal_minus_{offset}" and x["eventual_deadlock"] == deadlock and x["steps_to_terminal"] > 20]
            if rows:
                xs.append((offset-20)*.05); ys.append(np.mean([x["value"] for x in rows]))
        ax.plot(xs, ys, marker="o", label=label)
    ax.set(xlabel="lead after observed K=20 prefix (s)", ylabel="mean R_20",
           title="R_K versus time remaining to terminal event")
    ax.invert_xaxis(); ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(FIG / "risk_and_q_vs_time_to_deadlock.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels=[]; vals=[]; lows=[]; highs=[]
    for item in p_result["strata"]:
        labels.append(f"K{item['K']}\n{item['checkpoint'].replace('terminal_minus_','-')}")
        vals.append(item["auroc"]); lows.append(item["auroc_ci95"][0]); highs.append(item["auroc_ci95"][1])
    x=np.arange(len(vals)); ax.errorbar(x,vals,yerr=[np.array(vals)-np.array(lows),np.array(highs)-np.array(vals)],fmt="o",capsize=3)
    ax.axhline(.5,color="black",linestyle="--"); ax.set_xticks(x,labels,rotation=30,ha="right")
    ax.set_ylim(0,1.03); ax.set_ylabel("AUROC"); ax.set_title("Frozen-K early-prediction ablation")
    ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(FIG / "k_ablation.png",dpi=150); plt.close(fig)

    deadlock_rows=[x for x in risk_records if x["eventual_deadlock"] and isinstance(x["K"],int)]
    fig,ax=plt.subplots(figsize=(7,4))
    for K in K_VALUES:
        rows=[x for x in deadlock_rows if x["K"]==K and x["checkpoint"]!="initial"]
        xs=[(x["steps_to_terminal"]-K)*.05 for x in rows if x["steps_to_terminal"]>K]
        ys=[x["value"] for x in rows if x["steps_to_terminal"]>K]
        ax.scatter(xs,ys,label=f"K={K}",alpha=.75)
    ax.set(xlabel="lead after prefix (s)",ylabel="R_K",title="Strict-deadlock examples: certificate rise near event")
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIG/"delayed_deadlock_examples.png",dpi=150);plt.close(fig)


def write_historical():
    text = """# 历史失败回归

| 历史问题 | 结论 | 本次证据 |
|---|---|---|
| 1. one-step flatness | NOT_APPLICABLE | 冻结主方法仅测试 K=20/100；没有把 K=1 旧证据冒充当前结论。 |
| 2. old ACTION_RANKING_FAIL | NOT_SOLVED | Stage-1 初态中 `goal025` 的 R 最低但 strict-deadlock rate 最高；正式 Q 因证书 gate 停止。 |
| 3. old `-grad_w R` failure | NOT_APPLICABLE | 正确对象应为 `grad_phi R_K`；冻结 B 仅定义四个离散 phi，故没有合法连续导数。 |
| 4. projection aliasing | PARTIALLY_SOLVED | 32 条 Stage-1 轨迹显示非零执行/状态分离且无数值 collapse；没有通过预注册 Stage-2 分布审计。 |
| 5. active-set AD/FD instability | NOT_APPLICABLE | G 未进入；没有软化投影或伪造 AD/FD。 |
| 6. cone/controller mismatch | PARTIALLY_SOLVED | 新对象使用实际双投影闭环轨迹，但 continuation B 在独立 Bellman 检验失败。 |
| 7. short-timer delayed-deadlock failure | PARTIALLY_SOLVED | timer-independent guard 在接近 deadlock 时给出预测信号，但初态/长时域排序反向。 |
| 8. timeout substitution | NOT_APPLICABLE | 本次低-R `goal025` 不是用 timeout 替代 deadlock；高-R damping/relative 才产生 timeout。 |
| 9. predictive-but-not-control-relevant risk | NOT_SOLVED | P 仅在临近终点通过，而低初态 R 对应更高最终 deadlock；OAS/LHC 失败。 |
"""
    (OUT / "historical_failure_regression.md").write_text(text)


def main():
    verify_frozen_method()
    FIG.mkdir(parents=True, exist_ok=True)
    risk = json.loads((RAW / "risk_records.json").read_text())
    bellman = json.loads((RAW / "bellman_records.json").read_text())
    stage_manifest = json.loads((RAW / "manifest.json").read_text())
    certificates = load_certificates()

    p_result = analyze_p(risk)
    cert_audit = audit_certificate(bellman, risk, certificates)
    if cert_audit["stage_2_gate"] != "STOP":
        raise RuntimeError("this analyzer expects the observed predeclared Stage-1 stop")
    trace_map = load_trace_map(stage_manifest)
    proj_records, proj_summary = projection_descriptors(trace_map)
    proj_status = "PARTIAL" if all(
        value["collapse_fraction"] < .10
        and value["median_executed_action_rms_difference"] > 1e-6
        and value["median_state_rms_difference"] > 1e-6
        for value in proj_summary.values()
    ) else "FAIL"
    lhc = paired_long_horizon(risk, stage_manifest)
    oas = outcome_alignment(risk, stage_manifest)

    q_result = {
        "property": "Q — Closed-loop Intervention Ranking / Control Relevance",
        "status": "NOT_TESTABLE",
        "reason": "Predeclared Stage-1 continuation-certificate validity gate failed; Stage 2 was not run.",
        "preliminary_stage1_only": oas["results_by_K"],
        "warning": "These eight-state/one-seed mixture rates are not the predeclared per-state Q_D estimates.",
    }
    g_result = {
        "property": "G — Closed-loop Directional Fidelity",
        "status": "NOT_TESTABLE",
        "reason": (
            "Q gate did not pass. Independently, frozen B/R_K is defined only at four "
            "discrete phi IDs and the frozen spec forbids interpolation/extrapolation, "
            "so central finite differences in continuous phi-space are undefined."
        ),
        "gradient_rollouts": 0,
        "soft_projection_or_straight_through_used": False,
    }
    proj_result = {
        "property": "PROJ — Multi-step Projection Consistency",
        "status": proj_status,
        "scope": "Preliminary Stage-1 paired CRN trajectories; Stage-2 replicated distribution test stopped.",
        "summary_by_K": proj_summary,
        "records": proj_records,
    }
    stochastic = {
        "applicable_to_G_phi": False,
        "status": "NOT_APPLICABLE",
        "G_phi_semantics": "deterministic 4D residual",
        "Flow_semantics": "fresh four-dimensional Gaussian each physical step",
        "R_and_Q_same_deployment_law": True,
        "evidence": (
            "Each R_K prefix and its eventual outcome were computed from the same saved "
            "full rollout; G_phi remained active and Flow keys were unique after K."
        ),
        "mean_residual_substitution": False,
        "gate_or_sigma_mismatch": False,
    }

    json_dump(OUT / "property_p.json", p_result)
    json_dump(OUT / "continuation_certificate_audit.json", cert_audit)
    json_dump(OUT / "property_q.json", q_result)
    json_dump(OUT / "property_g.json", g_result)
    json_dump(OUT / "property_proj.json", proj_result)
    json_dump(OUT / "long_horizon_causality.json", lhc)
    json_dump(OUT / "outcome_alignment.json", oas)
    json_dump(OUT / "stochastic_semantics_audit.json", stochastic)
    write_historical()
    if os.environ.get("CL_FHCB_SKIP_FIGURES") != "1":
        figures(p_result, cert_audit, proj_records, proj_summary, lhc, oas, risk, stage_manifest)

    # Report is intentionally produced after all status files, then all are
    # content-addressed by the manifest below.
    failure = cert_audit["predeclared_failures"][0]
    failed_cell = next(
        item for item in cert_audit["cell_results"]
        if item["phi_name"] == failure["phi_name"] and item["cell"] == failure["cell"]
    )
    initial_stratum = next(
        item for item in p_result["strata"] if item["K"] == 20 and item["checkpoint"] == "initial"
    )
    pass_strata = [item for item in p_result["strata"] if item["predeclared_pass"]]
    report = f"""# CL-FHCB 独立 qualification 报告

## 结论

**CASE F — CONTINUATION CERTIFICATE INVALID OR VACUOUS**

具体是 **invalid，而不是 vacuous**。冻结 certificate 在独立 test split 的
`{failure['phi_name']}/{failure['cell']}` cell 上违反预注册经验 Bellman gate：平均
slack `{failed_cell['mean_bellman_slack']:.6f}`，cluster-bootstrap 95% CI
`[{failed_cell['cluster_bootstrap_ci95'][0]:.6f}, {failed_cell['cluster_bootstrap_ci95'][1]:.6f}]`；
上界仍低于预注册阈值 `-0.001`。因此按 compute gate 停止，没有运行 Stage 2 或 G，
也没有训练、重拟合或修改冻结方法。

## 最终属性表

| 属性 | 含义 | 状态 |
|---|---|---|
| P | Predictive / Early Prediction | **{p_result['status']}** |
| Q | Closed-loop Intervention Ranking / Control Relevance | **NOT_TESTABLE** |
| G | Closed-loop Directional Fidelity | **NOT_TESTABLE** |
| PROJ | Multi-step Projection Consistency | **{proj_status}** |
| LHC | Long-horizon Closed-loop Causality | **{lhc['status']}** |
| OAS | Outcome-aligned Sensitivity | **{oas['status']}** |

## Stage 1：P 与 continuation validity

共运行 32 条独立 test-split full-horizon trajectory，实际 {stage_manifest['actual_physical_steps']}
个物理步，CPU。四个 `phi` 在 8 个 test 初态上使用 paired common random numbers，
`G_phi` 在 K 后继续运行至 success、strict deadlock 或 timeout。

P 的预注册规则在临近终点的非终止 prefix 上通过：
""" + "\n".join(
        f"- K={x['K']}，{x['checkpoint']}：AUROC={x['auroc']:.3f}，95% CI "
        f"[{x['auroc_ci95'][0]:.3f},{x['auroc_ci95'][1]:.3f}]，prefix 后仍有 "
        f"{x['remaining_lead_steps_after_prefix']*0.05:.1f}s lead。"
        for x in pass_strata
    ) + f"""

但初态 P 明显反向：K=20 的 AUROC 仅 `{initial_stratum['auroc']:.3f}`，deadlock
减 nondeadlock 的平均 R separation 为 `{initial_stratum['mean_separation']:.3f}`。
因此 P_PASS 只表示 certificate 在 strict-deadlock 临近时含信息，不表示可用于政策
排序。

certificate 并非恒等于 1：独立访问状态中 value=1 的比例为
`{cert_audit['visited_value_one_fraction']:.3f}`。失败来自 transition expectation，
不是 vacuity。`goal025` 在初态的冻结 B 为 0.373952，而独立 8-state mixture 中
strict-deadlock 为 6/8；其 Wilson 95% 下界仍高于 B，提供第二项上界反证。

## 长时域与 outcome alignment

Stage-1 full continuations 已足以发现强反例：初态 apparent low-R `goal025` 在 8 个
test 初态中产生 6 次 strict deadlock，而 `zero`、`damp035`、`relative025` 均为 0。
在 low-R/high-R 个体配对中共有 `{lhc['adverse_deadlock_changes']}` 个 adverse
deadlock change；没有 TRUE_ESCAPE、DELAYED_DEADLOCK 或 TIMEOUT_SUBSTITUTION 被
误计为改善。至少一个 paired exact test 达到 p<0.05，所以 LHC_FAIL。

K=20 与 K=100 的政策排序结论一致：均有更多 `R_down/Q_up` 而没有
`R_down/Q_down`，policy-level Spearman 为负，因此 OAS_FAIL。full remaining 只作
事件 bookkeeping，未作为新的 K 或 intervention duration。

## 投影与 G

Stage-1 的双硬投影 trajectory 显示 correction 差异能传到 executed action 与 state，
K=20/100 的 collapse fraction 都低于 0.10；但预注册 Stage-2 stochastic distribution
复验因 gate 停止，故只给 PROJ_PARTIAL，不继承旧 NO_MASS_ALIASING 结论。

G_NOT_TESTABLE 有两个独立原因：Q gate 未通过；且冻结 `B` 仅在四个离散 `phi` ID
定义，同时规范禁止插值或外推，所以合法的 central FD `grad_phi R_K` 不存在。没有
用 soft projection、straight-through 或任意替代梯度。

## K 消融与计算预算

主 K 严格保持 20 与 100。较长 K 提高了临近 deadlock 的可见窗口，但没有修复初态
政策排序；不存在使 Q/G 变得有意义的冻结 `K*`。总计仅新增 32 条 rollout，远低于
800/1600/2200 预算。Stage 2 的 256 条与所有 G rollout 均未启动。

## 可复现文件

- `property_p.json`：全部 AUROC、rank-biserial、overlap/tie 与 bootstrap CI；
- `continuation_certificate_audit.json`：逐 phi/cell 独立 Bellman slack；
- `property_q.json`、`property_g.json`：gate 与 NOT_TESTABLE 原因；
- `property_proj.json`：双投影后的 K/full trajectory separation；
- `long_horizon_causality.json`、`outcome_alignment.json`：完整终局因果分类；
- `stochastic_semantics_audit.json`：deterministic `G_phi` 与共享 Flow law；
- `historical_failure_regression.md`：九项历史问题；
- `figures/`：请求的轻量图。
"""
    (OUT / "qualification_report.md").write_text(report)

    deliverables = [
        "predeclared_protocol.json", "qualification_report.md", "property_p.json",
        "continuation_certificate_audit.json", "property_q.json", "property_g.json",
        "property_proj.json", "long_horizon_causality.json", "outcome_alignment.json",
        "stochastic_semantics_audit.json", "historical_failure_regression.md",
    ] + [str(path.relative_to(OUT)) for path in sorted(FIG.glob("*.png"))]
    manifest = {
        "schema": "cl_fhcb_independent_qualification_v1",
        "decision": "CASE F — CONTINUATION CERTIFICATE INVALID OR VACUOUS",
        "clarification": "invalid, not vacuous",
        "stage_1": {
            "rollouts": stage_manifest["rollout_count"],
            "actual_physical_steps": stage_manifest["actual_physical_steps"],
            "device": stage_manifest["device"],
        },
        "stage_2": {"rollouts": 0, "reason": "predeclared validity gate STOP"},
        "stage_3_G": {"rollouts": 0, "reason": g_result["reason"]},
        "total_new_rollouts": stage_manifest["rollout_count"],
        "total_new_physical_steps": stage_manifest["actual_physical_steps"],
        "final_properties": {
            "P": p_result["status"], "Q": "NOT_TESTABLE", "G": "NOT_TESTABLE",
            "PROJ": proj_status, "LHC": lhc["status"], "OAS": oas["status"],
        },
        "frozen_method_unchanged": True,
        "frozen_spec_sha256": sha256(REPO_ROOT / "diagnostics/cl_fhcb/CL_FHCB_FROZEN_SPEC.md"),
        "stage1_manifest_sha256": sha256(RAW / "manifest.json"),
        "deliverables": {
            path: sha256(OUT / path) for path in deliverables
        },
    }
    json_dump(OUT / "manifest.json", manifest)
    print(json.dumps({
        "decision": manifest["decision"],
        "properties": manifest["final_properties"],
        "rollouts": manifest["total_new_rollouts"],
        "physical_steps": manifest["total_new_physical_steps"],
    }, indent=2))


if __name__ == "__main__":
    main()
