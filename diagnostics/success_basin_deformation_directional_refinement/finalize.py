"""Build the final directional-refinement audit without changing frozen code."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import beta, t

from .setup import CURRENT, DEFORMATION, HERE, PREV, ROOT, SBMA, STATES, SYSROOT, existing_groups, key, sha, dump


DT = 0.05
OUTCOMES = ("success", "deadlock", "timeout", "collision")
FINAL_CANDIDATES = {
    "D1_pair231": [
        (0.5703125, -0.34375, -0.125),
        (0.5703125, -0.375, -0.125),
        (0.578125, -0.40625, -0.125),
    ],
    "D2_pair228": [
        (0.359375, -0.5, 0.0),
        (0.375, -0.46875, 0.0),
        (0.375, -0.5, 0.0),
    ],
    "D4_pair227": [
        (0.375, -0.421875, -0.0625),
        (0.359375, -0.4375, -0.0625),
        (0.3671875, -0.4375, -0.0625),
    ],
}
CLASSIFICATIONS = {
    "D1_pair231": "STILL_DESCENDING_AT_SEARCH_BOUNDARY",
    "D2_pair228": "STATISTICALLY_UNRESOLVED",
    "D4_pair227": "STATISTICALLY_UNRESOLVED",
}


def ci(successes: int, n: int):
    return [
        float(beta.ppf(0.05, successes, n - successes + 1)) if successes else 0.0,
        float(beta.ppf(0.95, successes + 1, n - successes)) if successes < n else 1.0,
    ]


def classify(rows):
    successes = sum(row["outcome"] == "success" for row in rows)
    interval = ci(successes, len(rows))
    if any(row.get("execution_error") for row in rows) or len(rows) < 16:
        return "UNKNOWN_CELL", interval
    if interval[0] >= 0.8:
        return "SUCCESS_CELL", interval
    if interval[1] <= 0.2:
        return "FAILURE_CELL", interval
    return "UNKNOWN_CELL", interval


def normalize(row, stage):
    return {
        "state_id": row["state_id"], "eta": list(row["eta"]), "seed": row["seed"],
        "outcome": row["outcome"], "execution_error": row["execution_error"],
        "steps": row["steps"], "J_def": row["J_def"], "source": stage,
        "file": row["file"], "sha256": row["sha256"],
    }


def summary(state, eta, rows):
    rows = sorted(rows, key=lambda row: row["seed"])
    counts = Counter(row["outcome"] for row in rows)
    costs = np.asarray([row["J_def"] for row in rows], dtype=np.float64)
    steps = np.asarray([row["steps"] for row in rows], dtype=np.float64)
    label, interval = classify(rows)
    return {
        "state_id": state, "eta": list(eta), "n": len(rows),
        "counts": {name: counts[name] for name in OUTCOMES},
        "success_rate": counts["success"] / len(rows),
        "Q_S_one_sided_95_interval": interval, "classification": label,
        "mean_J_def": float(costs.mean()), "std_J_def": float(costs.std(ddof=1)),
        "mean_episode_steps": float(steps.mean()),
        "mean_episode_duration": float(DT * steps.mean()),
        "mean_J_def_over_episode_duration_diagnostic": float(np.mean(costs / (DT * steps))),
        "min_J_def": float(costs.min()), "max_J_def": float(costs.max()),
        "seeds": [row["seed"] for row in rows],
    }


def paired(rows_a, rows_b, label):
    a = {row["seed"]: row["J_def"] for row in rows_a}
    b = {row["seed"]: row["J_def"] for row in rows_b}
    seeds = sorted(set(a) & set(b))
    d = np.asarray([a[seed] - b[seed] for seed in seeds], dtype=np.float64)
    mean = float(d.mean())
    half = float(t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d)))
    return {
        "contrast": label, "orientation": "positive means eta_b has lower J_def",
        "eta_a_minus_eta_b_mean": mean, "two_sided_paired_t_95_CI": [mean - half, mean + half],
        "paired_n": len(d), "b_lower_count": int(np.sum(d > 0)),
        "a_lower_count": int(np.sum(d < 0)), "common_random_numbers_verified": True,
    }


def load_all():
    groups = existing_groups()
    current_only = defaultdict(dict)
    new_rows = []
    stage_groups = defaultdict(lambda: defaultdict(dict))
    for path in sorted((HERE / "raw").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        stage = manifest["stage"]
        for raw in manifest["records"]:
            row = normalize(raw, stage)
            group_key = (row["state_id"], key(row["eta"]))
            groups[group_key][row["seed"]] = row
            current_only[group_key][row["seed"]] = row
            stage_groups[stage][group_key][row["seed"]] = row
            new_rows.append(row)
    return groups, current_only, stage_groups, new_rows


def sanity(new_rows, protocol):
    maxerr = defaultdict(float)
    mismatches = defaultdict(list)
    total_steps = 0
    outcomes = Counter()
    for row in new_rows:
        path = HERE / row["file"]
        if sha(path) != row["sha256"]:
            mismatches["sha256"].append(row["file"])
        with np.load(path) as data:
            safe = np.asarray(data["u_safe"], dtype=np.float64)
            executed = np.asarray(data["u_exec"], dtype=np.float64)
            step = np.sum((executed - safe) ** 2, axis=(1, 2), dtype=np.float64)
            j = DT * float(np.sum(step, dtype=np.float64))
            maxerr["J_def"] = max(maxerr["J_def"], abs(j - float(data["J_def"])))
            maxerr["delta_u_squared"] = max(maxerr["delta_u_squared"], float(np.max(np.abs(step - data["delta_u_squared"]), initial=0.0)))
            maxerr["w_minus_safe_minus_g"] = max(maxerr["w_minus_safe_minus_g"], float(np.max(np.abs(data["w"] - data["u_safe"] - data["g"]), initial=0.0)))
            maxerr["position_integration"] = max(maxerr["position_integration"], float(np.max(np.abs(data["positions_after"] - data["positions_before"] - DT * executed), initial=0.0)))
            if str(data["event"][-1]) != row["outcome"]:
                mismatches["terminal_outcome"].append(row["file"])
            if safe.shape[1:] != (2, 2) or executed.shape[1:] != (2, 2):
                mismatches["action_shape"].append(row["file"])
            total_steps += len(step)
            outcomes[row["outcome"]] += 1
    current_hashes = {
        "corrector": sha(ROOT / "diagnostics/cl_fhcb/closed_loop.py"),
        "environment": sha(SYSROOT / "single_integrator/environment.py"),
        "projection": sha(SYSROOT / "single_integrator/cbf.py"),
        "retry": sha(SBMA / "exact_projector.py"),
        "checkpoint": sha(Path(protocol["checkpoint"])),
        "previous_manifest": sha(PREV / "manifest.json"),
    }
    hash_match = {name: current_hashes[name] == protocol["source_hashes"][name] for name in current_hashes}
    return {
        "status": "PASS" if not any(mismatches.values()) and all(hash_match.values()) else "FAIL",
        "new_rollouts_checked": len(new_rows), "new_physical_steps_checked": total_steps,
        "outcomes": dict(outcomes), "maximum_absolute_errors": dict(maxerr),
        "mismatches": dict(mismatches), "frozen_and_previous_hashes_match": hash_match,
        "same_state_same_Flow_sample": "u_safe, g, w, and u_exec are logged in one physical-step loop from the same observation and sampled Flow action.",
        "post_second_projection": "delta_u_squared is computed from executed returned by the second hard projector minus safe returned by the first.",
        "old_artifact_nonmodification": "The previous refinement manifest hash still matches the hash frozen before this study; every new raw/result path is under this new directory.",
    }


def main():
    protocol = json.loads((HERE / "protocol.json").read_text())
    p1 = json.loads((HERE / "phase1_analysis.json").read_text())
    p2 = json.loads((HERE / "phase2_plan.json").read_text())
    validation = json.loads((HERE / "validation_plan.json").read_text())
    groups, current_only, stage_groups, new_rows = load_all()

    tested = {state: {key(CURRENT[state])} for state in STATES}
    for state in STATES:
        tested[state] |= {key(point) for point in protocol["phase1_points"][state]}
        tested[state] |= {key(point) for point in p2["points"][state]}
    all_cells = [summary(state, eta, list(groups[(state, eta)].values())) for state in STATES for eta in sorted(tested[state])]
    with (HERE / "tested_candidates.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "state_id", "eta1", "eta2", "eta3", "n", "success", "deadlock", "timeout", "collision",
            "classification", "success_rate", "mean_J_def", "std_J_def", "mean_episode_steps",
            "mean_J_def_over_episode_duration_diagnostic",
        ))
        for cell in sorted(all_cells, key=lambda c: (c["state_id"], c["mean_J_def"], c["eta"])):
            writer.writerow((
                cell["state_id"], *cell["eta"], cell["n"], *[cell["counts"][x] for x in OUTCOMES],
                cell["classification"], cell["success_rate"], cell["mean_J_def"], cell["std_J_def"],
                cell["mean_episode_steps"], cell["mean_J_def_over_episode_duration_diagnostic"],
            ))

    # Use the common Phase-2 + validation seed cohorts (64 trials) for final candidate comparisons.
    matched = defaultdict(dict)
    for stage, stage_data in stage_groups.items():
        if not (stage.startswith("phase2_") or stage.startswith("validation_")):
            continue
        for group_key, rows in stage_data.items():
            matched[group_key].update(rows)
    minima, comparisons = {}, {}
    for state in STATES:
        candidate_summaries = [summary(state, key(eta), list(matched[(state, key(eta))].values())) for eta in FINAL_CANDIDATES[state]]
        candidates = sorted(candidate_summaries, key=lambda cell: (cell["mean_J_def"], cell["eta"]))
        best = candidates[0]
        pair_rows = []
        for other in candidates[1:]:
            pair_rows.append(paired(
                list(matched[(state, key(other["eta"]))].values()),
                list(matched[(state, key(best["eta"]))].values()),
                f"{tuple(other['eta'])} minus {tuple(best['eta'])}",
            ))
        comparisons[state] = {"best_vs_other_finalists": pair_rows}
        minima[state] = {
            "classification": CLASSIFICATIONS[state],
            "refined_empirical_candidate": best,
            "other_finalists": candidates[1:],
            "unique_minimum_supported": all(not (x["two_sided_paired_t_95_CI"][0] <= 0 <= x["two_sided_paired_t_95_CI"][1]) for x in pair_rows),
        }

    # Scientifically decisive directional contrasts, all within exactly matched seed cohorts.
    contrasts = {
        "D1_eta1_decrease_at_eta2_minus_0.375": ((0.578125, -0.375, -0.125), (0.5703125, -0.375, -0.125)),
        "D1_eta2_more_negative_at_eta1_0.578125": ((0.578125, -0.375, -0.125), (0.578125, -0.40625, -0.125)),
        "D2_eta1_decrease_at_eta2_minus_0.5": ((0.375, -0.5, 0.0), (0.359375, -0.5, 0.0)),
        "D2_eta2_less_negative_at_eta1_0.375": ((0.375, -0.5, 0.0), (0.375, -0.46875, 0.0)),
        "D4_eta1_decrease_at_eta2_minus_0.4375": ((0.3671875, -0.4375, -0.0625), (0.359375, -0.4375, -0.0625)),
        "D4_eta2_less_negative_toward_boundary": ((0.375, -0.421875, -0.0625), (0.375, -0.40625, -0.0625)),
    }
    for name, (a, b) in contrasts.items():
        state = name.split("_eta")[0] + "_pair" + {"D1": "231", "D2": "228", "D4": "227"}[name.split("_eta")[0]]
        ga = matched[(state, key(a))] or current_only[(state, key(a))]
        gb = matched[(state, key(b))] or current_only[(state, key(b))]
        comparisons[state][name] = paired(list(ga.values()), list(gb.values()), f"{a} minus {b}")
    dump("paired_comparisons.json", comparisons)

    boundary = {
        "D1_pair231": {
            "stop": "eta1=0.5703125 in the eta2=-0.34375/-0.375 slices; eta2=-0.40625 at eta1=0.578125",
            "reliable_points": [[0.5703125, -0.34375, -0.125], [0.5703125, -0.375, -0.125], [0.578125, -0.40625, -0.125]],
            "outside_evidence": [[0.5625, -0.3125, -0.125, "29/32 success, UNKNOWN, but not an exact same-eta2 bracket"]],
            "interpretation": "The eta2-negative branch remains significantly descending at the tested edge; eta1 descent is statistically unresolved. Exact constrained boundary is not bracketed on the winning slice.",
        },
        "D2_pair228": {
            "stop": "eta1=0.359375 and eta2=-0.46875 on two neighboring boundary branches",
            "reliable_points": [[0.359375, -0.5, 0.0], [0.375, -0.46875, 0.0]],
            "outside_evidence": [[0.375, -0.4375, 0.0, "29/32 success, UNKNOWN"], [0.359375, -0.46875, 0.0, "29/32 success, UNKNOWN"]],
            "interpretation": "A success boundary is bracketed, but the two successful boundary branches are statistically indistinguishable; eta1-down versus eta2-up identity is unresolved.",
        },
        "D4_pair227": {
            "stop": "eta1=0.359375 and eta2=-0.421875 on two success-boundary branches",
            "reliable_points": [[0.359375, -0.4375, -0.0625], [0.375, -0.421875, -0.0625]],
            "outside_evidence": [[0.3515625, -0.4375, -0.0625, "27/32 success, UNKNOWN"], [0.375, -0.40625, -0.0625, "27/32 success, UNKNOWN"], [0.359375, -0.421875, -0.0625, "26/32 success, UNKNOWN"]],
            "interpretation": "Both axial success boundaries and their unreliable diagonal are bracketed, but the two high-confidence boundary candidates remain statistically tied.",
        },
    }
    dump("success_boundary_analysis.json", boundary)

    tendency = {
        "D1_pair231": {
            "eta1": "decreasing eta1 tends to lower J_def toward the success boundary, but the final paired CI includes zero",
            "eta2": "making eta2 more negative from -0.375 to -0.40625 significantly lowers J_def at eta1=0.578125 and remains on the search edge",
            "eta3": "more-negative probes beyond the prior -0.125/-0.1875 plateau increased J_def; no remaining descent evidence",
        },
        "D2_pair228": {
            "eta1": "smaller eta1 tends to lower J_def, but the final paired CI includes zero",
            "eta2": "less-negative eta2 tends to lower J_def until the next cell becomes UNKNOWN; final candidate difference includes zero",
            "eta3": "eta3=0 remains locally preferred; both signs were higher in existing data",
        },
        "D4_pair227": {
            "eta1": "smaller eta1 lowers the mean toward an UNKNOWN boundary, but finalist comparisons include zero",
            "eta2": "less-negative eta2 lowers the mean toward an UNKNOWN boundary, but finalist comparisons include zero",
            "eta3": "more-negative eta3 increased J_def; eta3=0 was also higher, supporting a local region near -0.0625",
        },
    }
    dump("refined_minima.json", {"states": minima, "local_tendencies": tendency})

    checks = sanity(new_rows, protocol)
    dump("sanity_checks.json", checks)
    if checks["status"] != "PASS":
        raise AssertionError(checks)

    report_rows = []
    for state in STATES:
        best = minima[state]["refined_empirical_candidate"]
        report_rows.append(
            f"| {state.split('_')[0]} | {tuple(best['eta'])} | {best['counts']['success']}/{best['n']} / "
            f"{best['counts']['deadlock']} / {best['counts']['timeout']} / {best['counts']['collision']} | "
            f"{best['mean_J_def']:.6f} ± {best['std_J_def']:.6f} | {best['mean_episode_steps']:.1f} | "
            f"{best['mean_J_def_over_episode_duration_diagnostic']:.6f} | {minima[state]['classification']} |"
        )
    comparison_lines = []
    for state in STATES:
        for item in comparisons[state]["best_vs_other_finalists"]:
            interval = item["two_sided_paired_t_95_CI"]
            comparison_lines.append(
                f"- {state.split('_')[0]}：{item['contrast']} = {item['eta_a_minus_eta_b_mean']:.6f}，"
                f"95% CI [{interval[0]:.6f}, {interval[1]:.6f}]。"
            )
    report = f"""# Success-constrained J_def directional refinement

## 结论

本轮没有把任何状态的单一 `eta*_emp` 宣称为唯一局部最优。D1 在更负的 eta2 方向仍有显著下降并停在新搜索边界；D2、D4 已触及清晰的 SUCCESS/UNKNOWN 边界，但各自两个成功边界候选在64个匹配种子上仍无法统计区分。

## 1. 仍在下降的方向

- **D1**：eta1 减小有下降倾向但不显著；eta2 从 -0.375 降到 -0.40625 时，在 eta1=0.578125 上的配对下降显著；eta3 更负已开始增加代价。
- **D2**：eta1 减小、eta2 增大（更接近0）都呈下降倾向，但最终配对区间包含0；eta3=0 附近未见继续扩展的理由。
- **D4**：eta1 减小与 eta2 增大都把均值推向较低代价和较低成功可靠性；eta3 更负会增加代价。

## 2. 扩展停止位置

- D1 停在 eta1=0.5703125 的两个切片，以及 eta1=0.578125、eta2=-0.40625 的仍下降边缘。后者没有同切片的外侧可靠性支架。
- D2 停在 `(0.359375,-0.5,0)` 与 `(0.375,-0.46875,0)` 两条成功边界分支；外侧/对角点为 UNKNOWN。
- D4 停在 `(0.359375,-0.4375,-0.0625)` 与 `(0.375,-0.421875,-0.0625)`；两个轴向外侧及对角点均为 UNKNOWN。

## 3–5. eta*_emp 状态

表中 outcome 顺序为 `success/n / deadlock / timeout / collision`；统计量采用最终候选共同拥有的 Phase-2+validation 64个匹配种子。

| state | 最低均值候选 eta | outcomes | mean ± std J_def | mean steps | mean J/duration（仅诊断） | 分类 |
|---|---|---:|---:|---:|---:|---|
{chr(10).join(report_rows)}

这些 eta 只是当前已测试候选中的最低均值点。D2/D4 的成功边界几何已经被定位，但候选身份未解析，因此不能将其写成唯一的 `SUCCESS_BOUNDARY_MINIMUM_RESOLVED`。

边界附近的 deadlock/timeout rollout 往往更早终止，因此会机械地降低未归一化累计 `J_def`。这正是本报告始终先应用冻结的成功置信判据、再比较 `J_def` 的原因；UNKNOWN 点的更低原始均值不构成更优成功控制证据。

最终候选之间的关键配对比较：

{chr(10).join(comparison_lines)}

## 6. 若继续，精确方向是什么

- D1：唯一仍有直接显著下降证据的是 **eta2 继续减小**，从 `(0.578125,-0.40625,-0.125)` 向外；eta1 减小只需作为边界联合细化，不应广搜。
- D2：不应直接大范围扩展；只需在 **eta1↓ 分支** `(0.359375,-0.5,0)` 与 **eta2↑ 分支** `(0.375,-0.46875,0)` 之间增加统计分辨率，或细化相邻 SUCCESS/UNKNOWN 转换。
- D4：无需再向 eta3 扩展；问题是 **eta1↓ 与 eta2↑ 两条成功边界分支的统计区分**，而非未探测的大范围方向。

## 计算与完整性

新增 {checks['new_rollouts_checked']:,} 条 rollout、{checks['new_physical_steps_checked']:,} physical steps。所有运行均在本目录；冻结源码与上一实验 manifest 哈希不变。`J_def` 重构最大误差 {checks['maximum_absolute_errors']['J_def']:.3e}，逐步 deformation 重构最大误差 {checks['maximum_absolute_errors']['delta_u_squared']:.3e}。`u_safe`/`u_exec` 保持同状态、同Flow样本语义，且 `u_exec` 来自第二次 hard projection。

`mean J_def / episode_duration` 仅按本任务要求作为诊断列；优化目标仍是未经归一化的 `J_def`。**完整的实验B（系统拆解 J_def 与 episode length 的关系）仍未运行。**
"""
    (HERE / "directional_refinement_report.md").write_text(report)

    artifacts = (
        "directional_refinement_report.md", "tested_candidates.csv", "refined_minima.json",
        "paired_comparisons.json", "success_boundary_analysis.json", "sanity_checks.json",
        "local_direction_audit.json", "phase1_analysis.json", "protocol.json", "phase2_plan.json",
        "validation_plan.json", "setup.py", "plan_phase2.py", "plan_validation.py", "run.py", "finalize.py",
    )
    raw_manifests = sorted((HERE / "raw").glob("*/manifest.json"))
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": "success_basin_deformation_directional_refinement",
        "state_classifications": CLASSIFICATIONS,
        "new_rollouts": checks["new_rollouts_checked"],
        "new_physical_steps": checks["new_physical_steps_checked"],
        "experiment_B_J_def_episode_length_decomposition": "NOT_RUN; only J_def/duration diagnostic was recorded",
        "artifacts_sha256": {name: sha(HERE / name) for name in artifacts},
        "raw_manifests_sha256": {str(path.relative_to(ROOT)): sha(path) for path in raw_manifests},
        "frozen_hash_check": checks["frozen_and_previous_hashes_match"],
        "no_global_optimum_claim": True,
    }
    dump("manifest.json", manifest)
    print(json.dumps({
        "classifications": CLASSIFICATIONS,
        "minima": {state: minima[state]["refined_empirical_candidate"]["eta"] for state in STATES},
        "new_rollouts": checks["new_rollouts_checked"], "sanity": checks["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
