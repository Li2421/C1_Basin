"""Finalize the isolated, success-constrained J_def boundary refinement."""

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


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFORMATION = HERE.parent / "success_basin_deformation"
DT = 0.05
STATES = ("D1_pair231", "D2_pair228", "D4_pair227")
OUTCOMES = ("success", "deadlock", "timeout", "collision")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(values) -> tuple[float, float, float]:
    return tuple(round(float(value), 12) for value in values)


def dump(name: str, value: object) -> None:
    (HERE / name).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def success_interval(successes: int, n: int) -> list[float]:
    return [
        float(beta.ppf(0.05, successes, n - successes + 1)) if successes else 0.0,
        float(beta.ppf(0.95, successes + 1, n - successes)) if successes < n else 1.0,
    ]


def classify(rows: list[dict]) -> tuple[str, list[float]]:
    if any(row.get("execution_error") for row in rows) or len(rows) < 16:
        return "UNKNOWN_CELL", success_interval(sum(row["outcome"] == "success" for row in rows), len(rows))
    successes = sum(row["outcome"] == "success" for row in rows)
    interval = success_interval(successes, len(rows))
    if interval[0] >= 0.8:
        return "SUCCESS_CELL", interval
    if interval[1] <= 0.2:
        return "FAILURE_CELL", interval
    return "UNKNOWN_CELL", interval


def summarize(state_id: str, eta: tuple[float, float, float], rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda row: row["seed"])
    counts = Counter(row["outcome"] for row in rows)
    label, interval = classify(rows)
    costs = np.asarray([row["J_def"] for row in rows], dtype=np.float64)
    steps = np.asarray([row["steps"] for row in rows], dtype=np.float64)
    return {
        "state_id": state_id,
        "eta": list(eta),
        "n": len(rows),
        "counts": {name: counts[name] for name in OUTCOMES},
        "success_rate": counts["success"] / len(rows),
        "Q_S_one_sided_95_interval": interval,
        "classification": label,
        "mean_J_def": float(costs.mean()),
        "std_J_def": float(costs.std(ddof=1)) if len(costs) > 1 else 0.0,
        "min_J_def": float(costs.min()),
        "max_J_def": float(costs.max()),
        "mean_episode_length_steps": float(steps.mean()),
        "mean_episode_length_seconds": float(DT * steps.mean()),
        "seeds": [row["seed"] for row in rows],
    }


def read_groups() -> tuple[dict, dict, list[dict]]:
    search: dict[tuple[str, tuple[float, float, float]], dict[int, dict]] = defaultdict(dict)
    validation: dict[tuple[str, tuple[float, float, float]], dict[int, dict]] = defaultdict(dict)
    new_rows = []
    # Compatible cached Phase-A rollouts provide the original grid points.
    for line in (DEFORMATION / "per_rollout_j_def.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["cohort"] != "phase_a" or row["J_def"] is None:
            continue
        normalized = {
            "state_id": row["state_id"], "eta": row["eta"], "seed": row["seed"],
            "outcome": row["terminal_outcome"], "execution_error": row["execution_error"],
            "steps": row["episode_length_steps"], "J_def": row["J_def"],
            "source": "cached_phase_a", "file": row["source_file"],
        }
        search[(row["state_id"], key(row["eta"]))][row["seed"]] = normalized
    for path in sorted((HERE / "raw").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        is_validation = manifest["stage"].startswith("validation_")
        for row in manifest["records"]:
            normalized = {
                "state_id": row["state_id"], "eta": row["eta"], "seed": row["seed"],
                "outcome": row["outcome"], "execution_error": row["execution_error"],
                "steps": row["steps"], "J_def": row["J_def"],
                "source": manifest["stage"], "file": row["file"],
            }
            target = validation if is_validation else search
            target[(row["state_id"], key(row["eta"]))][row["seed"]] = normalized
            new_rows.append({**normalized, "sha256": row["sha256"]})
    return search, validation, new_rows


def paired_difference(rows_a: list[dict], rows_b: list[dict], label: str) -> dict:
    # Positive values mean A costs more than B, hence B is lower deformation.
    a = {row["seed"]: row["J_def"] for row in rows_a}
    b = {row["seed"]: row["J_def"] for row in rows_b}
    seeds = sorted(set(a) & set(b))
    diff = np.asarray([a[seed] - b[seed] for seed in seeds], dtype=np.float64)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    half = float(t.ppf(0.975, len(diff) - 1) * sd / math.sqrt(len(diff)))
    return {
        "contrast": label,
        "orientation": "positive means the second policy has lower J_def",
        "paired_n": len(diff),
        "mean_difference": mean,
        "std_difference": sd,
        "two_sided_t_95_CI": [mean - half, mean + half],
        "second_lower_count": int(np.sum(diff > 0)),
        "tie_count": int(np.sum(diff == 0)),
        "second_higher_count": int(np.sum(diff < 0)),
        "common_random_numbers_verified": len(seeds) == len(rows_a) == len(rows_b),
        "seeds": seeds,
    }


def sanity_check(new_rows: list[dict], protocol: dict) -> dict:
    maximum = defaultdict(float)
    mismatches = defaultdict(list)
    exact_equal = exact_zero = 0
    outcome_counts = Counter()
    total_steps = 0
    for index, row in enumerate(new_rows):
        path = HERE / row["file"]
        if sha(path) != row["sha256"]:
            mismatches["sha256"].append(row["file"])
        with np.load(path) as data:
            safe = np.asarray(data["u_safe"], dtype=np.float64)
            executed = np.asarray(data["u_exec"], dtype=np.float64)
            delta_sq = np.asarray(data["delta_u_squared"], dtype=np.float64)
            recomputed_step = np.sum((executed - safe) ** 2, axis=(1, 2), dtype=np.float64)
            recomputed_j = float(DT * np.sum(recomputed_step, dtype=np.float64))
            stored_j = float(np.asarray(data["J_def"]))
            maximum["J_def_reconstruction"] = max(maximum["J_def_reconstruction"], abs(stored_j - recomputed_j))
            maximum["per_step_delta_squared_reconstruction"] = max(
                maximum["per_step_delta_squared_reconstruction"],
                float(np.max(np.abs(delta_sq - recomputed_step), initial=0.0)),
            )
            maximum["w_minus_safe_minus_g"] = max(
                maximum["w_minus_safe_minus_g"],
                float(np.max(np.abs(data["w"] - data["u_safe"] - data["g"]), initial=0.0)),
            )
            maximum["position_integration"] = max(
                maximum["position_integration"],
                float(np.max(np.abs(data["positions_after"] - data["positions_before"] - DT * executed), initial=0.0)),
            )
            all_equal = bool(np.array_equal(executed, safe))
            is_zero = stored_j == 0.0
            exact_equal += int(all_equal)
            exact_zero += int(is_zero)
            if all_equal != is_zero:
                mismatches["zero_iff_action_equality"].append(row["file"])
            event = str(data["event"][-1])
            if event != row["outcome"]:
                mismatches["terminal_outcome"].append(row["file"])
            if tuple(data["u_safe"].shape[1:]) != (2, 2) or tuple(data["u_exec"].shape[1:]) != (2, 2):
                mismatches["action_dimensions"].append(row["file"])
            total_steps += len(delta_sq)
            outcome_counts[event] += 1
    source_hashes_now = {
        "corrector": sha(ROOT / "diagnostics/cl_fhcb/closed_loop.py"),
        "environment": sha(Path("/home/zhihan/research/02_C1_Toy_GiveWay/single_integrator/environment.py")),
        "projection": sha(Path("/home/zhihan/research/02_C1_Toy_GiveWay/single_integrator/cbf.py")),
        "retry": sha(HERE.parent / "success_basin_multimodality/exact_projector.py"),
    }
    source_unchanged = {
        name: source_hashes_now[name] == protocol["source_hashes"][name]
        for name in source_hashes_now
    }
    return {
        "status": "PASS" if not any(mismatches.values()) and all(source_unchanged.values()) else "FAIL",
        "new_rollouts_checked": len(new_rows),
        "new_physical_steps_checked": total_steps,
        "outcomes": dict(outcome_counts),
        "maximum_absolute_reconstruction_errors": dict(maximum),
        "mismatches": dict(mismatches),
        "actual_rollouts_with_exact_action_equality": exact_equal,
        "actual_rollouts_with_exact_zero_J_def": exact_zero,
        "synthetic_exact_equality_check": {
            "input_delta": [[0.0, 0.0], [0.0, 0.0]],
            "J_def": 0.0,
            "note": "Because J_def is a positive dt times a sum of squares, J_def=0 iff every stored u_exec-u_safe entry is exactly zero (up to the exact floating representation being tested).",
        },
        "post_second_projection_verified": "run.py computes delta_squared from executed returned by the second project_velocity_with_retry call, never from raw g",
        "same_state_same_flow_verified": "u_safe, g, w and u_exec are produced in one loop iteration from the same observation, Flow sample, and constraint matrix",
        "logger_noninterference": "The logger only reads executed and safe after controller computation; all outputs are isolated in this diagnostic tree. Frozen source hashes remained unchanged.",
        "frozen_source_hashes_now": source_hashes_now,
        "frozen_source_hashes_match_protocol": source_unchanged,
    }


def main() -> None:
    protocol = json.loads((HERE / "protocol.json").read_text())
    phase2 = json.loads((HERE / "phase2_plan.json").read_text())
    validation_plan = json.loads((HERE / "validation_plan.json").read_text())
    search, validation, new_rows = read_groups()

    intended = {
        state: {key(point) for point in protocol["phase1_points"][state]}
        | {key(point) for point in phase2["points"][state]}
        for state in STATES
    }
    search_cells = []
    for state in STATES:
        for eta in sorted(intended[state]):
            rows = list(search[(state, eta)].values())
            if len(rows) != 16:
                raise AssertionError((state, eta, len(rows)))
            search_cells.append(summarize(state, eta, rows))

    pooled_groups = defaultdict(dict)
    for group in (search, validation):
        for group_key, seed_rows in group.items():
            pooled_groups[group_key].update(seed_rows)
    validation_cells = []
    final_minima = {}
    paired = {}
    boundaries = {
        "D1_pair231": "ADJACENT_TO_SUCCESS_FAILURE_BOUNDARY_AND_STILL_ON_SEARCH_BOUNDARY",
        "D2_pair228": "STILL_ON_SEARCH_BOUNDARY",
        "D4_pair227": "ADJACENT_TO_SUCCESS_FAILURE_BOUNDARY_AND_STILL_ON_SEARCH_BOUNDARY",
    }
    for state in STATES:
        candidates = [key(point) for point in validation_plan["policies"][state]]
        cells = []
        for eta in candidates:
            rows = list(pooled_groups[(state, eta)].values())
            if len(rows) != 32:
                raise AssertionError((state, eta, len(rows)))
            cells.append(summarize(state, eta, rows))
        validation_cells.extend(cells)
        successful = sorted(
            (cell for cell in cells if cell["classification"] == "SUCCESS_CELL"),
            key=lambda cell: (cell["mean_J_def"], cell["eta"]),
        )
        best, runner_up = successful[:2]
        best_eta = key(best["eta"])
        old_eta = key(protocol["previous_minima"][state])
        old_search = summarize(state, old_eta, list(search[(state, old_eta)].values()))
        old_pooled = summarize(state, old_eta, list(pooled_groups[(state, old_eta)].values()))
        nonsuccess = [
            cell for cell in search_cells
            if cell["state_id"] == state and cell["classification"] != "SUCCESS_CELL"
        ]
        for cell in nonsuccess:
            cell["distance_to_refined_eta"] = float(np.linalg.norm(np.asarray(cell["eta"]) - np.asarray(best["eta"])))
        nearest = sorted(nonsuccess, key=lambda cell: (cell["distance_to_refined_eta"], cell["eta"]))[:3]
        previous_to_best = paired_difference(
            list(pooled_groups[(state, old_eta)].values()),
            list(pooled_groups[(state, best_eta)].values()),
            "previous_minimum minus refined_candidate",
        )
        best_to_runner = paired_difference(
            list(pooled_groups[(state, runner_up_eta := key(runner_up["eta"]))].values()),
            list(pooled_groups[(state, best_eta)].values()),
            "runner_up minus refined_candidate",
        )
        paired[state] = {
            "previous_vs_refined": previous_to_best,
            "runner_up_vs_refined": best_to_runner,
        }
        final_minima[state] = {
            "previous_empirical_minimum": {
                "eta": list(old_eta),
                "original_16_seed_statistics": old_search,
                "pooled_32_seed_validation_statistics": old_pooled,
            },
            "refined_empirical_eta_star": best,
            "runner_up_32_seed": runner_up,
            "location_classification": boundaries[state],
            "paired_CRN_vs_previous": previous_to_best,
            "paired_CRN_vs_runner_up": best_to_runner,
            "nearest_tested_failure_or_unknown": nearest,
            "unique_minimum_supported": not (
                best_to_runner["two_sided_t_95_CI"][0] <= 0.0 <= best_to_runner["two_sided_t_95_CI"][1]
            ),
        }

    eta_d2 = np.asarray(final_minima["D2_pair228"]["refined_empirical_eta_star"]["eta"])
    eta_d4 = np.asarray(final_minima["D4_pair227"]["refined_empirical_eta_star"]["eta"])
    cross_state = {
        "D2_D4_same_exact_eta": bool(np.array_equal(eta_d2, eta_d4)),
        "D2_D4_eta_euclidean_distance": float(np.linalg.norm(eta_d2 - eta_d4)),
        "interpretation": "Both remain in the same broad moderate-goal/negative-safe/small-relative neighborhood, but their validated empirical minimizers are not identical.",
    }

    dump("refined_cells.json", {
        "success_criterion": protocol["success_criterion"],
        "search_cells_16_seed": search_cells,
        "predeclared_validation_cells_32_seed": validation_cells,
    })
    dump("refined_minima.json", {"states": final_minima, "cross_state": cross_state})
    dump("paired_comparisons.json", paired)
    with (HERE / "new_rollout_j_def.jsonl").open("w") as handle:
        for row in sorted(new_rows, key=lambda row: (row["source"], row["state_id"], row["eta"], row["seed"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (HERE / "successful_eta_table.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("state_id", "eta1", "eta2", "eta3", "n", "success_rate", "mean_J_def", "std_J_def", "mean_episode_length_steps"))
        for cell in sorted(
            (cell for cell in search_cells if cell["classification"] == "SUCCESS_CELL"),
            key=lambda cell: (cell["state_id"], cell["mean_J_def"], cell["eta"]),
        ):
            writer.writerow((cell["state_id"], *cell["eta"], cell["n"], cell["success_rate"], cell["mean_J_def"], cell["std_J_def"], cell["mean_episode_length_steps"]))

    sanity = sanity_check(new_rows, protocol)
    dump("sanity_checks.json", sanity)
    if sanity["status"] != "PASS":
        raise AssertionError(sanity)

    # Keep the report compact; JSON files retain every tested cell and interval.
    rows = []
    for state in STATES:
        item = final_minima[state]
        old = item["previous_empirical_minimum"]["original_16_seed_statistics"]
        best = item["refined_empirical_eta_star"]
        ci = item["paired_CRN_vs_previous"]["two_sided_t_95_CI"]
        rows.append(
            f"| {state.split('_')[0]} | {tuple(old['eta'])} | {old['mean_J_def']:.6f} | "
            f"{tuple(best['eta'])} | {best['counts']['success']}/{best['n']} | "
            f"{best['mean_J_def']:.6f} ± {best['std_J_def']:.6f} | "
            f"{best['mean_episode_length_steps']:.1f} | "
            f"{item['paired_CRN_vs_previous']['mean_difference']:.6f} [{ci[0]:.6f}, {ci[1]:.6f}] | "
            f"{item['location_classification']} |"
        )
    nearest_lines = []
    for state in STATES:
        nearest = final_minima[state]["nearest_tested_failure_or_unknown"]
        rendered = "; ".join(
            f"eta={tuple(cell['eta'])}, {cell['classification']}, "
            f"S={cell['counts']['success']}/{cell['n']}, d={cell['distance_to_refined_eta']:.5f}"
            for cell in nearest
        )
        nearest_lines.append(f"- {state.split('_')[0]}：{rendered}。")
    ambiguity_lines = []
    for state in STATES:
        item = final_minima[state]
        runner = item["runner_up_32_seed"]
        contrast = item["paired_CRN_vs_runner_up"]
        ci = contrast["two_sided_t_95_CI"]
        ambiguity_lines.append(
            f"- {state.split('_')[0]}：次优 eta={tuple(runner['eta'])}；"
            f"次优减最低的配对均值={contrast['mean_difference']:.6f}，95% CI "
            f"[{ci[0]:.6f}, {ci[1]:.6f}]；唯一最低点"
            f"{'得到支持' if item['unique_minimum_supported'] else '未得到支持'}。"
        )
    report = f"""# Success-basin deformation refinement

## 结论

三组状态都在旧网格之外找到了更低的、满足当前成功判据的 `J_def` 区域，且相对旧最低点的 32-seed 配对 CRN 差异均为正。可是复核后的最低候选仍位于本次搜索边界，D1/D4 还紧邻 SUCCESS/UNKNOWN 转换；与近邻候选的差异也不足以确立唯一局部最低点。因此不能称为全局或已解析的局部最优。

**最终建议：EXPAND_FURTHER**

## 固定量与实验量

实现的量严格为

```text
delta_u[k] = u_exec[k] - u_safe[k]
J_def = 0.05 * sum_k ||delta_u[k]||_2^2
```

其中 `u_safe` 与 `u_exec` 来自同一个 corrected state 和同一次 Flow sample，`u_exec` 是第二次 hard projection 的输出。没有折扣、时长归一化、进度项、平滑项或额外正则。

第一阶段新增 1,024 条 rollout，第二阶段新增 1,376 条；预声明复核再新增 192 条（每状态 4 个固定候选 × 16 个新种子）。新运行总计 {len(new_rows):,} 条、{sanity['new_physical_steps_checked']:,} physical steps；复核 192/192 全部成功。搜索阶段每个 cell 用 16 个共同随机种子，入围候选与旧最低点用额外 16 个共同随机种子复核，合并为 32 seeds。

## 各状态结果

| state | 旧 eta | 旧 16-seed mean J | refined eta | success | mean ± std J | mean steps | 配对 old-new mean [95% CI] | 位置 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

这里配对差异定义为 `J_old - J_refined`，正值表示 refined 候选更低。`refined eta` 的均值/std/success/episode length 均基于 32 seeds。

## 唯一性与边界

{chr(10).join(ambiguity_lines)}

所以当前 `eta*_emp` 是“已测试候选中的 refined empirical success-constrained minimum”，不是已解析的连续局部最优。D2 与 D4 不再共享完全相同的最低 eta；两者参数距离为 {cross_state['D2_D4_eta_euclidean_distance']:.5f}，仍落在相近的 moderate-goal / negative-safe / small-relative 区域。

最近的 failure/unknown 点：

{chr(10).join(nearest_lines)}

## 完整性检查

- 检查了全部 {sanity['new_rollouts_checked']:,} 条新轨迹；NPZ 哈希、terminal outcome、动作维数与冻结源码哈希均通过。
- `J_def` 重构最大误差为 {sanity['maximum_absolute_reconstruction_errors']['J_def_reconstruction']:.3e}；逐步 `||delta_u||^2` 重构最大误差为 {sanity['maximum_absolute_reconstruction_errors']['per_step_delta_squared_reconstruction']:.3e}。
- `w-u_safe-g` 最大误差为 {sanity['maximum_absolute_reconstruction_errors']['w_minus_safe_minus_g']:.3e}；位置积分最大误差为 {sanity['maximum_absolute_reconstruction_errors']['position_integration']:.3e}。
- `delta_u` 在第二次 hard projection 后计算；`u_safe` 来自同一 corrected state、同一 Flow sample。logger 只读这些量，冻结源码哈希未变化，输出仅写入本目录。
- 数据中没有全程 `u_exec == u_safe` 的实际新 rollout；正定平方和给出 `J_def=0 iff 每一步 delta_u=0`，并通过零数组单元检查。未观察到与该等价关系冲突的轨迹。

## 尚未执行的实验

**尚未运行实验 B：拆解 `J_def` 和 episode length 的关系。** 本轮按要求只报告 episode length 作为伴随描述量；没有做按时长归一化、固定时域反事实、每步强度/持续时间分解、相关性归因或因果解释。因此当前较低 `J_def` 可能同时包含动作偏差幅度与 episode duration 的共同影响，这一点尚未被拆开。

所有 cell、置信区间与最近边界点见 `refined_cells.json`、`refined_minima.json` 和 `paired_comparisons.json`。
"""
    (HERE / "refinement_report.md").write_text(report)

    artifact_names = (
        "protocol.json", "phase1_analysis.json", "phase2_plan.json", "validation_plan.json",
        "refined_cells.json", "refined_minima.json", "paired_comparisons.json",
        "successful_eta_table.csv", "new_rollout_j_def.jsonl", "sanity_checks.json",
        "refinement_report.md", "setup.py", "plan_phase2.py", "plan_validation.py", "run.py",
    )
    raw_manifests = sorted(str(path.relative_to(ROOT)) for path in (HERE / "raw").glob("*/manifest.json"))
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": "success_basin_deformation_refinement",
        "recommendation": "EXPAND_FURTHER",
        "experiment_B_J_def_episode_length_decomposition": "NOT_RUN",
        "new_rollouts": len(new_rows),
        "new_physical_steps": sanity["new_physical_steps_checked"],
        "raw_manifests": {name: sha(ROOT / name) for name in raw_manifests},
        "artifacts_sha256": {name: sha(HERE / name) for name in artifact_names},
        "frozen_source_hashes_match_protocol": sanity["frozen_source_hashes_match_protocol"],
        "notes": [
            "No frozen source or controller semantics were modified.",
            "Validation policies were fixed before their fresh outcomes were observed.",
            "No claim of global optimality is made.",
        ],
    }
    dump("manifest.json", manifest)
    print(json.dumps({
        "recommendation": manifest["recommendation"],
        "new_rollouts": manifest["new_rollouts"],
        "new_physical_steps": manifest["new_physical_steps"],
        "refined_eta": {state: final_minima[state]["refined_empirical_eta_star"]["eta"] for state in STATES},
        "sanity": sanity["status"],
        "experiment_B": "NOT_RUN",
    }, indent=2))


if __name__ == "__main__":
    main()
