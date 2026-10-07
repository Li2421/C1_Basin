"""Analyze predeclared true-D_H chunk outcomes without surrogate scores."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta, fisher_exact


HERE = Path(__file__).resolve().parent


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def clopper_pearson(successes, trials, alpha=.05):
    lo = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes,
                                                    trials - successes + 1))
    hi = 1.0 if successes == trials else float(beta.ppf(
        1 - alpha / 2, successes + 1, trials - successes))
    return [lo, hi]


def holm_adjust(rows):
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["p_raw"])
    running = 0.0
    count = len(rows)
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * row["p_raw"])
        running = max(running, adjusted)
        rows[index]["p_holm"] = running
    return rows


def arm_projection_summary(row):
    live = [outcome["chunk"] for outcome in row["outcomes"]]
    def values(key):
        return [item[key] for item in live if item[key] is not None]
    removal = values("mean_removed_fraction")
    executed = values("mean_executed_correction_norm")
    return dict(
        mean_live_chunk_steps=float(np.mean(values("live_chunk_steps"))),
        mean_live_requested_energy=float(np.mean(values(
            "live_requested_energy"))),
        mean_removed_fraction=float(np.mean(removal)) if removal else None,
        median_removed_fraction=float(np.median(removal)) if removal else None,
        mean_executed_correction_norm=(float(np.mean(executed))
                                       if executed else None),
        mean_cumulative_executed_correction_norm=float(np.mean(values(
            "cumulative_executed_correction_norm"))),
        mean_executed_correction_energy=float(np.mean(values(
            "executed_correction_energy"))),
        mean_active_set_switches=float(np.mean(values("active_set_switches"))),
        max_frozen_G_phi_output_norm=float(max(values(
            "max_frozen_G_phi_output_norm"), default=0.0)),
    )


def planned_contrasts(rows, material_effect):
    by_id = {row["arm_id"]: row for row in rows}
    baseline = by_id["baseline"]
    pairs = []
    seen = set()
    for row in rows:
        if row["arm_id"] == "baseline":
            continue
        key = tuple(sorted(("baseline", row["arm_id"])))
        seen.add(key)
        pairs.append((baseline, row, "baseline_vs_arm"))
    grouped = defaultdict(dict)
    for row in rows:
        if row["axis"] is not None:
            grouped[(round(row["alpha_norm"], 12), row["axis"])][row["sign"]] = row
    for group in grouped.values():
        if -1 in group and 1 in group:
            a, b = group[-1], group[1]
            key = tuple(sorted((a["arm_id"], b["arm_id"])))
            if key not in seen:
                pairs.append((a, b, "minus_vs_plus"))
                seen.add(key)
    tests = []
    for a, b, kind in pairs:
        table = [[a["deadlocks"], a["continuations"] - a["deadlocks"]],
                 [b["deadlocks"], b["continuations"] - b["deadlocks"]]]
        p = float(fisher_exact(table, alternative="two-sided").pvalue)
        effect = float(b["Q_D"] - a["Q_D"])
        tests.append(dict(
            arm_a=a["arm_id"], arm_b=b["arm_id"], contrast=kind,
            Q_a=a["Q_D"], Q_b=b["Q_D"], delta_Q_b_minus_a=effect,
            abs_effect=abs(effect), p_raw=p,
        ))
    holm_adjust(tests)
    for test in tests:
        test["qualifies"] = bool(
            test["abs_effect"] >= material_effect and test["p_holm"] <= .05)
    return tests


def cell_category(rows, tests, protocol):
    statistics = protocol["statistics"]
    baseline = next(row for row in rows if row["arm_id"] == "baseline")
    minimum = min(row["Q_D"] for row in rows)
    qualifying = [test for test in tests if test["qualifies"]]
    recovered = (baseline["Q_D"] >= .75 and minimum <= .25 and
                 any(test["qualifies"] for test in tests))
    if recovered:
        return "RECOVERED"
    if qualifying:
        return "CONTROL_DISCRIMINATIVE"
    if minimum >= .875:
        return "SATURATED"
    return "FLAT_INTERMEDIATE"


def budget_order(protocol):
    return {item["name"]: index for index, item in enumerate(
        protocol["pilot"]["budget_families"])}


def select_replication(cell_rows, tests, protocol):
    """Predeclared budget, then baseline plus best/worst coordinate arms."""
    qualifying = [test for test in tests if test["qualifies"]]
    if not qualifying:
        return None
    by_id = {row["arm_id"]: row for row in cell_rows}
    order = budget_order(protocol)
    candidates = []
    for test in qualifying:
        families = set(by_id[test["arm_a"]]["budget_families"]) | set(
            by_id[test["arm_b"]]["budget_families"])
        families.discard("baseline")
        family_rank = min((order[name] for name in families), default=999)
        candidates.append((family_rank, -test["abs_effect"],
                           test["arm_a"], test["arm_b"], test))
    selected = min(candidates)[-1]
    selected_nonzero = next(
        by_id[arm_id] for arm_id in (selected["arm_a"], selected["arm_b"])
        if arm_id != "baseline")
    norm = selected_nonzero["alpha_norm"]
    same_budget = [row for row in cell_rows
                   if row["arm_id"] != "baseline" and
                   np.isclose(row["alpha_norm"], norm)]
    best = min(same_budget, key=lambda row: (row["Q_D"], row["arm_id"]))
    worst = max(same_budget, key=lambda row: (row["Q_D"], row["arm_id"]))
    arms = ["baseline", best["arm_id"], worst["arm_id"]]
    return dict(selected_contrast=selected, selected_alpha_norm=norm,
                best_arm=best["arm_id"], worst_arm=worst["arm_id"],
                arm_ids=sorted(set(arms)))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path,
                        default=HERE / "predeclared_protocol.json")
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    records = []
    for path in args.records:
        complete = path.parent / "complete.json"
        if not complete.exists() or not json.loads(complete.read_text())["complete"]:
            raise RuntimeError(f"incomplete input: {path}")
        records.extend(json.loads(path.read_text()))
    scenario_ids = [row["scenario_id"] for row in records]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise RuntimeError("duplicate scenario IDs")
    material = float(protocol["statistics"]["material_effect"])
    arms_table = []
    for row in records:
        projection = arm_projection_summary(row)
        arms_table.append(dict(
            scenario_id=row["scenario_id"], stage=row["stage"],
            state_id=row["state"]["state_id"],
            rid=row["state"]["rid"], stratum=row["state"]["stratum"],
            action_step=row["state"]["action_step"],
            time_before_reference_event_seconds=(
                row["state"]["reference_event_action"] -
                row["state"]["action_step"]) * .05,
            horizon_steps=row["horizon_steps"],
            horizon_seconds=row["horizon_seconds"], arm_id=row["arm_id"],
            axis=row["axis"], sign=row["sign"],
            alpha_norm=row["alpha_norm"],
            budget_families="|".join(row["budget_families"]),
        planned_total_energy=row["planned_total_energy"],
        deadlocks=row["deadlocks"], continuations=row["continuations"],
        successes=sum(outcome["success"] for outcome in row["outcomes"]),
        collisions=sum(outcome["collision"] for outcome in row["outcomes"]),
        timeouts=sum(outcome["timeout"] for outcome in row["outcomes"]),
        max_reference_state_error=max(
            outcome["reference_state_max_abs_error"]
            for outcome in row["outcomes"]),
        Q_D=row["Q_D"],
            Q_D_ci95=clopper_pearson(row["deadlocks"], row["continuations"]),
            **projection,
        ))
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["state"]["state_id"], row["horizon_steps"])].append(row)
    cells, contrasts = [], []
    cell_tests = {}
    for (state_id, horizon), rows in sorted(grouped.items()):
        tests = planned_contrasts(rows, material)
        cell_tests[(state_id, horizon)] = tests
        contrasts.extend(dict(state_id=state_id, horizon_steps=horizon, **test)
                         for test in tests)
        category = cell_category(rows, tests, protocol)
        best = min(rows, key=lambda row: (row["Q_D"], row["arm_id"]))
        worst = max(rows, key=lambda row: (row["Q_D"], row["arm_id"]))
        removals = [arm_projection_summary(row)["median_removed_fraction"]
                    for row in rows if row["alpha_norm"] > 0]
        state = rows[0]["state"]
        cells.append(dict(
            state_id=state_id, rid=state["rid"], stratum=state["stratum"],
            action_step=state["action_step"],
            time_before_reference_event_seconds=(
                state["reference_event_action"] - state["action_step"]) * .05,
            horizon_steps=horizon, horizon_seconds=horizon * .05,
            category=category, best_arm=best["arm_id"], best_Q_D=best["Q_D"],
            worst_arm=worst["arm_id"], worst_Q_D=worst["Q_D"],
            Q_D_range=worst["Q_D"] - best["Q_D"],
            qualifying_contrasts=sum(test["qualifies"] for test in tests),
            median_projection_removal=(float(np.median(removals))
                                       if removals else 0.0),
        ))
    qualifying_cells = [cell for cell in cells
                        if cell["category"] in
                        {"CONTROL_DISCRIMINATIVE", "RECOVERED"}]
    budget_cells = []
    family_names = [item["name"] for item in protocol["pilot"]["budget_families"]]
    for (state_id, horizon), rows in sorted(grouped.items()):
        baseline = next(row for row in rows if row["arm_id"] == "baseline")
        for family in family_names:
            selected = [row for row in rows
                        if row["arm_id"] == "baseline" or
                        family in row["budget_families"]]
            if len(selected) == 1:
                continue
            best = min(selected, key=lambda row: (row["Q_D"], row["arm_id"]))
            worst = max(selected, key=lambda row: (row["Q_D"], row["arm_id"]))
            budget_cells.append(dict(
                state_id=state_id, stratum=baseline["state"]["stratum"],
                horizon_steps=horizon, horizon_seconds=horizon * .05,
                budget_family=family,
                alpha_norm=next(row["alpha_norm"] for row in selected
                                if row["arm_id"] != "baseline"),
                requested_energy=next(row["planned_total_energy"]
                                      for row in selected
                                      if row["arm_id"] != "baseline"),
                baseline_Q_D=baseline["Q_D"], best_arm=best["arm_id"],
                best_Q_D=best["Q_D"], worst_arm=worst["arm_id"],
                worst_Q_D=worst["Q_D"], Q_D_range=worst["Q_D"]-best["Q_D"],
            ))
    earliest = min((cell["horizon_steps"] for cell in qualifying_cells),
                   default=None)
    replication = None
    if qualifying_cells:
        stage_order = {"early": 0, "emerging": 1, "late": 2}
        selected_cell = min(
            qualifying_cells,
            key=lambda cell: (cell["horizon_steps"],
                              stage_order[cell["stratum"]], cell["state_id"]))
        key = (selected_cell["state_id"], selected_cell["horizon_steps"])
        replication = dict(selected_cell=selected_cell,
                           **select_replication(grouped[key], cell_tests[key], protocol))
    all_saturated = bool(cells) and all(cell["category"] == "SATURATED"
                                        for cell in cells)
    nonzero_removal = [row["median_removed_fraction"] for row in arms_table
                       if row["alpha_norm"] > 0]
    projection_mostly_removed = bool(nonzero_removal) and float(
        np.median(nonzero_removal)) >= .8
    gate = dict(
        pilot_discrimination=bool(qualifying_cells),
        earliest_qualifying_horizon_steps=earliest,
        all_cells_saturated=all_saturated,
        median_projection_removal=float(np.median(nonzero_removal)),
        projection_mostly_removed=projection_mostly_removed,
        replication=replication,
        gradient_allowed=False,
        reserve_expansion_recommended=bool(
            qualifying_cells or (all_saturated and not projection_mostly_removed)),
        L5_refinement_recommended=earliest in {10, 20},
        L40_extension_recommended=bool(
            earliest is None and not projection_mostly_removed),
    )
    result = dict(
        schema="rrisk_true_chunk_analysis_v1",
        arm_scenarios=len(records),
        continuations=sum(row["continuations"] for row in records),
        state_ids=sorted({row["state"]["state_id"] for row in records}),
        horizons_steps=sorted({row["horizon_steps"] for row in records}),
        cells=cells, budget_cells=budget_cells,
        planned_contrasts=contrasts, gate=gate,
        note="No surrogate score enters arm construction, testing, or gates.",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / "analysis.json", result)
    atomic_json(args.out / "arm_summary.json", arms_table)
    write_csv(args.out / "tables" / "arm_summary.csv", [
        {**row, "Q_D_ci95": json.dumps(row["Q_D_ci95"])} for row in arms_table])
    write_csv(args.out / "tables" / "cell_summary.csv", cells)
    write_csv(args.out / "tables" / "budget_cell_summary.csv", budget_cells)
    write_csv(args.out / "tables" / "planned_contrasts.csv", contrasts)
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
