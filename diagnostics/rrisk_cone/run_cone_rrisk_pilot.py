"""Projection-cone diagnostic on the frozen old TEST-P/TEST-Q evidence.

This script performs no rollout, training, policy update, environment change,
or risk tuning.  It reads the immutable traces and continuation outcomes from
the preceding R_risk audit, reconstructs the two projection inputs, and writes
new evidence only under diagnostics/rrisk_cone/.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import beta, kendalltau, spearmanr

from diagnostics.rrisk_cone.cone_geometry import (
    projection_kkt_audit, self_test, signed_cone_margin,
)
from single_integrator.c1.differentiable_rollout import (
    ResidualFlowField, bounded_nominal,
)
from single_integrator.c1.train import observation
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.evaluate import load_policy


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n")


def save_csv(path: Path, rows, fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{key: clean(row.get(key)) for key in fields}
                          for row in rows])


def one_noise(seed: int, rid: int, step: int):
    key = jax.random.fold_in(jax.random.PRNGKey(seed), rid)
    return jax.random.normal(jax.random.fold_in(key, step), (4,),
                             dtype=jnp.float32)


class NominalReconstructor:
    def __init__(self, checkpoint: Path):
        jax.config.update("jax_enable_x64", True)
        baseline, provenance = load_policy(checkpoint)
        self.field = ResidualFlowField(baseline, None)
        self.sample = jax.jit(self.field.baseline_sample)
        self.plant = Config(**provenance["evaluation_environment"])
        self.goals = jnp.asarray(GiveWayEnv(self.plant).goals)

    def __call__(self, before, previous_applied, seed, rid, step):
        positions = jnp.asarray(before, jnp.float64)[None]
        velocity = jnp.asarray(previous_applied, jnp.float64).reshape(1, 2, 2)
        obs = observation(positions, velocity, self.goals)
        raw = self.sample(obs, one_noise(seed, rid, step)[None])
        return np.asarray(bounded_nominal(raw, self.plant.max_speed)[0])


def score_projection(nominal, projected, A, b, tolerance, protocol):
    kkt = projection_kkt_audit(
        nominal, projected, A, b, speed=0.5,
        active_tolerance=tolerance,
    )
    indices = kkt["active_linear"]
    generators = -np.asarray(A)[indices]
    cone = signed_cone_margin(
        nominal, generators,
        membership_tolerance=protocol["tolerances"]["cone_membership"],
        facet_tolerance=protocol["tolerances"]["facet"],
        zero_tolerance=protocol["tolerances"]["generator_zero"],
    )
    cone["speed_confounded"] = bool(kkt["active_speed"])
    return dict(kkt=kkt, cone=cone)


def numeric_risk(stage):
    value = stage["cone"]["risk"]
    if value is not None:
        return float(value)
    if stage["cone"].get("extended_risk") == "+inf":
        return np.inf
    return np.nan


def auc(y, score):
    y = np.asarray(y, bool)
    score = np.asarray(score, float)
    if not y.any() or y.all() or not np.isfinite(score).all():
        return None
    # Pairwise definition gives half credit to ties.
    positive, negative = score[y], score[~y]
    return float(np.mean((positive[:, None] > negative[None, :]) +
                         .5 * (positive[:, None] == negative[None, :])))


def average_precision(y, score):
    y = np.asarray(y, bool)
    score = np.asarray(score, float)
    if not y.any() or not np.isfinite(score).all():
        return None
    thresholds = np.unique(score)[::-1]
    tp = fp = 0
    previous_recall = 0.0
    result = 0.0
    positives = int(y.sum())
    for threshold in thresholds:
        group = score == threshold
        tp += int(np.count_nonzero(y & group))
        fp += int(np.count_nonzero(~y & group))
        recall = tp / positives
        result += (recall - previous_recall) * tp / (tp + fp)
        previous_recall = recall
    return float(result)


def bootstrap_auc(rows, replicates, seed):
    generator = np.random.default_rng(seed)
    values = []
    rows = list(rows)
    for _ in range(replicates):
        sample = [rows[index] for index in generator.integers(0, len(rows), len(rows))]
        value = auc([row["deadlock"] for row in sample],
                    [row["R_cone_pi2"] for row in sample])
        if value is not None:
            values.append(value)
    if not values:
        return None
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def distribution(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return dict(n=int(len(values)), mean=float(np.mean(values)),
                std=float(np.std(values)), min=float(np.min(values)),
                q25=float(np.quantile(values, .25)),
                median=float(np.median(values)),
                q75=float(np.quantile(values, .75)), max=float(np.max(values)))


def safe_spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (not np.isfinite(x).all() or not np.isfinite(y).all() or
            np.ptp(x) == 0 or np.ptp(y) == 0):
        return None
    return float(spearmanr(x, y).statistic)


def clopper_pearson(successes, total, alpha=.05):
    lower = 0.0 if successes == 0 else float(beta.ppf(
        alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(
        1 - alpha / 2, successes + 1, total - successes))
    return [lower, upper]


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total ** 2)) / denominator
    return [float(center - radius), float(center + radius)]


def previous_velocity(trace, step):
    return (np.zeros(4) if step == 0
            else np.asarray(trace["applied"][step - 1]))


def score_state(trace, step, nominal, protocol, active_tolerance=None):
    tolerance = (protocol["tolerances"]["active_primary"]
                 if active_tolerance is None else active_tolerance)
    A, b = np.asarray(trace["A"][step]), np.asarray(trace["b"][step])
    safe = np.asarray(trace["safe"][step])
    candidate = np.asarray(trace["w"][step])
    executed = np.asarray(trace["applied"][step])
    pi1 = score_projection(nominal, safe, A, b, tolerance, protocol)
    pi2 = score_projection(candidate, executed, A, b, tolerance, protocol)
    return pi1, pi2


def public_stage(stage):
    """Retain all audit numbers but keep generator arrays in the record only once."""
    return stage


def load_trace(old_root, record):
    path = old_root / record["risk"][0]["trace"]
    return path, np.load(path, allow_pickle=False)


def p_rows(old_root, records, reconstructor, protocol):
    rows = []
    primary_tolerance = protocol["tolerances"]["active_primary"]
    for record in records:
        step, rid = int(record["early_step"]), int(record["rid"])
        risk_record = record["risk"][0]
        path, trace = load_trace(old_root, record)
        with trace:
            previous = previous_velocity(trace, step)
            nominal = reconstructor(trace["before"][step], previous,
                                      int(risk_record["seed"]), rid, step)
            pi1, pi2 = score_state(trace, step, nominal, protocol)
            safe = np.asarray(trace["safe"][step])
            candidate = np.asarray(trace["w"][step])
            executed = np.asarray(trace["applied"][step])
            before = np.asarray(trace["before"][step])
            after = np.asarray(trace["after"][step])
            goals = np.asarray(reconstructor.goals)
            direction = goals - before
            direction /= np.maximum(np.linalg.norm(direction, axis=1,
                                                    keepdims=True), 1e-15)
            progress_speed = float(np.sum(direction * executed.reshape(2, 2)))
            actual_progress_speed = float(
                (np.linalg.norm(goals - before, axis=1).sum() -
                 np.linalg.norm(goals - after, axis=1).sum()) /
                reconstructor.plant.dt)
            outcome = record["outcomes"][0]
            sensitivity = {}
            for tolerance in protocol["tolerances"]["active_sensitivity"]:
                _, stage = score_state(trace, step, nominal, protocol, tolerance)
                sensitivity[f"{tolerance:.0e}"] = dict(
                    risk=numeric_risk(stage), status=stage["cone"]["status"],
                    active_linear=stage["kkt"]["active_linear"],
                    rank=stage["cone"]["rank"])
            row = dict(
                state_id=f"P:rid{rid}:step{step}", rid=rid, early_step=step,
                trace=str(path), trace_sha256=sha256(path),
                risk_seed=int(risk_record["seed"]),
                outcome_seed=int(outcome["seed"]),
                deadlock=bool(outcome["primary_deadlock"]),
                success=bool(outcome["success"]),
                first_deadlock_step=int(outcome["first_deadlock_step"]),
                lead_seconds=((int(outcome["first_deadlock_step"]) - step) *
                              reconstructor.plant.dt
                              if outcome["primary_deadlock"] else None),
                old_direct_R_CERT=float(risk_record["direct_R_CERT"]),
                old_augmented_R_CERT=float(risk_record["augmented_R_CERT"]),
                nominal=nominal.tolist(), safe=safe.tolist(),
                candidate=candidate.tolist(), executed=executed.tolist(),
                nominal_reconstruction_error=float(np.max(np.abs(
                    # Pi_1 output is stored safe; the KKT residual checks the
                    # reconstruction without invoking a new projection.
                    np.asarray(pi1["kkt"]["projection_residual"]) -
                    (nominal - safe)))),
                safety_intervention_norm=float(np.linalg.norm(nominal - safe)),
                safe_norm=float(np.linalg.norm(safe)),
                executed_norm=float(np.linalg.norm(executed)),
                progress_speed=progress_speed,
                actual_one_step_progress_speed=actual_progress_speed,
                pairwise_center_distance=float(np.linalg.norm(before[0] - before[1])),
                pairwise_surface_clearance=float(
                    np.linalg.norm(before[0] - before[1]) -
                    2 * reconstructor.plant.agent_radius),
                active_safety_count_pi1=len(pi1["kkt"]["active_linear"]),
                active_safety_count_pi2=len(pi2["kkt"]["active_linear"]),
                R_cone_pi1=numeric_risk(pi1), R_cone_pi2=numeric_risk(pi2),
                pi1=public_stage(pi1), pi2=public_stage(pi2),
                active_tolerance=primary_tolerance,
                active_tolerance_sensitivity=sensitivity,
            )
            rows.append(row)
    return rows


def q_rows(old_root, records, reconstructor, protocol, prefix_seed):
    rows = []
    nominal_cache = {}
    for record in records:
        rid, arm, step = int(record["rid"]), record["arm"], 100
        risk_record = record["risk"][0]
        path, trace = load_trace(old_root, record)
        with trace:
            key = rid
            if key not in nominal_cache:
                nominal_cache[key] = reconstructor(
                    trace["before"][step], previous_velocity(trace, step),
                    prefix_seed, rid, step)
            nominal = nominal_cache[key]
            pi1, pi2 = score_state(trace, step, nominal, protocol)
            before = np.asarray(trace["before"][step])
            safe = np.asarray(trace["safe"][step])
            candidate = np.asarray(trace["w"][step])
            executed = np.asarray(trace["applied"][step])
            outcomes = [bool(item["primary_deadlock"])
                        for item in record["outcomes"]]
            successes = int(sum(outcomes))
            sensitivity = {}
            for tolerance in protocol["tolerances"]["active_sensitivity"]:
                _, stage = score_state(trace, step, nominal, protocol, tolerance)
                sensitivity[f"{tolerance:.0e}"] = dict(
                    risk=numeric_risk(stage), status=stage["cone"]["status"],
                    active_linear=stage["kkt"]["active_linear"],
                    rank=stage["cone"]["rank"])
            rows.append(dict(
                state_action_id=f"Q:rid{rid}:{arm}", rid=rid, arm=arm,
                trace=str(path), trace_sha256=sha256(path),
                current_action_seed=prefix_seed,
                risk_seed=int(risk_record["seed"]),
                continuation_seeds=[int(item["seed"])
                                    for item in record["outcomes"]],
                deadlock_samples=[int(value) for value in outcomes],
                deadlocks=successes, continuations=len(outcomes),
                Q_D=successes / len(outcomes),
                Q_D_clopper_pearson95=clopper_pearson(successes, len(outcomes)),
                old_direct_R_CERT=float(risk_record["direct_R_CERT"]),
                old_augmented_R_CERT=float(risk_record["augmented_R_CERT"]),
                delta=record["delta"], nominal=nominal.tolist(),
                safe=safe.tolist(), candidate=candidate.tolist(),
                executed=executed.tolist(), position=before.tolist(),
                safety_intervention_norm=float(np.linalg.norm(nominal - safe)),
                pi2_projection_norm=float(np.linalg.norm(candidate - executed)),
                safe_norm=float(np.linalg.norm(safe)),
                executed_norm=float(np.linalg.norm(executed)),
                pairwise_center_distance=float(np.linalg.norm(before[0] - before[1])),
                active_safety_count_pi1=len(pi1["kkt"]["active_linear"]),
                active_safety_count_pi2=len(pi2["kkt"]["active_linear"]),
                R_cone_pi1=numeric_risk(pi1), R_cone_pi2=numeric_risk(pi2),
                pi1=public_stage(pi1), pi2=public_stage(pi2),
                active_tolerance_sensitivity=sensitivity,
            ))
    return rows


def prediction_analysis(rows, protocol):
    result = {"by_early_step": {}}
    passing = 0
    for step in sorted({row["early_step"] for row in rows}):
        subset = [row for row in rows if row["early_step"] == step]
        deadlock = [row for row in subset if row["deadlock"]]
        nondead = [row for row in subset if not row["deadlock"]]
        value = auc([row["deadlock"] for row in subset],
                    [row["R_cone_pi2"] for row in subset])
        if value is not None and value >= .75:
            passing += 1
        result["by_early_step"][str(step)] = dict(
            n=len(subset), deadlocks=len(deadlock), non_deadlocks=len(nondead),
            roc_auc=value,
            roc_auc_bootstrap_ci95=bootstrap_auc(
                subset, protocol["statistics"]["bootstrap_replicates"],
                protocol["statistics"]["bootstrap_seed"] + step),
            average_precision=average_precision(
                [row["deadlock"] for row in subset],
                [row["R_cone_pi2"] for row in subset]),
            deadlock_R=distribution([row["R_cone_pi2"] for row in deadlock]),
            non_deadlock_R=distribution([row["R_cone_pi2"] for row in nondead]),
            median_difference=(float(np.median([row["R_cone_pi2"] for row in deadlock]) -
                                     np.median([row["R_cone_pi2"] for row in nondead]))
                               if deadlock and nondead else None),
            rank_biserial=(2 * value - 1 if value is not None else None),
            lead_seconds=distribution([row["lead_seconds"] for row in deadlock]),
        )
    result["verdict"] = "PREDICTIVE" if passing >= 3 else (
        "WEAKLY_PREDICTIVE" if passing >= 2 else "NON_PREDICTIVE")
    result["passing_early_steps"] = passing
    result["representative_states"] = dict(
        highest_risk=max(rows, key=lambda row: row["R_cone_pi2"])["state_id"],
        lowest_risk=min(rows, key=lambda row: row["R_cone_pi2"])["state_id"],
        high_risk_non_deadlock=(max(
            (row for row in rows if not row["deadlock"]),
            key=lambda row: row["R_cone_pi2"])["state_id"]),
        low_risk_deadlock=(min(
            (row for row in rows if row["deadlock"]),
            key=lambda row: row["R_cone_pi2"])["state_id"]),
    )
    return result


def qualification_analysis(rows):
    proxies = [
        "active_safety_count_pi2", "safety_intervention_norm", "safe_norm",
        "progress_speed", "pairwise_center_distance", "old_direct_R_CERT",
        "old_augmented_R_CERT",
    ]
    correlations = []
    risk = np.asarray([row["R_cone_pi2"] for row in rows])
    for proxy in proxies:
        values = np.asarray([row[proxy] for row in rows])
        correlations.append(dict(proxy=proxy,
                                 spearman=safe_spearman(risk, values)))
    # Numerical residual after an ordinary least-squares proxy model.  This is
    # a redundancy diagnostic, not a fitted predictor or a new score.
    base = proxies[:5]
    columns = []
    retained = []
    for proxy in base:
        values = np.asarray([row[proxy] for row in rows], float)
        if np.std(values) > 0:
            columns.append((values - values.mean()) / values.std())
            retained.append(proxy)
    design = np.column_stack([np.ones(len(rows)), *columns])
    coefficients = np.linalg.lstsq(design, risk, rcond=None)[0]
    fitted = design @ coefficients
    residual = risk - fitted
    total = np.sum((risk - risk.mean()) ** 2)
    r2 = 1 - np.sum(residual ** 2) / total if total > 0 else None
    by_count = {}
    for count in sorted({row["active_safety_count_pi2"] for row in rows}):
        values = [row["R_cone_pi2"] for row in rows
                  if row["active_safety_count_pi2"] == count]
        by_count[str(count)] = distribution(values)
    return dict(
        correlations=correlations,
        proxy_linear_redundancy=dict(proxies=retained, r_squared=r2,
                                     residual=distribution(residual)),
        R_cone_by_active_count=by_count,
        interpretation=(
            "Residual spread only establishes numerical information beyond the "
            "listed instantaneous proxies; it is not causal control evidence."),
    )


def compare_scores(a, b, tolerance):
    if np.isposinf(a) and np.isposinf(b):
        return 0
    if np.isposinf(a):
        return 1
    if np.isposinf(b):
        return -1
    delta = a - b
    if abs(delta) <= tolerance:
        return 0
    return 1 if delta > 0 else -1


def action_pair_analysis(rows, protocol, tolerance_key="R_cone_pi2"):
    score_tolerance = protocol["tolerances"]["score_difference"]
    q_tolerance = protocol["tolerances"]["true_risk_difference"]
    alias_absolute = protocol["tolerances"]["projection_alias_absolute"]
    alias_relative = protocol["tolerances"]["projection_alias_relative"]
    pairs = []
    for rid in sorted({row["rid"] for row in rows}):
        state = [row for row in rows if row["rid"] == rid]
        for left, right in combinations(state, 2):
            score_i, score_j = left[tolerance_key], right[tolerance_key]
            ordering = compare_scores(score_i, score_j, score_tolerance)
            q_delta = left["Q_D"] - right["Q_D"]
            q_ordering = (0 if abs(q_delta) <= q_tolerance
                          else (1 if q_delta > 0 else -1))
            pre = float(np.linalg.norm(np.asarray(left["candidate"]) -
                                       np.asarray(right["candidate"])))
            post = float(np.linalg.norm(np.asarray(left["executed"]) -
                                        np.asarray(right["executed"])))
            scale = max(np.linalg.norm(left["executed"]),
                        np.linalg.norm(right["executed"]), 1.0)
            alias = post <= alias_absolute + alias_relative * scale
            active_i = tuple(left["pi2"]["kkt"]["active_linear"])
            active_j = tuple(right["pi2"]["kkt"]["active_linear"])
            if ordering == 0:
                category = "score_tie"
            elif q_ordering == 0:
                category = "score_changes_Q_unchanged"
            elif ordering == q_ordering:
                category = "concordant"
            else:
                category = "discordant"
            delta_r = (None if not np.isfinite(score_i - score_j)
                       else float(score_i - score_j))
            pairs.append(dict(
                pair_id=f"rid{rid}:{left['arm']}__{right['arm']}",
                rid=rid, arm_i=left["arm"], arm_j=right["arm"],
                R_i=score_i, R_j=score_j, delta_R=delta_r,
                Q_i=left["Q_D"], Q_j=right["Q_D"], delta_Q=q_delta,
                score_ordering=ordering, true_risk_ordering=q_ordering,
                score_different=ordering != 0,
                true_risk_different=q_ordering != 0,
                concordant=bool(ordering != 0 and q_ordering != 0 and
                                ordering == q_ordering),
                discordant=bool(ordering != 0 and q_ordering != 0 and
                                ordering != q_ordering),
                category=category,
                pre_projection_difference=pre,
                executed_action_difference=post,
                attenuation=(post / pre if pre > 0 else None),
                projection_alias=bool(alias),
                active_set_i=list(active_i), active_set_j=list(active_j),
                active_set_switch=active_i != active_j,
                cone_rank_i=left["pi2"]["cone"]["rank"],
                cone_rank_j=right["pi2"]["cone"]["rank"],
                cone_status_i=left["pi2"]["cone"]["status"],
                cone_status_j=right["pi2"]["cone"]["status"],
                cone_changes=(active_i != active_j),
                paired_outcome_differences=[a - b for a, b in zip(
                    left["deadlock_samples"], right["deadlock_samples"])],
            ))
    score_pairs = [pair for pair in pairs if pair["score_different"]]
    informative = [pair for pair in score_pairs if pair["true_risk_different"]]
    concordant = sum(pair["concordant"] for pair in informative)
    discordant = sum(pair["discordant"] for pair in informative)
    unchanged_or_worse = sum(
        pair["true_risk_ordering"] == 0 or
        pair["score_ordering"] != pair["true_risk_ordering"]
        for pair in score_pairs)
    ranks = []
    for rid in sorted({row["rid"] for row in rows}):
        state = [row for row in rows if row["rid"] == rid]
        score = np.asarray([row[tolerance_key] for row in state])
        q = np.asarray([row["Q_D"] for row in state])
        ranks.append(dict(
            rid=rid, spearman=safe_spearman(score, q),
            kendall_tau_b=(None if np.ptp(score) == 0 or np.ptp(q) == 0
                           else float(kendalltau(score, q).statistic))))
    median_spearman = (float(np.median([item["spearman"] for item in ranks
                                       if item["spearman"] is not None]))
                       if any(item["spearman"] is not None for item in ranks)
                       else None)
    agreement = concordant / len(informative) if informative else None
    unchanged_fraction = (unchanged_or_worse / len(score_pairs)
                          if score_pairs else None)
    if (len(informative) >= 12 and agreement >= .70 and
            median_spearman is not None and median_spearman >= .50):
        verdict = "ACTION_RANKING_PASS"
    elif (len(informative) >= 8 and agreement >= .58 and
          median_spearman is not None and median_spearman >= .20):
        verdict = "ACTION_RANKING_WEAK"
    elif ((len(informative) >= 8) or
          (score_pairs and unchanged_fraction >= .80)):
        verdict = "ACTION_RANKING_FAIL"
    else:
        verdict = "ACTION_RANKING_INCONCLUSIVE"
    summary = dict(
        total_pairs=len(pairs), score_different_pairs=len(score_pairs),
        score_ties=len(pairs) - len(score_pairs),
        true_risk_different_pairs=sum(pair["true_risk_different"] for pair in pairs),
        true_risk_ties=sum(not pair["true_risk_different"] for pair in pairs),
        informative_pairs=len(informative), concordant_pairs=concordant,
        discordant_pairs=discordant, sign_agreement=agreement,
        sign_agreement_wilson95=wilson(concordant, len(informative)),
        score_changes_Q_unchanged=sum(
            pair["category"] == "score_changes_Q_unchanged" for pair in pairs),
        unchanged_or_worse_fraction=unchanged_fraction,
        delta_R_abs=distribution([abs(pair["delta_R"]) for pair in score_pairs
                                  if pair["delta_R"] is not None]),
        delta_Q_abs=distribution([abs(pair["delta_Q"]) for pair in pairs]),
        rank_by_state=ranks, median_within_state_spearman=median_spearman,
        projection_alias_pairs=sum(pair["projection_alias"] for pair in score_pairs),
        active_set_switch_pairs=sum(pair["active_set_switch"] for pair in score_pairs),
        cone_change_pairs=sum(pair["cone_changes"] for pair in score_pairs),
        pre_projection_difference=distribution(
            [pair["pre_projection_difference"] for pair in score_pairs]),
        executed_action_difference=distribution(
            [pair["executed_action_difference"] for pair in score_pairs]),
        attenuation=distribution([pair["attenuation"] for pair in score_pairs
                                  if pair["attenuation"] is not None]),
        verdict=verdict,
        uncertainty_note=(
            "All pairwise continuation outcomes are paired by seed. A paired "
            "empirical bootstrap is degenerate when no discordance is observed; "
            "each all-deadlock arm's exact marginal 95% interval remains wide."),
    )
    return summary, pairs


def tolerance_sensitivity(rows, protocol):
    results = {}
    for tolerance in protocol["tolerances"]["active_sensitivity"]:
        label = f"{tolerance:.0e}"
        alternate = []
        for row in rows:
            copy = dict(row)
            copy["R_cone_pi2"] = row["active_tolerance_sensitivity"][label]["risk"]
            alternate.append(copy)
        summary, _ = action_pair_analysis(alternate, protocol)
        results[label] = summary
    return results


def failure_audit(p_records, q_records, old_root, protocol):
    all_records = p_records + q_records
    statuses = Counter(row["pi2"]["cone"]["status"] for row in all_records)
    ranks = Counter(row["pi2"]["cone"]["rank"] for row in all_records)
    lineality = Counter(row["pi2"]["cone"]["lineality_dimension"]
                        for row in all_records)
    speed_confounded = sum(row["pi2"]["cone"]["speed_confounded"]
                           for row in all_records)
    near_collinear = 0
    condition_rows = []
    for row in all_records:
        active = row["pi2"]["kkt"]["active_linear"]
        # A is recoverable from the trace; cone rank/status already records
        # degeneracy.  Pairwise generator angles are computed from facet data
        # only when at least two active rows exist in the stored state below.
        if len(active) >= 2:
            near_collinear += int(row["pi2"]["cone"]["rank"] < len(active))
        condition_rows.append(dict(
            state_id=row.get("state_id", row.get("state_action_id")),
            active_count=len(active), rank=row["pi2"]["cone"]["rank"],
            status=row["pi2"]["cone"]["status"],
            lineality=row["pi2"]["cone"]["lineality_dimension"],
        ))
    # Scan actual saved trajectories for consecutive active-set switching.
    transitions = switching = aba = alive_steps = 0
    seen = set()
    for record in p_records:
        path = Path(record["trace"])
        if path in seen:
            continue
        seen.add(path)
        with np.load(path, allow_pickle=False) as trace:
            sets = []
            for step in np.flatnonzero(trace["alive_pre"]):
                slack = trace["A"][step] @ trace["applied"][step] - trace["b"][step]
                sets.append(tuple(np.flatnonzero(
                    slack <= protocol["tolerances"]["active_primary"])))
            alive_steps += len(sets)
            transitions += max(0, len(sets) - 1)
            switching += sum(a != b for a, b in zip(sets[:-1], sets[1:]))
            aba += sum(a == c and a != b for a, b, c in
                       zip(sets[:-2], sets[1:-1], sets[2:]))
    return dict(
        scored_state_count=len(all_records), cone_status_counts=dict(statuses),
        cone_rank_counts={str(key): value for key, value in ranks.items()},
        lineality_dimension_counts={str(key): value for key, value in lineality.items()},
        full_space_count=statuses.get("full_space", 0),
        low_dimensional_boundary_count=statuses.get("boundary_low_dimensional", 0),
        empty_active_count=(statuses.get("empty_active_apex", 0) +
                            statuses.get("empty_active_outside", 0)),
        speed_confounded_count=speed_confounded,
        active_rank_deficient_multirow_count=near_collinear,
        trajectory_active_set_scan=dict(
            unique_traces=len(seen), alive_steps=alive_steps,
            adjacent_transitions=transitions, active_set_switches=switching,
            switch_fraction=(switching / transitions if transitions else None),
            immediate_ABA_switches=aba,
            note="ABA is a descriptive chattering flag, not proof of livelock."),
        state_geometry=condition_rows,
        score_discontinuity=(
            "Expected at active-set and nearest-facet switches; gradient audit "
            "is gated off if TEST Q fails."),
        livelock=(
            "Not identified by the frozen first-event TEST-Q panel; no outcome "
            "is relabeled and no post-deadlock continuation is invented."),
    )


def tautology_audit(p_rows_, q_rows_):
    stages = [row[stage] for row in p_rows_ + q_rows_
              for stage in ("pi1", "pi2")]
    return dict(
        classifications=[
            dict(quantity="p-y in cone of active feasible-set Jacobians",
                 class_="A", explanation=(
                     "Automatic KKT stationarity for every certified Euclidean "
                     "projection; includes active safety rows and speed-ball tangents.")),
            dict(quantity="y-p in cone(-A_active, +speed radial normals)",
                 class_="A", explanation=(
                     "The same KKT identity with signs reversed; not deadlock information.")),
            dict(quantity="y in cone(-A_active) when p=0",
                 class_="B", explanation=(
                     "At exact zero output speed balls are inactive, so this follows "
                     "from KKT only at the special stall p=0.")),
            dict(quantity="ambient signed distance of y to boundary(cone(-A_active))",
                 class_="C", explanation=(
                     "Not fixed by KKT away from p=0; it uses cone facets/rank and "
                     "the nominal drive rather than the projection residual.")),
            dict(quantity="cone rank, lineality, full-space status, facet multiplicity",
                 class_="C", explanation=(
                     "Additional active-normal geometry independent of residual magnitude.")),
            dict(quantity="dist(y, cone(-A_active)) <= ||p|| with inactive speed balls",
                 class_="A/B bound", explanation=(
                     "A direct consequence of y=(y-p)+p and the KKT residual cone; "
                     "it explains why outside distance can collapse to an executed-speed proxy.")),
        ],
        class_key=dict(A="automatic projection identity",
                       B="special only at/near zero output",
                       C="additional geometry"),
        numerical_kkt=dict(
            stages=len(stages),
            max_stationarity_inf=float(max(
                stage["kkt"]["stationarity_inf"] for stage in stages)),
            max_linear_violation=float(max(
                max(0.0, -stage["kkt"]["min_linear_slack"])
                for stage in stages)),
            speed_active_stages=sum(bool(stage["kkt"]["active_speed"])
                                    for stage in stages),
        ),
    )


def flatten_p(row):
    return {key: row[key] for key in (
        "state_id", "rid", "early_step", "deadlock", "success",
        "lead_seconds", "R_cone_pi1", "R_cone_pi2",
        "active_safety_count_pi1", "active_safety_count_pi2",
        "safety_intervention_norm", "safe_norm", "executed_norm",
        "progress_speed", "actual_one_step_progress_speed",
        "pairwise_center_distance", "pairwise_surface_clearance",
        "old_direct_R_CERT", "old_augmented_R_CERT")}


def flatten_q(row):
    return {key: row[key] for key in (
        "state_action_id", "rid", "arm", "R_cone_pi1", "R_cone_pi2",
        "Q_D", "deadlocks", "continuations", "active_safety_count_pi1",
        "active_safety_count_pi2", "safety_intervention_norm",
        "pi2_projection_norm", "safe_norm", "executed_norm",
        "pairwise_center_distance", "old_direct_R_CERT",
        "old_augmented_R_CERT")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path,
                        default=Path("/home/zhihan/research/Rrisk_C1"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "/home/zhihan/research/02_C1_Toy_GiveWay/baseline_309_314/"
        "checkpoints/seed0/ckpt_0025000.pkl"))
    parser.add_argument("--out", type=Path, default=HERE / "data_v1")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("new output directory required")
    args.out.mkdir(parents=True)
    (args.out / "raw").mkdir()
    (args.out / "tables").mkdir()
    self_test()
    protocol = json.loads((HERE / "predeclared_protocol.json").read_text())
    old_data = args.old_root / "diagnostics/rrisk/data_v1"
    old_manifest = args.old_root / "diagnostics/rrisk/diagnostic_manifest.yaml"
    old_protocol_path = (args.old_root /
                         "results/c1_vi_r_cert_probe_v1/protocol.json")
    p_source = old_data / "test_p_records.json"
    q_source = old_data / "test_q_records.json"
    old_analysis = old_data / "analysis_p_q_c_proj.json"
    p_input = json.loads(p_source.read_text())
    q_input = json.loads(q_source.read_text())
    old_protocol = json.loads(old_protocol_path.read_text())
    reconstructor = NominalReconstructor(args.checkpoint)
    if reconstructor.plant.max_steps != 850 or reconstructor.plant.dt != .05:
        raise RuntimeError("frozen plant changed")
    p_scored = p_rows(args.old_root, p_input, reconstructor, protocol)
    q_scored = q_rows(args.old_root, q_input, reconstructor, protocol,
                      prefix_seed=84200)
    prediction = prediction_analysis(p_scored, protocol)
    qualification = qualification_analysis(p_scored)
    q_summary, q_pairs = action_pair_analysis(q_scored, protocol)
    pi1_summary, _ = action_pair_analysis(q_scored, protocol,
                                          tolerance_key="R_cone_pi1")
    sensitivity = tolerance_sensitivity(q_scored, protocol)
    failures = failure_audit(p_scored, q_scored, args.old_root, protocol)
    tautology = tautology_audit(p_scored, q_scored)
    if prediction["verdict"] == "PREDICTIVE" and q_summary["verdict"] == "ACTION_RANKING_FAIL":
        final_case = "CASE B — GOOD GEOMETRIC PREDICTOR, BAD CONTROL SCORE"
    elif q_summary["verdict"] in ("ACTION_RANKING_PASS", "ACTION_RANKING_WEAK"):
        final_case = "UNRESOLVED_AFTER_Q_PASS"
    else:
        final_case = "CASE E — INCONCLUSIVE"
    stop = q_summary["verdict"] not in ("ACTION_RANKING_PASS",
                                        "ACTION_RANKING_WEAK")
    summary = dict(
        test_p=prediction, qualification=qualification,
        test_q=q_summary, pi1_secondary_test_q=pi1_summary,
        active_tolerance_sensitivity=sensitivity,
        projection_analysis={key: q_summary[key] for key in (
            "projection_alias_pairs", "active_set_switch_pairs",
            "cone_change_pairs", "pre_projection_difference",
            "executed_action_difference", "attenuation")},
        gradient_gate=dict(run=not stop,
                           result=("SKIPPED_TEST_Q_FAILED" if stop else
                                   "REQUIRED_BUT_NOT_RUN_BY_PILOT_SCRIPT")),
        minimum_change_escape_gate=dict(
            derivation_only=True, outcome_rollouts_run=False,
            reason=("TEST Q did not establish action ranking" if stop else
                    "TEST Q passed; a separately frozen intervention protocol is required")),
        final_case=final_case,
        stopped_after_pilot=stop,
        fresh_rollouts=0, training_updates=0,
    )
    manifest = dict(
        schema=protocol["schema"],
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
        checkpoints=[dict(path=str(args.checkpoint), sha256=sha256(args.checkpoint),
                          seed=0, update=25000)],
        old_evidence=dict(
            root=str(args.old_root),
            test_p_records=dict(path=str(p_source), sha256=sha256(p_source)),
            test_q_records=dict(path=str(q_source), sha256=sha256(q_source)),
            analysis=dict(path=str(old_analysis), sha256=sha256(old_analysis)),
            diagnostic_manifest=dict(path=str(old_manifest), sha256=sha256(old_manifest)),
            old_protocol=dict(path=str(old_protocol_path),
                              sha256=sha256(old_protocol_path))),
        source_hashes={str(path.relative_to(ROOT)): sha256(path) for path in (
            ROOT / "single_integrator/cbf.py",
            ROOT / "single_integrator/c1/differentiable_rollout.py",
            ROOT / "single_integrator/c1/risk/joint_frozen.py",
            ROOT / "single_integrator/c1/rollout_vi_r_cert.py",
            HERE / "cone_geometry.py", HERE / "run_cone_rrisk_pilot.py",
            HERE / "predeclared_protocol.json")},
        seeds=dict(
            prefix=84200, test_p_risk=[85101], test_p_outcome=[85102],
            test_q_risk=[85201],
            test_q_continuation=[85210, 85211, 85212, 85213,
                                 85214, 85215, 85216, 85217],
            bootstrap=protocol["statistics"]["bootstrap_seed"]),
        state_ids=dict(
            test_p=[row["state_id"] for row in p_scored],
            test_q=[row["state_action_id"] for row in q_scored]),
        old_test_q_pair_ids=[pair["pair_id"] for pair in q_pairs],
        tolerances=protocol["tolerances"],
        intervention_budgets=dict(
            test_q_single_action_pre_projection_l2=0.2,
            tested_action_step=100,
            gradient=None, minimum_change_escape=None),
        continuation_counts=dict(
            test_p_per_state=1, test_q_per_action=8,
            test_q_total=len(q_scored) * 8, fresh=0),
        exact_risk_definition=protocol["primary_candidate"],
        primary_declared_before_outcomes="Pi_2",
        secondary="Pi_1",
        no_training=True, no_fresh_rollouts=True,
        old_start_count=len(old_protocol["starts"]),
    )
    save_json(args.out / "raw/p_state_scores.json", p_scored)
    save_json(args.out / "raw/q_action_scores.json", q_scored)
    save_json(args.out / "raw/q_pair_scores.json", q_pairs)
    save_json(args.out / "prediction.json", prediction)
    save_json(args.out / "qualification.json", qualification)
    save_json(args.out / "test_q.json", q_summary)
    save_json(args.out / "projection_analysis.json", summary["projection_analysis"])
    save_json(args.out / "tolerance_sensitivity.json", sensitivity)
    save_json(args.out / "projection_kkt_tautology.json", tautology)
    save_json(args.out / "failure_modes.json", failures)
    save_json(args.out / "manifest.json", manifest)
    save_json(args.out / "summary.json", summary)
    save_csv(args.out / "tables/test_p_states.csv", map(flatten_p, p_scored))
    save_csv(args.out / "tables/test_q_actions.csv", map(flatten_q, q_scored))
    save_csv(args.out / "tables/test_q_pairs.csv", q_pairs)
    save_csv(args.out / "tables/qualification_correlations.csv",
             qualification["correlations"])
    save_csv(args.out / "tables/geometry_failures.csv",
             failures["state_geometry"])
    print(json.dumps(clean(dict(
        final_case=final_case, test_p=prediction["verdict"],
        test_q=q_summary["verdict"],
        score_different_pairs=q_summary["score_different_pairs"],
        true_risk_different_pairs=q_summary["true_risk_different_pairs"],
        stopped_after_pilot=stop)), indent=2))


if __name__ == "__main__":
    main()
