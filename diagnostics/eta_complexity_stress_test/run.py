"""Run the isolated eta-capacity stress test; no checkpoints or training."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np

from .benchmark import DEFAULT_COUNT, DEFAULT_SEED, freeze_benchmark, generate_benchmark, load_benchmark
from .controller import rollout
from .environment import StressConfig
from .oracle import ORACLE_MODES, rollout_oracle, verify_oracle
from .plots import plot_eta_scatter, plot_oracle_trajectory, plot_solver_difficulty, plot_success_heatmaps, plot_trajectory
from .search import EtaBounds, SearchSamples, local_robustness, sobol_search, solver_benchmark, uniform_search


HERE = Path(__file__).resolve().parent


def _sobol_worker(payload: tuple[np.ndarray, int, int, StressConfig, EtaBounds]) -> SearchSamples:
    """Pickle-safe independent rollout batch; eta samples share no state."""
    positions, count, seed, config, bounds = payload
    return sobol_search(positions, count, seed, config, bounds)


def _write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _replay_screen_successes(output: Path, items, config: StressConfig) -> list[dict]:
    """Persist replayable trajectories for actual global-screen successes."""
    rows = []
    for item in items:
        source = output / "searches" / f"sobol_{item.identifier}.npz"
        if not source.exists():
            continue
        with np.load(source, allow_pickle=False) as archive:
            points = archive["points"]
            successful = archive["success"]
        for mode_index, eta in enumerate(points[successful][:2], start=1):
            trace = rollout(item.positions, eta=eta, variant="eta", config=config)
            if not trace.success:
                raise RuntimeError("saved Sobol success did not reproduce deterministically")
            stem = f"eta_screen_{item.identifier}_mode{mode_index}"
            trace.save(output / "trajectories", stem)
            plot_trajectory(trace, output / "figures" / f"trajectory_{stem}.png", config)
            rows.append({
                "id": item.identifier,
                "family": item.family,
                "eta": np.asarray(eta, dtype=float).tolist(),
                "trajectory_stem": stem,
                **trace.summary,
            })
    _write_json(output / "observed_screen_successes.json", rows)
    return rows


def _save_samples(path: Path, result: SearchSamples, bounds: EtaBounds) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), points=result.points, success=result.success, steps=result.steps, objective=result.objective)
    _write_json(path.with_suffix(".json"), {
        **result.summary(bounds), "terminations": list(result.terminations),
        "failure_modes": list(result.failure_modes), "coordination_modes": list(result.coordination_modes),
        "array_file": path.with_suffix(".npz").name,
    })


def _select_representatives(items, safe_records: list[dict], search_records: list[dict], limit: int) -> list:
    """Prioritize observed novel eta successes, then failed family coverage."""
    failed = {row["id"] for row in safe_records if not row["success"]}
    selected, seen = [], set()
    records = {row["id"]: row for row in search_records}
    for item in items:
        record = records[item.identifier]
        if item.identifier in failed and record["successes"] > 0:
            selected.append(item)
            seen.add(item.identifier)
            if len(selected) >= limit:
                return selected
    for item in items:
        if item.identifier in failed and item.family not in {entry.family for entry in selected}:
            selected.append(item)
    selected_ids = {item.identifier for item in selected}
    for item in items:
        if item.identifier in failed and item.identifier not in selected_ids:
            selected.append(item)
            selected_ids.add(item.identifier)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _load_verified_gate(path: Path, benchmark_meta: dict, config: StressConfig) -> tuple[dict, list[dict]]:
    """Reuse an already persisted gate only when its corpus and plant match."""
    result_path = path / "results.json"
    baseline_path = path / "baseline_results.json"
    if not result_path.exists() or not baseline_path.exists():
        raise FileNotFoundError("verification directory needs results.json and baseline_results.json")
    verified = json.loads(result_path.read_text(encoding="utf-8"))
    if verified["benchmark"]["digest"] != benchmark_meta["digest"]:
        raise ValueError("verification benchmark digest does not match the requested IC set")
    if verified["benchmark"]["environment_fingerprint"] != config.fingerprint:
        raise ValueError("verification environment fingerprint does not match")
    oracle = verified["oracle"]
    if oracle["success_rate"] < 0.99:
        raise RuntimeError("reused verification did not pass the oracle solvability gate")
    return oracle, json.loads(baseline_path.read_text(encoding="utf-8"))


def _classify_capacity(oracle_rate: float, eta_existence_rate: float, median_rho: float) -> str:
    if oracle_rate < 0.99:
        return "UNRESOLVED_ORACLE"
    if eta_existence_rate >= 0.75 and median_rho >= 0.02:
        return "STRONG"
    if eta_existence_rate > 0.0:
        return "PARTIAL"
    return "FAILED"


def _classify_difficulty(solver_rows: list[dict]) -> str:
    successful = [row["evaluations_to_first_success"] for row in solver_rows if row["evaluations_to_first_success"] is not None]
    if not successful:
        return "EXTREME"
    # A fast first hit in one lucky repeat cannot make a landscape easy when
    # most independent fixed-budget searches miss it entirely.
    if len(successful) / len(solver_rows) < 0.5:
        return "EXTREME"
    median = float(np.median(successful))
    if median <= 64:
        return "EASY"
    if median <= 256:
        return "MODERATE"
    if median <= 1024:
        return "HARD"
    return "EXTREME"


def _report(path: Path, result: dict) -> None:
    search = result["search"]
    samples = search["samples_per_ic"]
    eta_all = search.get("eta_exists_all_ics", search.get("eta_exists", 0))
    median_all = search.get("median_success_fraction_all_ics", search.get("median_success_fraction", float("nan")))
    safety_failed = search.get("safety_failed_ics")
    eta_novel = search.get("eta_exists_on_safety_failed_ics")
    median_novel = search.get("median_success_fraction_on_safety_failed_ics")
    novel_text = "" if safety_failed is None else f"""
Of the {safety_failed} ICs where hard safety itself failed, sampled eta found a
success for {eta_novel}; their median sampled success fraction was
{median_novel:.6f}. This is the capacity-relevant comparison, rather than
counting the eta=0 equivalence on already-safe-successful ICs.
"""
    dense_text = ""
    if result.get("dense_representatives"):
        rows = "\n".join(
            f"| {entry['id']} ({entry['family']}) | {entry['successes']} / {entry['evaluations']} | {entry['first_success_evaluation']} |"
            for entry in result["dense_representatives"]
        )
        dense_text = f"""
### Dense representative scans

| IC | successes | first success eval |
|---|---:|---:|
{rows}
"""
    solver_text = ""
    if result.get("solver_benchmark"):
        grouped: dict[str, list[dict]] = {}
        for entry in result["solver_benchmark"]:
            grouped.setdefault(entry["solver"], []).append(entry)
        rows = "\n".join(
            f"| {name} | {sum(row['success'] for row in values)} / {len(values)} | {min((row['evaluations_to_first_success'] for row in values if row['evaluations_to_first_success'] is not None), default='--')} |"
            for name, values in sorted(grouped.items())
        )
        solver_text = f"""
### Fixed-budget solver probes

Each repeat used {result['solver_benchmark'][0]['budget']} rollout evaluations.

| solver | successful repeats | best first success eval |
|---|---:|---:|
{rows}
"""
    observed_text = ""
    if result.get("observed_screen_successes"):
        rows = "\n".join(
            f"| {entry['id']} | `{entry['eta']}` | {entry['episode_steps']} | `{entry['coordination_signature']}` |"
            for entry in result["observed_screen_successes"]
        )
        observed_text = f"""
### Replayed global-screen successes

| IC | eta | steps | coordination signature |
|---|---|---:|---|
{rows}
"""
    text = f"""# Coupled Dual-Intersection eta stress test

## Scope

This isolated diagnostic uses no MAC Flow/Flow-BC checkpoint, neural nominal
policy, demonstration collection, or training.  The nominal controller follows
fixed geometric waypoints independently for all eight agents.  The tested law
is the repository's unchanged shared 3-D correction:

```text
g = eta_goal * B_goal + eta_safe * u_safe + eta_relative * B_rel
u_exec = hard_project(u_safe + g)
```

`B_goal` is bounded goal displacement; `B_rel` is the all-other-agents,
pairwise-bound-then-mean basis; both hard projections are the existing generic
N-agent projector.  The search domain was `{result['bounds']}`.

## Environment and solvability gate

The fixed corpus has {result['benchmark']['count']} initial conditions and SHA-256
`{result['benchmark']['digest']}`.  The explicit centralized scheduler solved
{result['oracle']['successes']}/{result['oracle']['rollouts']} ({result['oracle']['success_rate']:.3f})
using its selected phase ordering.  The oracle is not used by nominal, safety,
or eta rollouts.

## Baselines

| controller | successes / ICs |
|---|---:|
| analytic nominal | {result['baselines']['nominal_successes']} / {result['benchmark']['count']} |
| hard safety only | {result['baselines']['safe_successes']} / {result['benchmark']['count']} |

## Eta existence and basin estimate

The global Sobol screen tested {samples} eta values
per IC.  Empirical nonempty success sets occurred for
{eta_all} / {result['benchmark']['count']} ICs.  The
median sampled success fraction was {median_all:.6f}.
{novel_text}
These are finite-sample estimates, not proofs that an unobserved eta set is
empty.
{dense_text}{solver_text}{observed_text}

## Verdict

**ETA CAPACITY: {result['verdict']['eta_capacity']}**

**ETA SOLVING DIFFICULTY: {result['verdict']['eta_solving_difficulty']}**

The capacity label separates oracle-verified physical solvability from sampled
eta existence.  The difficulty label comes from independent fixed-budget
uniform, Sobol, and diagonal-CEM searches and is only reported for
representative ICs with empirical eta success.
"""
    path.write_text(text, encoding="utf-8")


def run_study(args) -> dict:
    config, bounds = StressConfig(), EtaBounds()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "refresh":
        result_path = output / "results.json"
        if not result_path.exists():
            raise FileNotFoundError("refresh requires an existing results.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        search = result["search"]
        failed = search.get("safety_failed_ics", 0)
        found = search.get("eta_exists_on_safety_failed_ics", search.get("eta_exists", 0))
        rate = found / max(failed, 1)
        rho = search.get("median_success_fraction_on_safety_failed_ics", search.get("median_success_fraction", float("nan")))
        result["verdict"] = {
            "eta_capacity": _classify_capacity(result["oracle"]["success_rate"], rate, rho),
            "eta_solving_difficulty": _classify_difficulty(result.get("solver_benchmark", [])),
        }
        _write_json(result_path, result)
        _report(output / "REPORT.md", result)
        return result
    ic_path = Path(args.ic_path).resolve()
    if args.stage == "replay":
        result_path = output / "results.json"
        if not result_path.exists():
            raise FileNotFoundError("replay requires an existing results.json")
        items = load_benchmark(ic_path, config)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        meta = json.loads(ic_path.read_text(encoding="utf-8"))
        if result["benchmark"]["digest"] != meta["digest"] or result["benchmark"]["environment_fingerprint"] != config.fingerprint:
            raise ValueError("replay corpus or environment does not match persisted results")
        result["observed_screen_successes"] = _replay_screen_successes(output, items, config)
        _write_json(result_path, result)
        _report(output / "REPORT.md", result)
        return result
    if ic_path.exists():
        items = load_benchmark(ic_path, config)
    else:
        items = generate_benchmark(args.ic_count, args.seed, config)
        freeze_benchmark(ic_path, items, config)
    benchmark_meta = json.loads(ic_path.read_text(encoding="utf-8"))
    if args.reuse_verification is not None:
        oracle, baseline_rows = _load_verified_gate(Path(args.reuse_verification).resolve(), benchmark_meta, config)
    else:
        oracle = verify_oracle(items, config=config, mode=args.oracle_mode)
        if oracle["success_rate"] < 0.99:
            _write_json(output / "oracle_gate_failure.json", oracle)
            raise RuntimeError("oracle solvability gate failed; do not judge eta before redesigning geometry")
        baseline_rows = []
        for item in items:
            for variant in ("nominal", "safe"):
                trace = rollout(item.positions, variant=variant, config=config)
                baseline_rows.append({"id": item.identifier, "family": item.family, **trace.summary})
    nominal = [row for row in baseline_rows if row["variant"] == "nominal"]
    safe = [row for row in baseline_rows if row["variant"] == "safe"]
    _write_json(output / "baseline_results.json", baseline_rows)
    if args.stage == "verify":
        result = {
            "benchmark": {"count": len(items), "digest": benchmark_meta["digest"], "path": str(ic_path), "environment_fingerprint": config.fingerprint},
            "bounds": bounds.to_dict(), "oracle": oracle,
            "baselines": {"nominal_successes": sum(row["success"] for row in nominal), "safe_successes": sum(row["success"] for row in safe)},
            "search": {"samples_per_ic": 0, "eta_exists": 0, "median_success_fraction": float("nan")},
            "verdict": {"eta_capacity": "PENDING_SEARCH", "eta_solving_difficulty": "PENDING_SEARCH"},
        }
        _write_json(output / "results.json", result); _report(output / "REPORT.md", result)
        return result

    summaries_by_id: dict[str, dict] = {}
    started = time.perf_counter()
    pending = []
    for index, item in enumerate(items):
        saved = output / "searches" / f"sobol_{item.identifier}.json"
        if args.resume and saved.exists():
            summary = json.loads(saved.read_text(encoding="utf-8"))
            if summary.get("evaluations") == args.search_samples and summary.get("bounds") == bounds.to_dict():
                summaries_by_id[item.identifier] = {"id": item.identifier, "family": item.family, **summary}
                continue
        pending.append((index, item))
    if args.workers == 1:
        completed = ((item, _sobol_worker((item.positions, args.search_samples, args.seed + index, config, bounds))) for index, item in pending)
        for item, search in completed:
            _save_samples(output / "searches" / f"sobol_{item.identifier}", search, bounds)
            summaries_by_id[item.identifier] = {"id": item.identifier, "family": item.family, **search.summary(bounds)}
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_sobol_worker, (item.positions, args.search_samples, args.seed + index, config, bounds)): item
                for index, item in pending
            }
            for future in as_completed(futures):
                item = futures[future]
                search = future.result()
                _save_samples(output / "searches" / f"sobol_{item.identifier}", search, bounds)
                summaries_by_id[item.identifier] = {"id": item.identifier, "family": item.family, **search.summary(bounds)}
    summaries = [summaries_by_id[item.identifier] for item in items]
    _write_json(output / "eta_search_summary.json", summaries)
    representatives = _select_representatives(items, safe, summaries, args.representatives)
    dense_rows, solver_rows, robustness_rows = [], [], []
    dense_by_id: dict[str, SearchSamples] = {}
    dense_jobs = list(enumerate(representatives))
    if args.workers == 1:
        for rep_index, item in dense_jobs:
            dense_by_id[item.identifier] = _sobol_worker((item.positions, args.dense_samples, args.seed + 100_000 + rep_index, config, bounds))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(dense_jobs))) as executor:
            futures = {
                executor.submit(_sobol_worker, (item.positions, args.dense_samples, args.seed + 100_000 + rep_index, config, bounds)): item
                for rep_index, item in dense_jobs
            }
            for future in as_completed(futures):
                item = futures[future]
                dense_by_id[item.identifier] = future.result()
    for rep_index, item in enumerate(representatives):
        dense = dense_by_id[item.identifier]
        _save_samples(output / "dense" / f"sobol_{item.identifier}", dense, bounds)
        dense_rows.append({"id": item.identifier, "family": item.family, **dense.summary(bounds)})
        plot_eta_scatter(dense, bounds, output / "figures" / f"eta_scatter_{item.identifier}.png", f"{item.identifier}: sampled eta outcomes")
        plot_success_heatmaps(dense, bounds, output / "figures" / f"eta_heatmap_{item.identifier}.png", f"{item.identifier}: pairwise eta success fraction")
        success_points = dense.points[dense.success][:args.local_centers]
        if len(success_points):
            robustness_rows.extend([{ "id": item.identifier, **row } for row in local_robustness(item.positions, success_points, args.seed + 200_000 + rep_index, config, bounds, args.local_samples)])
            solver_rows.extend([{ "id": item.identifier, **row } for row in solver_benchmark(item.positions, config, bounds, args.solver_budget, args.solver_repeats, args.seed + 300_000 + rep_index)])
            traces_by_mode = {}
            for eta in success_points:
                trace = rollout(item.positions, eta=eta, variant="eta", config=config)
                traces_by_mode.setdefault(trace.summary["coordination_signature"], trace)
            for mode_index, trace in enumerate(traces_by_mode.values()):
                if mode_index >= 2:
                    break
                trace.save(output / "trajectories", f"eta_{item.identifier}_mode{mode_index + 1}")
                plot_trajectory(trace, output / "figures" / f"trajectory_eta_{item.identifier}_mode{mode_index + 1}.png", config)
        else:
            trace = rollout(item.positions, eta=(1.0, 0.0, 0.25), variant="eta", config=config)
            trace.save(output / "trajectories", f"eta_representation_failure_{item.identifier}")
            plot_trajectory(trace, output / "figures" / f"trajectory_representation_failure_{item.identifier}.png", config)
        safety_trace = rollout(item.positions, variant="safe", config=config)
        safety_trace.save(output / "trajectories", f"safety_{item.identifier}")
        plot_trajectory(safety_trace, output / "figures" / f"trajectory_safety_{item.identifier}.png", config)
        oracle_trace = rollout_oracle(item.positions, mode=args.oracle_mode, config=config)
        # Reuse the standard trace-like visualization contract for safety/eta only;
        # oracle positions/actions are persisted separately for transparent inspection.
        np.savez_compressed(output / "trajectories" / f"oracle_{item.identifier}.npz", positions=oracle_trace.positions, actions=oracle_trace.actions, active_agents=oracle_trace.active_agents)
        plot_oracle_trajectory(oracle_trace.positions, oracle_trace.termination, oracle_trace.mode, output / "figures" / f"trajectory_oracle_{item.identifier}.png", config)
    if solver_rows:
        plot_solver_difficulty(solver_rows, output / "figures" / "solver_difficulty.png", "Eta solver comparison across representative ICs")
    _write_json(output / "dense_summary.json", dense_rows)
    _write_json(output / "local_robustness.json", robustness_rows)
    _write_json(output / "solver_benchmark.json", solver_rows)
    success_fractions = [row["success_fraction"] for row in summaries]
    existence = sum(row["successes"] > 0 for row in summaries)
    safety_failed = {row["id"] for row in safe if not row["success"]}
    novel_summaries = [row for row in summaries if row["id"] in safety_failed]
    novel_fractions = [row["success_fraction"] for row in novel_summaries]
    novel_existence = sum(row["successes"] > 0 for row in novel_summaries)
    novel_rate = novel_existence / max(len(novel_summaries), 1)
    novel_median = float(np.median(novel_fractions)) if novel_fractions else float("nan")
    result = {
        "benchmark": {"count": len(items), "digest": benchmark_meta["digest"], "path": str(ic_path), "environment_fingerprint": config.fingerprint},
        "bounds": bounds.to_dict(), "oracle": oracle,
        "baselines": {"nominal_successes": sum(row["success"] for row in nominal), "safe_successes": sum(row["success"] for row in safe)},
        "search": {
            "samples_per_ic": args.search_samples,
            "eta_exists_all_ics": existence,
            "median_success_fraction_all_ics": float(np.median(success_fractions)),
            "safety_failed_ics": len(novel_summaries),
            "eta_exists_on_safety_failed_ics": novel_existence,
            "median_success_fraction_on_safety_failed_ics": novel_median,
            "wall_seconds": time.perf_counter() - started,
        },
        "dense_representatives": dense_rows, "solver_benchmark": solver_rows,
        "verdict": {
            "eta_capacity": _classify_capacity(oracle["success_rate"], novel_rate, novel_median),
            "eta_solving_difficulty": _classify_difficulty(solver_rows),
        },
    }
    _write_json(output / "results.json", result); _report(output / "REPORT.md", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("verify", "search", "refresh", "replay"), default="verify", help="verify freezes/validates ICs and baselines; search runs eta analysis; refresh recomputes verdict/report; replay saves deterministic screen-success trajectories")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "latest")
    parser.add_argument("--reuse-verification", type=Path, default=None, help="reuse a matching persisted oracle/baseline gate")
    parser.add_argument("--ic-path", type=Path, default=HERE / "data" / "benchmark_ics_v5.json")
    parser.add_argument("--ic-count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--oracle-mode", choices=tuple(ORACLE_MODES), default="ltr_then_vertical_then_rtl")
    parser.add_argument("--search-samples", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1, help="independent IC screen workers; never changes rollout semantics")
    parser.add_argument("--resume", action="store_true", help="reuse matching per-IC Sobol artifacts already saved in --output")
    parser.add_argument("--dense-samples", type=int, default=16_384)
    parser.add_argument("--representatives", type=int, default=4)
    parser.add_argument("--local-centers", type=int, default=3)
    parser.add_argument("--local-samples", type=int, default=96)
    parser.add_argument("--solver-budget", type=int, default=512)
    parser.add_argument("--solver-repeats", type=int, default=12)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.ic_count <= 0 or args.search_samples <= 0 or args.dense_samples <= 0 or args.workers <= 0:
        raise ValueError("IC/search counts and workers must be positive")
    result = run_study(args)
    print(json.dumps(result["verdict"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
