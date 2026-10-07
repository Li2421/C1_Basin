"""Freeze the first local refinement design before running new outcomes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SBMA = ROOT / "diagnostics" / "success_basin_multimodality"
DEFORMATION = ROOT / "diagnostics" / "success_basin_deformation"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name: str, value: object) -> None:
    (HERE / name).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def key(values) -> tuple[float, float, float]:
    return tuple(round(float(value), 12) for value in values)


def d1_points() -> set[tuple[float, float, float]]:
    points = {(g, 0.0, 0.0) for g in (0.25, 0.375, 0.5, 0.625, 0.75, 0.875)}
    points |= {(0.75, s, 0.0) for s in (-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25)}
    points |= {(0.75, 0.0, r) for r in (-0.25, -0.125, 0.0, 0.125, 0.25)}
    points |= set(product((0.625, 0.75), (-0.25, -0.125, 0.0, 0.125), (-0.125, 0.0)))
    return {key(point) for point in points}


def d24_points() -> set[tuple[float, float, float]]:
    points = {(g, -0.5, 0.0) for g in (0.25, 0.375, 0.5, 0.625, 0.75)}
    points |= {(0.5, s, 0.0) for s in (-0.875, -0.75, -0.625, -0.5, -0.375, -0.25)}
    points |= {(0.5, -0.5, r) for r in (-0.25, -0.125, 0.0, 0.125, 0.25)}
    points |= set(product((0.375, 0.5, 0.625), (-0.75, -0.625, -0.5), (-0.125, 0.0)))
    return {key(point) for point in points}


def old_phase_a_points(state_id: str) -> set[tuple[float, float, float]]:
    manifest = json.loads(
        (SBMA / "raw" / f"phase_a_{state_id}" / "manifest.json").read_text()
    )
    return {key(row["eta"]) for row in manifest["records"]}


def main() -> None:
    old_protocol = json.loads((SBMA / "protocol.json").read_text())
    states = {
        item["state_id"]: item
        for item in old_protocol["primary_states"]
        if item["state_id"] in {"D1_pair231", "D2_pair228", "D4_pair227"}
    }
    designs = {
        "D1_pair231": sorted(d1_points()),
        "D2_pair228": sorted(d24_points()),
        "D4_pair227": sorted(d24_points()),
    }
    seeds = list(range(95101001, 95101017))
    jobs = []
    reused = {}
    for state_id, points in designs.items():
        cached = old_phase_a_points(state_id)
        reused[state_id] = [list(point) for point in points if point in cached]
        for eta in points:
            if eta in cached:
                continue
            for seed in seeds:
                jobs.append(
                    {
                        "state_id": state_id,
                        "eta": list(eta),
                        "seed": seed,
                        "cell_id": "phase1_" + "_".join(f"{value:+.3f}" for value in eta),
                    }
                )
    protocol = {
        "study": "success_basin_deformation_refinement",
        "phase": "phase1_local_boundary_expansion",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "states": states,
        "environment": old_protocol["environment"],
        "checkpoint": old_protocol["checkpoint"],
        "eta_semantics": old_protocol["policy_family"],
        "eta_validity_audit": {
            "DiagnosticPhi_requirement": "three finite scalars only",
            "eta_clipping": False,
            "eta_normalization": False,
            "negative_relative_semantics": "sign reversal of the existing bounded self-minus-other basis; no controller change",
            "eta2_below_minus_half_semantics": "continues to reduce the pre-second-projection u_safe coefficient 1+eta2; eta2=-1 would cancel it",
            "all_phase1_values_finite": True,
        },
        "J_def": {
            "formula": "0.05 * sum_k ||u_exec[k]-u_safe[k]||_F^2",
            "same_corrected_state_and_flow_sample": True,
            "uses_post_second_projection_action": True,
            "discount": None,
            "normalization": None,
        },
        "success_criterion": old_protocol["classification"],
        "seeds": seeds,
        "common_random_numbers": True,
        "previous_minima": {
            "D1_pair231": [0.75, 0.0, 0.0],
            "D2_pair228": [0.5, -0.5, 0.0],
            "D4_pair227": [0.5, -0.5, 0.0],
        },
        "phase1_points": {state: [list(point) for point in points] for state, points in designs.items()},
        "reused_current_phase_a_points": reused,
        "new_points": {
            state: [list(point) for point in points if point not in old_phase_a_points(state)]
            for state, points in designs.items()
        },
        "new_rollouts": len(jobs),
        "source_hashes": {
            "sbma_protocol": sha(SBMA / "protocol.json"),
            "sbma_run": sha(SBMA / "run.py"),
            "corrector": sha(ROOT / "diagnostics/cl_fhcb/closed_loop.py"),
            "environment": sha(SYSROOT / "single_integrator/environment.py"),
            "projection": sha(SYSROOT / "single_integrator/cbf.py"),
            "retry": sha(SBMA / "exact_projector.py"),
            "deformation_results": sha(DEFORMATION / "results.json"),
            "checkpoint": sha(Path(old_protocol["checkpoint"])),
        },
    }
    write("protocol.json", protocol)
    write("phase1_jobs.json", jobs)
    for state_id in designs:
        write(
            f"phase1_{state_id}_jobs.json",
            [job for job in jobs if job["state_id"] == state_id],
        )
    print(
        json.dumps(
            {
                "points": {state: len(points) for state, points in designs.items()},
                "reused_points": {state: len(points) for state, points in reused.items()},
                "new_points": {state: len(protocol["new_points"][state]) for state in designs},
                "new_rollouts": len(jobs),
                "expected_max_physical_steps": sum(
                    old_protocol["environment"]["max_steps"] - states[job["state_id"]]["start_step"]
                    for job in jobs
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
