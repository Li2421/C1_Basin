#!/usr/bin/env python3
"""Pre-register the structural eta-capacity experiment before evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from diagnostics.double_bottleneck_eta_representation_capacity.tools.representations import REPRESENTATIONS


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta_representation_capacity"
PRIOR = ROOT / "diagnostics/double_bottleneck_eta3_basin"
HARD = ROOT / "diagnostics/double_bottleneck_hard_safety_baseline"
SELECTION_SEED = 740921
CONTROL_SEED = 740922
STATE_SEED = 740923
SEARCH_SEEDS = {"P0-3D": 810031, "P1-Agent6": 810041, "P2-Pair8": 810053, "P3-Temporal6": 810067}
LOCAL_SEEDS = {"P0-3D": 820031, "P1-Agent6": 820041, "P2-Pair8": 820053, "P3-Temporal6": 820067}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_new(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sample_rows(name: str, initial: int, dense: int) -> dict:
    rep = REPRESENTATIONS[name]
    total_power = int(np.ceil(np.log2(initial + dense)))
    unit = qmc.Sobol(rep.dimension, scramble=True, seed=SEARCH_SEEDS[name]).random_base2(total_power)
    scaled = qmc.scale(unit, rep.low, rep.high)
    stage_a = [
        {"parameter_id": f"A{index:03d}", "theta": scaled[index].tolist(), "sample_type": "sobol_stage_a"}
        for index in range(initial)
    ]
    stage_a.append({"parameter_id": "EMBED", "theta": rep.embed_p0().tolist(), "sample_type": "p0_a004_embedding"})
    stage_c = [
        {"parameter_id": f"C{index:03d}", "theta": scaled[initial + index].tolist(), "sample_type": "sobol_negative_confirmation"}
        for index in range(dense)
    ]
    return {
        "representation": name,
        "dimension": rep.dimension,
        "low": rep.low.tolist(),
        "high": rep.high.tolist(),
        "sobol_seed": SEARCH_SEEDS[name],
        "stage_a": stage_a,
        "stage_c_negative_confirmation": stage_c,
        "local_rule": {
            "normalized_radius": 0.03125,
            "axis_offsets": 2 * rep.dimension,
            "scrambled_sobol_offsets": 16,
            "sobol_seed": LOCAL_SEEDS[name],
            "clip_to_global_domain": True,
            "deduplicate": True,
        },
    }


def main() -> int:
    if (STUDY / "PREREGISTRATION.json").exists():
        raise FileExistsError(STUDY / "PREREGISTRATION.json")
    prior = json.loads((PRIOR / "per_episode_basin.json").read_text())["episodes"]
    positives = sorted([row for row in prior if row["basin_exists"]], key=lambda row: row["episode_id"])
    negatives = sorted([row for row in prior if not row["basin_exists"]], key=lambda row: row["episode_id"])
    rng = np.random.default_rng(SELECTION_SEED)
    selected_positive = [positives[i] for i in np.sort(rng.choice(len(positives), 4, replace=False))]
    selected_negative = [negatives[i] for i in np.sort(rng.choice(len(negatives), 8, replace=False))]
    prior_catalog = json.loads((PRIOR / "episode_catalog.json").read_text())
    controls = sorted(prior_catalog["controls"], key=lambda row: row["episode_id"])
    control_rng = np.random.default_rng(CONTROL_SEED)
    selected_controls = [controls[i] for i in np.sort(control_rng.choice(len(controls), 6, replace=False))]
    prior_by_id = {row["episode_id"]: row for row in prior_catalog["targets"] + prior_catalog["controls"]}
    pilot = []
    for stratum, rows in (("p0_positive", selected_positive), ("p0_negative", selected_negative)):
        for row in rows:
            pilot.append({**prior_by_id[row["episode_id"]], "pilot_stratum": stratum})
    pilot.extend({**row, "pilot_stratum": "baseline_success_control"} for row in selected_controls)

    state_rng = np.random.default_rng(STATE_SEED)
    state_anchors = []
    for episode in pilot:
        path = HARD / "trajectories/hard_safety" / episode["set"] / f"rollout_{episode['rollout_id']:03d}.npz"
        with np.load(path, allow_pickle=False) as data:
            action_count = len(data["executed_actions"])
        indices = np.sort(state_rng.choice(action_count, 3, replace=False))
        for ordinal, step in enumerate(indices):
            state_anchors.append({
                "state_id": f"{episode['episode_id']}|S{ordinal}",
                "episode_id": episode["episode_id"],
                "set": episode["set"],
                "rollout_id": episode["rollout_id"],
                "family_id": episode["family_id"],
                "seed": episode["seed"],
                "pilot_stratum": episode["pilot_stratum"],
                "source_step": int(step),
                "source_action_count": action_count,
            })

    designs = {
        "P0-3D": sample_rows("P0-3D", 255, 0),
        "P1-Agent6": sample_rows("P1-Agent6", 128, 128),
        "P2-Pair8": sample_rows("P2-Pair8", 192, 192),
        "P3-Temporal6": sample_rows("P3-Temporal6", 128, 128),
    }
    # P0 uses 255 Sobol points plus the exact A004 anchor, with no additional branch.
    designs["P0-3D"]["stage_a"] = designs["P0-3D"]["stage_a"][:255] + [designs["P0-3D"]["stage_a"][-1]]
    designs["P0-3D"]["stage_c_negative_confirmation"] = []
    design_path = STUDY / "parameter_designs.json"
    catalog_path = STUDY / "pilot_catalog.json"
    state_path = STUDY / "state_anchor_manifest.json"
    dump_new(design_path, designs)
    dump_new(catalog_path, {"selection_seed": SELECTION_SEED, "control_seed": CONTROL_SEED, "episodes": pilot})
    dump_new(state_path, {"state_seed": STATE_SEED, "anchors_per_episode": 3, "anchors": state_anchors})

    protocol = {
        "schema": "double_bottleneck_eta_representation_capacity_preregistration_v1",
        "registered_at": datetime.now().astimezone().isoformat(),
        "hypotheses": {
            "H0": "Prior 65-point P0 search missed narrow basins.",
            "H1": "Shared goal coefficient across agents is restrictive.",
            "H2": "Mean/shared relational coefficient loses pair-specific information.",
            "H3": "One coefficient vector fixed for the episode is restrictive.",
            "H4": "Second projection collapses raw parameter freedom.",
        },
        "frozen": {
            "checkpoint": "6e2ed4e31443bbb34457d3b3aabe0e7d741b904259391f8893b1104c143546bd",
            "dataset_manifest": "771b575f5641562a4b4631da02c70ac06fea993c29c1f2fb802b10e3d17c6c56",
            "macflow_source": "02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8",
            "environment": "3159b98f180f18d2f270d2b093e547d7d9f3c9f5b25347fb60d42b3ada149cdc",
            "hard_projection": "847f7045ffb617f403abb5af3a4edd092c70eef7734e819d42718c3391b0ff79",
            "canonical_3d": "48f73555d542d77581852450edb0d0c7d9c582ff9262d063384df88b08dadf40",
            "horizon_steps": 850,
            "success": "all four agents reach their respective 0.08 m goals before frozen runtime termination",
        },
        "parameter_domain_fairness": {
            "rule": "Every scalar coefficient uses its corresponding P0 interval. Pair8 keeps the 1/(N-1) sum, so per-row relation magnitude cannot exceed P0. Temporal6 is a convex interpolation inside P0 ranges. Agent6 changes assignment but not any row coefficient bound.",
            "energy_cap": None,
            "reason_no_extra_cap": "All P0/P1/P2/P3 representations retain the same 1.25 m/s analytical maximum raw correction-row bound.",
        },
        "pilot": {
            "targets": 12,
            "p0_positive": 4,
            "p0_negative": 8,
            "controls": 6,
            "selection": "uniform without replacement inside frozen P0-positive/P0-negative/control strata",
        },
        "p0_dense_gate": {
            "points": 256,
            "same_domain": True,
            "stop_capacity_expansion_if_new_negative_successes_at_least": 3,
            "reason": "3/8 new recoveries would materially overturn the old negative-case inference.",
        },
        "search": {
            "P1-Agent6": {"stage_a_sobol": 128, "p0_embedding": 1, "negative_confirmation_sobol": 128},
            "P2-Pair8": {"stage_a_sobol": 192, "p0_embedding": 1, "negative_confirmation_sobol": 192},
            "P3-Temporal6": {"stage_a_sobol": 128, "p0_embedding": 1, "negative_confirmation_sobol": 128},
            "same_samples_across_episodes": True,
            "no_domain_expansion": True,
            "local_center": "successful global point with smallest mean raw correction, then normalized distance to P0 embedding, then parameter_id",
        },
        "promotion": {
            "clear_capacity_gain": "At least +3/12 pilot timeout basin existence over dense P0 with median local success >=0.25, OR at least 25% median valid-state expert-fit residual reduction plus at least two newly recovered P0-negative pilot cases.",
            "full_population": "All 61 frozen safe timeouts plus all 24 preregistered baseline-success controls.",
            "tie_break": "smallest dimension, then higher pilot existence, local robustness, expert fit",
        },
        "p4_trigger": "Run only if P1 exceeds P0 by >=4 targets, exceeds both P2 and P3 by >=2, remains below 10/12, and has median expert-fit residual >=0.03 m/s.",
        "offline_audit": {
            "anchors": "exactly 3 uniformly sampled action-state indices per pilot episode; no phase rejection or resampling",
            "expert": "existing centralized expert enumerating all eight hypotheses; retain only validated successful continuation",
            "fit": "1024 scrambled Sobol candidates per representation plus bounded Powell refinement from the best candidate",
            "fit_seed": 830017,
            "finite_difference_normalized_step": 0.015625,
            "effective_rank_threshold": "max(1e-3, 0.01*sigma_max)",
            "sensitivity_points": "P0 embedding and per-state best fit",
        },
        "stochastic_robustness": {
            "new_seeds": [1103, 1201, 1301, 1409],
            "selection": "up to four uniformly indexed successful pilot episode/parameter centers per promoted representation",
            "same_seed_protocol_across_representations": True,
        },
        "prohibitions": ["G_phi training", "safety weakening", "horizon change", "mode or phase labels", "per-step free eta", "per-episode representation definitions", "test-triggered representation changes"],
        "artifact_hashes": {
            "prior_p0_results": sha(PRIOR / "per_episode_basin.json"),
            "parameter_designs": sha(design_path),
            "pilot_catalog": sha(catalog_path),
            "state_anchors": sha(state_path),
        },
    }
    dump_new(STUDY / "PREREGISTRATION.json", protocol)
    print(json.dumps({"pilot_episodes": len(pilot), "state_anchors": len(state_anchors), "designs": {k: len(v["stage_a"]) for k, v in designs.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
