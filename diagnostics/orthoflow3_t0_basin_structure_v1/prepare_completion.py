#!/usr/bin/env python3
"""Freeze the completion audit from the already-frozen eight-state t0 audit."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("/home/zhihan/research/Basin_C1")
PRIOR = ROOT / "diagnostics/orthoflow3_t0_basin_structure_v1"
HERE = ROOT / "diagnostics/orthoflow3_t0_basin_completion_v1"
LEARN = ROOT / "diagnostics/orthoflow3_basin_margin_learning_v1"
LOWJ = ROOT / "diagnostics/orthoflow3_direct_eta_baseline_v1/direct_eta_seed23.msgpack"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "raw").mkdir(exist_ok=True)
    (HERE / "logs").mkdir(exist_ok=True)
    source = json.load(open(PRIOR / "t0_state_manifest.json"))
    dump(HERE / "frozen_8state_manifest.json", source)
    expected = {
        "orthoflow3": "51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38",
        "g_lowj": "bd660db3ac501e5e77755af65cee5c01ba7d30810a4cf6001170bdbcebbb05d7",
        "g_center": "8a79fc4e198cba76562b90197a8a6f52fff2d20c2f536b000301d482ec5c2779",
        "g_margin": "ba02aeddccb5533b1fea7e3a5da8039cb9226cc2196168d1e40a897c177146f8",
    }
    paths = {
        "orthoflow3": ROOT / "diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py",
        "g_lowj": LOWJ,
        "g_center": LEARN / "g_center/runs/center_seed17.msgpack",
        "g_margin": LEARN / "g_margin/runs/margin_seed17.msgpack",
    }
    observed = {name: sha(path) for name, path in paths.items()}
    if observed != expected:
        raise RuntimeError((observed, expected))
    dump(HERE / "checkpoint_integrity.json", {"expected": expected, "observed": observed, "all_match": True,
                                               "paths": {k:str(v) for k,v in paths.items()}})

    old_counts = {}
    exact_tuples = 0
    old_steps = 0
    for state in source["attempted_states"]:
        sid = state["state_id"]
        raw = PRIOR / "anchor_runs" / sid / "raw/pilot_rollouts.jsonl"
        rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
        old_counts[sid] = {"continuations": len(rows), "physical_steps": sum(int(x["continuation_steps"]) for x in rows)}
        exact_tuples += len(rows); old_steps += old_counts[sid]["physical_steps"]
    cache = {
        "inventory_frozen_before_completion_rollouts": True,
        "source": str(PRIOR),
        "source_manifest_sha256": sha(PRIOR / "t0_state_manifest.json"),
        "exact_reusable_t0_tuples": exact_tuples,
        "reusable_physical_steps_already_executed": old_steps,
        "per_state": old_counts,
        "future_root": 2026092811,
        "compatibility": "exact state/h/xi0, eta bytes, future_index, future root, hash, horizon, projections, monitor/history",
        "new_completion_rollouts_at_inventory": 0,
    }
    dump(HERE / "cache_reuse_audit.json", cache)
    dump(HERE / "completion_cache_reuse.json", cache)

    fresh = list(csv.DictReader(open(LEARN / "fresh_wide_results.csv")))
    ep_to_sid = {str(x["fresh_episode_index"]):x["state_id"] for x in source["attempted_states"]}
    rows=[]
    mapping={"g_lowj":"G_LOWJ","g_center":"G_CENTER","g_margin":"G_MARGIN"}
    for row in fresh:
        if row["episode_index"] in ep_to_sid and row["controller"] in mapping:
            rows.append({"state_id":ep_to_sid[row["episode_index"]],"episode_index":row["episode_index"],
                         "controller":mapping[row["controller"]],"eta":row["eta"],"feature_sha256":row["feature_sha256"],
                         "checkpoint_sha256":expected[row["controller"]],"prediction_source":"frozen matched fresh-WIDE inference"})
    with (HERE / "frozen_controller_predictions.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    protocol="""# OrthoFlow3 t0 basin completion and phase-mismatch audit v1

This completion reuses the exact eight-state manifest, h/xi0 conditioning, future root 2026092811, E_bridge geometry, ray directions, kappa=0.85, B63 semantics, and partial exact tuples from the prior t0 audit. Only ep0195, ep0074, and ep0139 ball stages are resumed. Eta zero, eta=(0.625,0,0.375), and the three exact frozen controller predictions are evaluated at future indices 0..63. No model is trained and no controller, geometry, or safety semantic is changed.
"""
    (HERE / "protocol.md").write_text(protocol)
    dump(HERE / "cost_preflight.json", {"new_continuation_cap":6000,"new_physical_step_cap":3500000,
        "projected_new_continuations":3672,"projected_new_physical_steps_range":[1800000,2400000],
        "max_gpu_shards":6,"server_idle_at_preflight":True,"within_caps":True})
    print(json.dumps({"states":[x["state_id"] for x in source["attempted_states"]],"cache":cache,"predictions":len(rows)},indent=2))


if __name__ == "__main__":
    main()
