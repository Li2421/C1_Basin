"""Quantify whether old strict LOSO TRAIN labels identified η interactions."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "orthoflow3_loso_root_cause_v1" / "joint" / "source_pairs.parquet"
STATES = SOURCE.parent / "source_states.json"


def main():
    rows = pq.read_table(SOURCE).to_pylist()
    formal_controllers = json.loads((HERE.parent / "physical_state_critic_formal_v1" / "loso_protocol.json").read_text())["controllers"]
    scenario_name = {"toy_giveway": "toy_give_way", "double_bottleneck": "double_bottleneck",
                     "four_way_intersection": "four_way_intersection", "ring_exchange": "ring_exchange"}
    result = {"schema": "historical_strict_loso_source_eta_signal_v1",
              "source": str(SOURCE), "criteria": {
                  "informative_state": "within-state observed q range >= 0.25",
                  "fully_separated_state": "at least one eta q <= 0.25 and another eta q >= 0.75"},
              "by_scene_split": {}, "state_phase": {}, "controller_identity": {}}
    for old_scene,new_scene in scenario_name.items():
        uids=sorted({r["controller_uid"] for r in rows if r["scenario"]==old_scene})
        result["controller_identity"][old_scene] = {
            "historical_controller_uids": uids,
            "new_formal_intervention_uid": formal_controllers[new_scene]["intervention_uid"],
            "exact_controller_uid_overlap": formal_controllers[new_scene]["intervention_uid"] in uids}
    states = json.loads(STATES.read_text())
    for scene in sorted({r["scenario"] for r in states}):
        f = np.asarray([r["physical"].get("remaining_fraction", 1.) for r in states
                        if r["scenario"] == scene], float)
        result["state_phase"][scene] = {
            "independent_states": len(f), "remaining_fraction_min": float(f.min()),
            "remaining_fraction_median": float(np.median(f)),
            "remaining_fraction_max": float(f.max()),
            "fraction_intermediate_lt_0_99": float(np.mean(f < .99))}
    for scene in sorted({r["scenario"] for r in rows}):
        for split in ("train", "validation"):
            rr = [r for r in rows if r["scenario"] == scene and r["split"] == split]
            by = defaultdict(list)
            for r in rr:
                by[r["state_uid"]].append(float(r["q"]))
            q = np.asarray([r["q"] for r in rr], float)
            span = np.asarray([max(x)-min(x) for x in by.values()])
            result["by_scene_split"][f"{scene}/{split}"] = {
                "independent_states": len(by), "state_eta_pairs": len(rr),
                "state_eta_pairs_per_state_mean": len(rr)/len(by),
                "success_q_mean": float(q.mean()), "success_q_sd": float(q.std()),
                "informative_states_ge_025": int(np.sum(span >= .25)),
                "fully_separated_states": int(sum(min(x) <= .25 and max(x) >= .75 for x in by.values())),
                "within_state_q_span_median": float(np.median(span)),
                "within_state_q_span_max": float(np.max(span))}
    target = HERE / "historical_source_signal_audit.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    for name, row in result["by_scene_split"].items():
        print(name, row)


if __name__ == "__main__":
    main()
