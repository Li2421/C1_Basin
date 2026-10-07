"""Freeze the one-step causal audit set before observing branch results."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


DIAG = Path("/home/zhihan/research/Basin_C1/diagnostics")
HERE = Path(__file__).resolve().parent
V4 = DIAG / "gphi_training_dataset_v4"

# These are the controls pre-registered by the earlier oracle confidence audit.
# Keeping the complete small control set avoids any result-dependent selection.
EASY_CONTROLS = (
    "N_r066_s107",
    "N_r143_s105",
    "N_r151_s124",
    "R_D2_s95101009_p112",
    "P_r198_m120",
    "Q_pair228_m120",
    "R_D1_s95106017_p018",
    "R_D4_s95101013_p027",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    hard_path = DIAG / "hard_stable_boundary_crossval/difficult_stable_states.csv"
    hard = list(csv.DictReader(hard_path.open()))
    if len(hard) != 13:
        raise AssertionError(f"expected 13 frozen difficult states, got {len(hard)}")
    audited_path = DIAG / "oracle_boundary_confidence_audit/audited_states.csv"
    audited = {row["state_id"]: row for row in csv.DictReader(audited_path.open())}
    manifest_path = V4 / "state_manifest.jsonl"
    manifests = {row["state_id"]: row for row in map(json.loads, manifest_path.read_text().splitlines())}

    label_info = {}
    for version in ("v2", "v3", "v4"):
        path = DIAG / f"gphi_training_dataset_{version}/label_statistics.json"
        obj = read_json(path)
        label_info.update(obj.get("per_state", {}))
        label_info.update(obj.get("new_states", {}))

    hard_ids = [row["state_id"] for row in hard]
    selected = hard_ids + list(EASY_CONTROLS)
    if len(selected) != len(set(selected)) or len(selected) != 21:
        raise AssertionError("audit state set must contain 21 unique states")

    rows = []
    for state_id in selected:
        if state_id not in manifests or state_id not in audited or state_id not in label_info:
            raise KeyError((state_id, state_id in manifests, state_id in audited, state_id in label_info))
        meta = manifests[state_id]
        confidence = audited[state_id]
        labels = label_info[state_id]
        y_long = int(confidence["original_gate_label"])
        eta = [float(value) for value in labels["eta_best"]]
        if (y_long == 0) != (eta == [0.0, 0.0, 0.0]):
            raise AssertionError((state_id, y_long, eta))
        state_path = V4 / meta["state_file"]
        if sha(state_path) != meta["state_sha256"]:
            raise AssertionError(f"state hash mismatch: {state_id}")
        rows.append({
            "state_id": state_id,
            "set_role": "DIFFICULT_STABLE" if state_id in hard_ids else confidence["audit_tags"],
            "category": confidence["category"],
            "source_group": confidence["source_group"],
            "source_trajectory": confidence["source_trajectory"],
            "state_file": str(state_path),
            "state_sha256": meta["state_sha256"],
            "step": int(meta["step"]),
            "rng_namespace": int(meta["rng_namespace"]),
            "y_long": y_long,
            "eta_best": eta,
            "Q0_enlarged": float(confidence["p_hat_Q0"]),
            "Q0_wilson95_lower": float(confidence["wilson95_lower"]),
            "Q0_wilson95_upper": float(confidence["wilson95_upper"]),
        })

    protocol = read_json(V4 / "protocol.json")
    plan = {
        "study": "one_step_intervention_necessity",
        "selection_frozen_before_branch_results": True,
        "selection": {
            "difficult_stable": hard_ids,
            "easy_controls": list(EASY_CONTROLS),
            "control_source": str(audited_path),
        },
        "semantics": {
            "branch_I": "eta_best active at current and all future physical steps",
            "branch_N": "eta=(0,0,0) at current physical step only; eta_best active from t+1",
            "future_policy": "identical frozen eta_best DiagnosticCorrector plus the same FlowBC/two projections in both branches from t+1",
            "future_policy_approximation": "closest existing oracle-consistent continuation; not a receding oracle re-query",
            "matched_randomness": "same seed, rng_namespace, and absolute-step Flow key in both branches",
        },
        "states": rows,
        "seeds_initial": [int(value) for value in protocol["oracle_seed_default"]],
        "environment": protocol["environment"],
        "checkpoint": protocol["checkpoint"],
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "frozen_input_hashes": {
            "difficult_stable_states.csv": sha(hard_path),
            "audited_states.csv": sha(audited_path),
            "state_manifest.jsonl": sha(manifest_path),
            "v4_protocol.json": sha(V4 / "protocol.json"),
        },
    }
    (HERE / "audit_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({
        "states": len(rows),
        "difficult": len(hard_ids),
        "easy_zero": sum(row["set_role"] == "EASY_GATE0_CONTROL" for row in rows),
        "easy_nonzero": sum(row["set_role"] == "EASY_GATE1_CONTROL" for row in rows),
        "audit_plan_sha256": sha(HERE / "audit_plan.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
