"""Outcome-blind identity/count audit. Never indexes success or unknown arrays."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
D = ROOT / "diagnostics"
SPECS = {
    "T1": (D / "ring_independent_flow_confirmation_v2", "R5_FINAL_FEATURES.npz"),
    "T2": (D / "controller_universal_eta_schedule_v1", None),
    "T3": (D / "controller_new_checkpoint_confirmation_v1/fp32_n8", "R5_FINAL_FEATURES.npz"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype)} for k in z.files}


def main():
    source = D / "ring_controller_population_v1"
    rows = [json.loads(x) for x in (source / "dataset.jsonl").read_text().splitlines()]
    keys = [(r["controller"], r["state_uid"], r["arm"], r["future_seed"]) for r in rows]
    states = json.loads((source / "FRESH_STATES.json").read_text())
    source_names = sorted({r["controller"] for r in rows})
    recipes = sorted({r["recipe"] for r in rows})
    source_arms = sorted({r["arm"] for r in rows})
    with np.load(source / "R5_FEATURES.npz", allow_pickle=False) as z:
        r5_names = list(z["controller_names"])
    with np.load(source / "R6_FEATURES.npz", allow_pickle=False) as z:
        r6_names = list(z["controller_names"])
    out = {
        "audit_scope": "identity, count, shape, alignment only; external success/unknown values unopened",
        "historical_evidence": {
            "round15": "same eight-arm semantics, complete source labels; reuse dataset and probes",
            "round5_6": "old fixed-eta K16 critic labels incompatible with current eight-arm selection; reuse architecture/physical23 only",
            "round10_12_14A": "external labels and R5 already exist; no rollout/probe needed",
        },
        "source": {
            "rows": len(rows), "unique_keys": len(set(keys)), "controllers": len(source_names),
            "recipes": len(recipes), "recipe_names": recipes,
            "state_count": len(states), "unique_state_uids": len({s["state_uid"] for s in states}),
            "arms": source_arms, "seeds": sorted({r["future_seed"] for r in rows}),
            "r5_metadata": metadata(source / "R5_FEATURES.npz"),
            "r6_metadata": metadata(source / "R6_FEATURES.npz"),
            "r5_name_match": set(source_names) == set(r5_names),
            "r6_name_match": set(source_names) == set(r6_names),
            "state_uid_match": {r["state_uid"] for r in rows} == {s["state_uid"] for s in states},
        },
        "external": {},
        "input_sha256": {},
    }
    assert len(rows) == len(set(keys)) == 56 * 32 * 8 * 4
    assert len(recipes) == 14 and out["source"]["r5_name_match"] and out["source"]["r6_name_match"]
    t2_repr = D / "controller_behavior_repr_v1/FEATURES.npz"
    with np.load(t2_repr, allow_pickle=False) as z:
        t2_names = set(z["controller_names"])
    for tag, (folder, r5_file) in SPECS.items():
        labels = folder / "LABELS.npz"
        with np.load(labels, allow_pickle=False) as z:
            names = list(z["controller_names"] if "controller_names" in z.files else z["controllers"])
            arms = list(z["arms"])
            available_shape = list(z["available"].shape) if "available" in z.files else None
        test_states = json.loads((folder / "FRESH_STATES.json").read_text())
        if r5_file:
            with np.load(folder / r5_file, allow_pickle=False) as z:
                feature_names = set(z["controller_names"])
        else:
            feature_names = t2_names
        out["external"][tag] = {
            "directory": str(folder.relative_to(ROOT)), "labels_metadata": metadata(labels),
            "controllers": len(names), "unique_controllers": len(set(names)),
            "states": len(test_states), "unique_state_uids": len({s["state_uid"] for s in test_states}),
            "arms": arms, "r5_coverage": len(set(names) & feature_names),
            "r6_feature_file_present": any(folder.glob("*R6*FEATURES*.npz")),
            "available_shape": available_shape,
        }
        assert len(names) == len(set(names)) and len(test_states) == 64 and set(names) <= feature_names
        for p in (labels, folder / "FRESH_STATES.json"):
            out["input_sha256"][str(p.relative_to(ROOT))] = sha(p)
    for p in (source / "dataset.jsonl", source / "R5_FEATURES.npz", source / "R6_FEATURES.npz", source / "FRESH_STATES.json", t2_repr):
        out["input_sha256"][str(p.relative_to(ROOT))] = sha(p)
    (OUT / "DATA_AUDIT.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"source_rows": len(rows), "external": {k: {"controllers": v["controllers"], "states": v["states"], "arms": v["arms"]} for k, v in out["external"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
