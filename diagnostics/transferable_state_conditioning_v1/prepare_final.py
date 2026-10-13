"""Gate final rollout requests on frozen source-only models and predictions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from diagnostics.transferable_state_conditioning_v1.assemble_source import HERE as GLOBAL_HERE, FORMAL, SCENES, read, sha

HERE = Path(os.environ.get("TRANSFER_Q_FINAL_ROOT", str(GLOBAL_HERE)))
EXPERIMENT = os.environ.get("TRANSFER_Q_FINAL_EXPERIMENT", "exp_transferable_q_final_anchor_20261008_v1")


def freeze(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"
    if path.exists():
        assert path.read_text() == payload, f"frozen final artifact changed: {path}"
    else:
        path.write_text(payload)


def prepare():
    split_path = HERE / "frozen_splits.json"
    split = read(split_path)
    models_path = HERE / "final_models_frozen.json"
    models = read(models_path)
    assert models["final_outcomes_opened"] is False
    assert models["frozen_splits_sha256"] == sha(split_path)
    assert set(models["panels"]) == {f"within_{s}" for s in SCENES} | {f"loso_{s}" for s in SCENES}
    predictions = {}
    for panel in models["panels"]:
        for bank in ("proposal", "anchor"):
            p = HERE / "predictions" / panel / f"{bank}.npz"
            a = read(p.parent / f"{bank}_audit.json")
            assert sha(p) == a["predictions_sha256"]
            assert a["final_models_frozen_sha256"] == sha(models_path)
            assert a["final_outcomes_opened"] is False
            predictions[f"{panel}/{bank}"] = sha(p)
    result = {"schema": "transferable_q_final_rollout_gate_v1",
              "final_models_sha256": sha(models_path), "frozen_splits_sha256": sha(split_path),
              "predictions_sha256": predictions, "final_outcomes_opened_before_freeze": False,
              "panels": {}, "experiment_uid_anchor": EXPERIMENT}
    work = {"proposal": [], "anchor": []}
    for bank in ("proposal", "anchor"):
        for scene in SCENES:
            panel = HERE / "final" / bank / scene
            if bank == "proposal":
                original = read(FORMAL / "loso" / scene / "test" / "cases.json")
                eligible = set(split["final_confirmation"][scene]["state_uids"])
                cases = [c for c in original if c["state_uid"] in eligible]
                assert len(cases) == len(eligible)*17*16
                # Preserve the exact formal identities, including Toy rollout_id.
                assert len({c["rollout_uid"] for c in cases}) == len(cases)
                source = FORMAL / "loso" / scene / "test" / "cases.json"
            else:
                source = GLOBAL_HERE / "anchor" / scene / "cases.json"
                eligible = set(split["anchor_subset"][scene]["state_uids"])
                cases = [c for c in read(source) if c["state_uid"] in eligible]
                assert len(cases) == len(eligible)*16*16
            freeze(panel / "cases.json", cases)
            grouped = {}
            for c in cases:
                k = (c["state_uid"], c["eta_uid"], c["controller_uid"])
                grouped.setdefault(k, []).append(c["seed_key"])
            requests = [dict(state_uid=s, eta_uid=e, controller_uid=c,
                             seed_keys=seeds) for (s,e,c),seeds in grouped.items()]
            assert all(len(r["seed_keys"]) == 16 for r in requests)
            freeze(panel / "planned_rollouts.json", dict(requests=requests))
            subprocess.run([sys.executable, "-m", "shared_rollout_db.plan",
                            "--manifest", str(panel / "planned_rollouts.json"),
                            "--output", str(panel / "cache_preflight.json")],
                           check=True, stdout=(panel / "preflight_stdout.json").open("w"))
            preflight = read(panel / "cache_preflight.json")
            assert not preflight["summary"]["ambiguous"]
            missing = {(r["state_uid"], r["eta_uid"], r["controller_uid"], seed)
                       for r in preflight["details"] for seed in r["missing_seeds"]}
            selected = [c for c in cases if (c["state_uid"], c["eta_uid"], c["controller_uid"], c["seed_key"]) in missing]
            assert len(selected) == preflight["summary"]["genuinely_missing"]
            freeze(panel / "missing_cases.json", selected)
            for start in range(0, len(selected), 128):
                work[bank].append({"scene": scene, "start": start,
                                   "end": min(start+128, len(selected))})
            result["panels"][f"{bank}/{scene}"] = {
                "source_cases_sha256": sha(source), "requests": len(cases),
                "reused_seed_slots": len(cases)-len(selected),
                "new_seed_slots": len(selected), "cases_sha256": sha(panel/"cases.json"),
                "preflight_sha256": sha(panel/"cache_preflight.json")}
    for bank in work:
        freeze(HERE / "final" / f"{bank}_work_items.json", work[bank])
    freeze(HERE / "final_protocol.json", result)
    print(json.dumps({"final_protocol_sha256": sha(HERE/"final_protocol.json"),
                      "proposal_shards": len(work["proposal"]),
                      "anchor_shards": len(work["anchor"]),
                      "panels": result["panels"]}, indent=2))


if __name__ == "__main__":
    prepare()
