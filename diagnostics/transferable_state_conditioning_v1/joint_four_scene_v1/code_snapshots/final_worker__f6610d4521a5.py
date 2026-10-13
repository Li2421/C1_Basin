"""Model-gated, exact-identity final Q16 rollout workers and anchor merge."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

from diagnostics.physical_state_critic_formal_v1 import loso_worker
from diagnostics.transferable_state_conditioning_v1.assemble_source import HERE as GLOBAL_HERE, FORMAL, read, sha
from diagnostics.transferable_state_conditioning_v1.prepare_final import EXPERIMENT
from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.rollout_db import canonical, connect, eta_identity, transaction, uid, ROOT as DB_ROOT

FORMAL_EXPERIMENT = loso_worker.EXPERIMENT
HERE = Path(os.environ.get("TRANSFER_Q_FINAL_ROOT", str(GLOBAL_HERE)))
FLAGS = ("success", "deadlock", "timeout", "collision", "numerical_failure")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False)+"\n")
    temp.replace(path)


def gate():
    protocol = read(HERE / "final_protocol.json")
    assert protocol["final_models_sha256"] == sha(HERE / "final_models_frozen.json")
    assert protocol["frozen_splits_sha256"] == sha(HERE / "frozen_splits.json")
    for name, digest in protocol["predictions_sha256"].items():
        panel, bank = name.split("/")
        assert sha(HERE / "predictions" / panel / f"{bank}.npz") == digest
    return protocol


def run(bank, index):
    assert bank in ("proposal", "anchor")
    gate()
    items = read(HERE / "final" / f"{bank}_work_items.json")
    assert 0 <= index < len(items)
    task = items[index]
    scene = task["scene"]
    panel = HERE / "final" / bank / scene
    cases = read(panel / "missing_cases.json")[task["start"]:task["end"]]
    assert cases
    proposals = read(FORMAL / "loso" / scene / "test" / "proposals.json")["states"]
    by_uid = {r["state_uid"]: r for r in proposals}
    source = {r["alias"]: r for r in read(FORMAL / f"states_{scene}.json")["states"]}
    controller = read(FORMAL / "loso_protocol.json")["controllers"][scene]
    runtime = loso_worker.make_runtime(scene)
    experiment = FORMAL_EXPERIMENT if bank == "proposal" else EXPERIMENT
    protocol_hash = sha(FORMAL / "loso_protocol.json") if bank == "proposal" else sha(HERE / "final_protocol.json")
    records = []
    new = 0
    for c in cases:
        assert c["scene"] == scene
        assert eta_identity(c["eta"])[0] == c["eta_uid"]
        assert uid("roll", {"state": c["state_uid"], "eta": c["eta_uid"],
                            "controller": c["controller_uid"], "seed": c["seed_key"]}) == c["rollout_uid"]
        proposal = by_uid[c["state_uid"]]
        physical = source[proposal["state_alias"]]["physical"]
        safety = bank == "proposal" and c["eta_index"] == 16
        expected = controller["safety_uid" if safety else "intervention_uid"]
        assert c["controller_uid"] == expected
        raw_path = (FORMAL / "loso" / scene / "test" / "raw" / f"{c['rollout_uid']}.json"
                    if bank == "proposal" else panel / "raw" / f"{c['rollout_uid']}.json")
        if raw_path.exists():
            record = read(raw_path)
            assert record["case"] == c and record["protocol_sha256"] == protocol_hash
        else:
            row = {"proposal": proposal, "source": source[proposal["state_alias"]],
                   "physical": physical}
            if scene in ("toy_give_way", "double_bottleneck"):
                raw = runtime.rollout(row, np.asarray(c["eta"], float), c["future_index"], safety)
            else:
                state = {"uid": c["state_uid"], "alias": proposal["state_alias"],
                         "index": c["state_index"], "physical": physical}
                raw = runtime.rollout(state, np.asarray(c["eta"], float), c["future_index"],
                                      "hard_safety_q16" if safety else "orthoflow3")
                assert raw["controller_uid"] == c["controller_uid"]
            assert raw["state_uid"] == c["state_uid"]
            assert raw["future_index"] == c["future_index"]
            assert eta_identity(raw["eta"])[0] == c["eta_uid"]
            record = {"schema": "formal_q_loso_seed_v1" if bank == "proposal" else "transferable_q_final_anchor_seed_v1",
                      "experiment_uid": experiment, "case": c,
                      "success": int(raw["success"]), "deadlock": int(raw["deadlock"]),
                      "timeout": int(raw["timeout"]), "collision": int(raw["collision"]),
                      "numerical_failure": int(raw["numerical_failure"]),
                      "episode_length": int(raw["episode_length"]),
                      "outcome": raw["outcome"], "raw": raw,
                      "protocol_sha256": protocol_hash}
            assert not (record["success"] and (record["collision"] or record["numerical_failure"]))
            write(raw_path, record)
            new += 1
        records.append(record)
    append_journal(records, experiment, f"final_{bank}_{index:04d}")
    receipt = {"bank": bank, "index": index, "scene": scene, "cases": len(cases),
               "new": new, "recovered": len(cases)-new, "protocol_sha256": protocol_hash,
               "complete": True}
    write(HERE / "final" / "receipts" / f"{bank}_{index:04d}.json", receipt)
    print(json.dumps(receipt), flush=True)


def merge_anchor():
    gate()
    digest = sha(HERE / "final_protocol.json")
    stats = Counter()
    with connect() as db:
        db.execute("INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,start_time,metadata_json) VALUES(?,?,?,?,CURRENT_TIMESTAMP,?)",
                   (EXPERIMENT, "Transferable Q final common-eta audit", str(HERE / "final" / "anchor"),
                    digest, canonical({"pure_Q": True, "final_confirmation": True})))
        db.commit()
        for path in sorted((DB_ROOT / "journals" / EXPERIMENT).glob("*.jsonl")):
            source_uid = uid("source", {"path": str(path), "sha256": sha(path)})
            if db.execute("SELECT 1 FROM source_file WHERE source_uid=?", (source_uid,)).fetchone():
                stats["already_merged"] += 1
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            with transaction(db):
                db.execute("INSERT INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen) VALUES(?,?,?,?,?,?,?)",
                           (source_uid, EXPERIMENT, str(path), sha(path), ".jsonl", "SEED_EXACT", len(rows)))
                inserted = duplicate = 0
                for line_number, rec in enumerate(rows, 1):
                    assert rec["experiment_uid"] == EXPERIMENT and rec["protocol_sha256"] == digest
                    c = rec["case"]
                    assert eta_identity(c["eta"])[0] == c["eta_uid"]
                    assert uid("roll", {"state": c["state_uid"], "eta": c["eta_uid"],
                                        "controller": c["controller_uid"], "seed": c["seed_key"]}) == c["rollout_uid"]
                    assert not (rec["success"] and (rec["collision"] or rec["numerical_failure"]))
                    old = db.execute("SELECT * FROM rollout WHERE rollout_uid=?", (c["rollout_uid"],)).fetchone()
                    if old:
                        assert tuple(rec[k] for k in FLAGS) == tuple(old[k] for k in FLAGS)
                        duplicate += 1
                    else:
                        db.execute("INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,outcome,experiment_uid,original_source_file,compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                   (c["rollout_uid"], c["state_uid"], c["eta_uid"], c["controller_uid"],
                                    c["seed_key"], c["seed_key"], *(rec[k] for k in FLAGS),
                                    rec["episode_length"], rec["outcome"], EXPERIMENT, str(path),
                                    "EXACT_REUSE", uid("record", rec)))
                        inserted += 1
                    db.execute("INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)",
                               (c["rollout_uid"], source_uid, line_number))
                db.execute("UPDATE source_file SET rows_imported=?,rows_duplicate=? WHERE source_uid=?",
                           (inserted, duplicate, source_uid))
                stats.update(journals=1, inserted=inserted, duplicate=duplicate)
        db.execute("UPDATE experiment SET new_rollout_count=? WHERE experiment_uid=?",
                   (db.execute("SELECT COUNT(*) FROM rollout WHERE experiment_uid=?", (EXPERIMENT,)).fetchone()[0], EXPERIMENT))
        db.commit()
    write(HERE / "final" / "anchor_merge_audit.json", dict(stats))
    print(dict(stats), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("run", "merge_anchor"))
    p.add_argument("--bank", choices=("proposal", "anchor"))
    p.add_argument("--index", type=int)
    a = p.parse_args()
    if a.action == "run":
        assert a.bank and a.index is not None
        run(a.bank, a.index)
    else:
        merge_anchor()
