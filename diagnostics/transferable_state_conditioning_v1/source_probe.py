"""Exact source-only broad-η Q4 probe using the frozen FF runtime."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from diagnostics.physical_state_critic_formal_v1 import loso_worker
from diagnostics.transferable_state_conditioning_v1.assemble_source import HERE, FORMAL, SCENES, read, sha
from diagnostics.transferable_state_conditioning_v1.prepare_source_probe import EXPERIMENT
from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.rollout_db import canonical, connect, eta_identity, transaction, uid, ROOT as DB_ROOT

PROTOCOL = HERE / "source_anchor_probe_counts.json"
FLAGS = ("success", "deadlock", "timeout", "collision", "numerical_failure")
PROBE_ORDER = ("double_bottleneck", "four_way_intersection", "ring_exchange", "toy_give_way")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False)+"\n")
    temp.replace(path)


def run(index):
    assert 0 <= index < 32
    scene = PROBE_ORDER[index // 8]
    shard = index % 8
    panel = HERE / "source_anchor_probe" / scene
    cases = read(panel / "cases.json")[shard*128:(shard+1)*128]
    assert len(cases) == 128
    proposals = read(FORMAL / "loso" / scene / "train" / "proposals.json")["states"][:16]
    source = {r["alias"]: r for r in read(FORMAL / f"states_{scene}.json")["states"]}
    runtime = loso_worker.make_runtime(scene)
    records = []
    new = 0
    for c in cases:
        assert c["scene"] == scene and c["state_index"] < 16
        assert c["state_uid"] == proposals[c["state_index"]]["state_uid"]
        assert eta_identity(c["eta"])[0] == c["eta_uid"]
        target = panel / "raw" / f"{c['rollout_uid']}.json"
        if target.exists():
            rec = read(target)
            assert rec["case"] == c and rec["protocol_sha256"] == sha(PROTOCOL)
        else:
            p = proposals[c["state_index"]]
            original = source[c["state_alias"]]
            row = {"proposal": p, "source": original, "physical": original["physical"]}
            if scene in ("toy_give_way", "double_bottleneck"):
                raw = runtime.rollout(row, np.asarray(c["eta"], float), c["future_index"], False)
            else:
                state = {"uid": c["state_uid"], "alias": c["state_alias"],
                         "index": c["state_index"], "physical": original["physical"]}
                raw = runtime.rollout(state, np.asarray(c["eta"], float), c["future_index"], "orthoflow3")
                assert raw["controller_uid"] == c["controller_uid"]
            assert raw["state_uid"] == c["state_uid"]
            assert raw["future_index"] == c["future_index"]
            assert eta_identity(raw["eta"])[0] == c["eta_uid"]
            rec = {"schema": "transferable_q_source_anchor_seed_v1",
                   "experiment_uid": EXPERIMENT, "case": c,
                   "success": int(raw["success"]), "deadlock": int(raw["deadlock"]),
                   "timeout": int(raw["timeout"]), "collision": int(raw["collision"]),
                   "numerical_failure": int(raw["numerical_failure"]),
                   "episode_length": int(raw["episode_length"]),
                   "outcome": raw["outcome"], "raw": raw,
                   "protocol_sha256": sha(PROTOCOL)}
            assert not (rec["success"] and (rec["collision"] or rec["numerical_failure"]))
            write(target, rec)
            new += 1
        records.append(rec)
    append_journal(records, EXPERIMENT, f"source_anchor_{index:03d}")
    receipt = {"index": index, "scene": scene, "shard": shard,
               "cases": len(cases), "new": new, "recovered": len(cases)-new,
               "protocol_sha256": sha(PROTOCOL), "complete": True}
    write(HERE / "source_anchor_probe" / "receipts" / f"{index:03d}.json", receipt)
    print(json.dumps(receipt), flush=True)


def merge():
    protocol_hash = sha(PROTOCOL)
    stats = Counter()
    with connect() as db:
        db.execute("INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,start_time,metadata_json) VALUES(?,?,?,?,CURRENT_TIMESTAMP,?)",
                   (EXPERIMENT, "Transferable Q source-only broad-eta probe", str(HERE/"source_anchor_probe"),
                    protocol_hash, canonical({"pure_Q": True, "source_only": True})))
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
                    assert rec["experiment_uid"] == EXPERIMENT and rec["protocol_sha256"] == protocol_hash
                    c = rec["case"]
                    assert eta_identity(c["eta"])[0] == c["eta_uid"]
                    identity = {"state": c["state_uid"], "eta": c["eta_uid"],
                                "controller": c["controller_uid"], "seed": c["seed_key"]}
                    assert uid("roll", identity) == c["rollout_uid"]
                    assert not(rec["success"] and (rec["collision"] or rec["numerical_failure"]))
                    assert db.execute("SELECT 1 FROM state WHERE state_uid=?", (c["state_uid"],)).fetchone()
                    assert db.execute("SELECT 1 FROM eta WHERE eta_uid=?", (c["eta_uid"],)).fetchone()
                    assert db.execute("SELECT 1 FROM controller_config WHERE controller_uid=?", (c["controller_uid"],)).fetchone()
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
    write(HERE / "source_anchor_probe" / "merge_audit.json", dict(stats))
    print(dict(stats), flush=True)


def signal():
    result = {"schema": "source_only_broad_eta_q4_signal_v1", "scenes": {}}
    with connect(True) as db:
        for scene in SCENES:
            cases = read(HERE / "source_anchor_probe" / scene / "cases.json")
            success = np.full((16,16,4), -1, np.int8)
            unknown = np.full((16,16,4), -1, np.int8)
            collision = np.full((16,16,4), -1, np.int8)
            for c in cases:
                row = db.execute("SELECT success,numerical_failure,collision FROM rollout WHERE rollout_uid=? AND conflict_quarantined=0",
                                 (c["rollout_uid"],)).fetchone()
                assert row is not None, c["rollout_uid"]
                i,j,k = c["state_index"], c["eta_index"], c["future_index"]
                success[i,j,k], unknown[i,j,k], collision[i,j,k] = row
            assert (success>=0).all() and (unknown>=0).all() and (collision>=0).all()
            q = success.sum(2)/4.
            span = q.max(1)-q.min(1)
            result["scenes"][scene] = {"states": 16, "pairs": 256, "rollouts": 1024,
                                       "overall_success": float(success.mean()),
                                       "informative_state_ge_025": int(np.sum(span>=.25)),
                                       "informative_state_ge_050": int(np.sum(span>=.5)),
                                       "full_success_eta_fraction": float(np.mean(q==1)),
                                       "within_state_span": span.tolist(),
                                       "success_counts": success.sum(2).tolist(),
                                       "numerical_failures": int(unknown.sum()),
                                       "collisions": int(collision.sum())}
    target = HERE / "source_anchor_probe" / "signal.json"
    target.write_text(json.dumps(result, indent=2)+"\n")
    for scene, r in result["scenes"].items():
        print(scene, {k:r[k] for k in ("informative_state_ge_025","informative_state_ge_050","overall_success","numerical_failures","collisions")})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("run","merge","signal"))
    p.add_argument("--index", type=int)
    a = p.parse_args()
    if a.action == "run":
        assert a.index is not None
        run(a.index)
    elif a.action == "merge": merge()
    else: signal()
