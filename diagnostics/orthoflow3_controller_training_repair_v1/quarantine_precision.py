"""Preserve, but never reuse, records from the accidental x64 task executor.

Raw immutable journals remain untouched. Wrongly registered canonical keys are
relocated to explicit QUARANTINED controller profiles and audited, not lost or
overwritten. Correct native retries must use the original exact controller key.
"""
from collections import Counter
import json
from .data import OUT, EXPERIMENT, REVISION, DBROOT, read, write, digest, old
from shared_rollout_db.src.rollout_db import connect, canonical, transaction, uid, eta_identity
from pathlib import Path


def main():
    if (OUT / "precision_quarantine.json").exists():
        print(json.dumps({"already_completed": True, **read(OUT / "precision_quarantine.json")["counts"]}))
        return
    protocol = read(OUT / "protocol.json")
    bad_experiment = EXPERIMENT + "_precision_quarantine"
    counts = Counter()
    mapping = []
    with connect() as db, transaction(db):
        db.execute("""INSERT OR IGNORE INTO experiment(experiment_uid,name,path,metadata_json)
            VALUES(?,?,?,?)""", (bad_experiment, "accidental_x64_sampler_preserved_excluded",
                                 str(OUT / "precision_quarantine"), canonical({"eligible_for_training": False, "reason": "import-time precision mismatch"})))
        profiles = {}
        for profile in protocol["profiles"]:
            original = dict(db.execute("SELECT * FROM controller_config WHERE controller_uid=?", (profile["controller_uid"],)).fetchone())
            cfg = json.loads(original["config_json"])
            cfg.update(flow_sampler_precision="float64_import_side_effect", recovery_status="QUARANTINED_NOT_NATIVE_SEEDS")
            new_uid = uid("ctl", cfg)
            original.update(controller_uid=new_uid, config_json=canonical(cfg), compatibility_quality="AMBIGUOUS")
            keys = list(original)
            db.execute(f"INSERT OR IGNORE INTO controller_config({','.join(keys)}) VALUES({','.join('?' for _ in keys)})", tuple(original.values()))
            assert db.execute("SELECT 1 FROM controller_config WHERE controller_uid=?", (new_uid,)).fetchone()
            profiles[profile["controller_uid"]] = new_uid
        paths = [p for p in sorted((DBROOT / "journals" / EXPERIMENT).glob("*.jsonl")) if not p.name.startswith(REVISION)]
        for path in paths:
            source = uid("src", {"path": str(path.resolve())})
            lines = path.read_text().splitlines()
            db.execute("INSERT OR IGNORE INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen) VALUES(?,?,?,?,?,?,?)",
                       (source, bad_experiment, str(path), old.digest(path), ".jsonl", "AMBIGUOUS_PRECISION", len(lines)))
            db.execute("UPDATE source_file SET experiment_uid=?, classification='AMBIGUOUS_PRECISION' WHERE source_uid=?", (bad_experiment, source))
            for number, line in enumerate(lines, 1):
                r = json.loads(line)["record"]
                cid = r["controller_uid"]
                assert cid in profiles
                eu = eta_identity(r["eta"])[0]
                sk = canonical({"future_index": r["future_index"]})
                old_rid = uid("roll", {"state": r["state_uid"], "eta": eu, "controller": cid, "seed": sk})
                new_rid = uid("roll", {"state": r["state_uid"], "eta": eu, "controller": profiles[cid], "seed": sk})
                previous = db.execute("SELECT * FROM rollout WHERE rollout_uid=?", (old_rid,)).fetchone()
                if previous:
                    assert previous["experiment_uid"] == EXPERIMENT, "Do not move an unrelated historical record"
                    assert previous["raw_record_hash"] == digest(r), "Do not relocate a corrected retry"
                    saved = dict(previous)
                else:
                    core = {k: int(r[k]) for k in ("success", "deadlock", "timeout", "collision", "numerical_failure")}
                    saved = dict(rollout_uid=new_rid, state_uid=r["state_uid"], eta_uid=eu, controller_uid=profiles[cid],
                                 seed_key=sk, continuation_seed_json=sk, **core, episode_length=r.get("episode_length"),
                                 j_def=r.get("J_def"), min_wall_distance=r.get("minimum_wall_clearance"),
                                 min_agent_distance=r.get("minimum_agent_clearance"), outcome=r.get("outcome"),
                                 original_source_file=str(path), timestamp=r.get("timestamp"), raw_record_hash=digest(r))
                saved.update(rollout_uid=new_rid, controller_uid=profiles[cid], experiment_uid=bad_experiment,
                             compatibility_quality="AMBIGUOUS_PRECISION", conflict_quarantined=1)
                keys = list(saved)
                db.execute(f"INSERT OR IGNORE INTO rollout({','.join(keys)}) VALUES({','.join('?' for _ in keys)})", tuple(saved.values()))
                if previous:
                    db.execute("INSERT OR IGNORE INTO rollout_source SELECT ?,source_uid,source_line FROM rollout_source WHERE rollout_uid=?", (new_rid, old_rid))
                    db.execute("DELETE FROM rollout_source WHERE rollout_uid=?", (old_rid,))
                    db.execute("DELETE FROM rollout WHERE rollout_uid=? AND experiment_uid=?", (old_rid, EXPERIMENT))
                    counts["relocated_from_incorrect_canonical_keys"] += 1
                db.execute("INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)", (new_rid, source, number))
                counts["preserved_records"] += 1
                mapping.append({"old_declared_key": old_rid, "quarantined_key": new_rid, "journal": str(path)})
        db.execute("UPDATE experiment SET new_rollout_count=(SELECT COUNT(*) FROM rollout WHERE experiment_uid=?) WHERE experiment_uid=?", (bad_experiment, bad_experiment))
        db.execute("UPDATE experiment SET new_rollout_count=(SELECT COUNT(*) FROM rollout WHERE experiment_uid=?) WHERE experiment_uid=?", (EXPERIMENT, EXPERIMENT))
    write(OUT / "precision_quarantine.json", {"reason": "New repair data.py imported h20_features, whose dependency set global x64=True; native rollout did not reset it",
        "all_affected_workers_stopped_before_recovery": True, "raw_journals_unchanged": True,
        "outcomes_not_discarded": True, "excluded_from_training_and_native_cache": True,
        "counts": dict(counts), "profile_mapping": profiles, "record_mapping": mapping})
    print(json.dumps(dict(counts)))


def archive():
    """Prevent a future generic merger from re-ingesting the invalid envelopes."""
    audit = read(OUT / "precision_quarantine.json")
    destination = DBROOT / "journals" / (EXPERIMENT + "_precision_quarantine")
    destination.mkdir(parents=True, exist_ok=True)
    paths = sorted({r["journal"] for r in audit["record_mapping"]})
    by_path = {}
    for row in audit["record_mapping"]:
        by_path.setdefault(row["journal"], []).append(row["quarantined_key"])
    mapping = []
    with connect() as db:
        for text in paths:
            path = Path(text)
            assert path.parent == DBROOT / "journals" / EXPERIMENT and not path.name.startswith(REVISION)
            target = destination / (path.name + ".quarantined")
            if path.exists():
                assert not target.exists()
                path.rename(target)
            assert target.exists()
            db.execute("UPDATE source_file SET path=? WHERE path=?", (str(target), str(path)))
            for rid in by_path[text]:
                db.execute("UPDATE rollout SET original_source_file=? WHERE rollout_uid=? AND experiment_uid=?",
                           (str(target), rid, EXPERIMENT + "_precision_quarantine"))
            mapping.append({"original": str(path), "preserved": str(target)})
        db.commit()
    write(OUT / "precision_journal_archive.json", {"files": len(mapping), "not_eligible_for_generic_jsonl_ingestion": True, "mapping": mapping})


if __name__ == "__main__":
    main()
    archive()
