"""Read-only journal/manifest/database reconciliation; no rollout or training."""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone

from .data import (OUT, DBROOT, EXPERIMENT, REVISION, CONTROLLERS, read, write,
                   connect, canonical, uid, digest, eta_identity, validate_record)


def main(complete=None):
    protocol = read(OUT / "protocol.json")
    states = {s["uid"]: s for s in read(OUT / "states.json")}
    pairs = {(p["state_uid"], p["eta_uid"]): p for p in read(OUT / "pairs.json")}
    profiles = {c["controller_uid"]: c for c in protocol["profiles"]}
    stats = {c: Counter() for c in CONTROLLERS}
    seen = set()
    with connect(True) as db:
        payloads = {}
        for cid, c in profiles.items():
            row = db.execute("SELECT * FROM controller_config WHERE controller_uid=?", (cid,)).fetchone()
            assert row["compatibility_quality"] == "EXACT_PROFILE"
            payloads[cid] = json.loads(row["config_json"])
            assert uid("ctl", payloads[cid]) == cid
            assert payloads[cid]["flow_checkpoint_sha256"] == c["sha256"]
        for suid, s in states.items():
            row = db.execute("SELECT * FROM state WHERE state_uid=?", (suid,)).fetchone()
            assert row["content_hash"] == s["content_hash"]
            assert row["source_group"] == s["source_group"]
            assert canonical(json.loads(row["physical_state_json"])) == canonical(s["physical"])
        for eu, eta in {p["eta_uid"]: p["eta"] for p in pairs.values()}.items():
            row = db.execute("SELECT * FROM eta WHERE eta_uid=?", (eu,)).fetchone()
            assert eta_identity([row["eta1"], row["eta2"], row["eta3"]])[0] == eu == eta_identity(eta)[0]
        for path in sorted((DBROOT / "journals" / EXPERIMENT).glob(f"{REVISION}_*.jsonl")):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                envelope = json.loads(line)
                assert envelope["schema"] == "controller_training_repair_v1"
                r = envelope["record"]
                eu = validate_record(r, protocol, states, pairs, payloads)
                name = profiles[r["controller_uid"]]["name"]
                count = stats[name]
                sk = canonical({"future_index": r["future_index"]})
                rid = uid("roll", {"state": r["state_uid"], "eta": eu, "controller": r["controller_uid"], "seed": sk})
                assert rid not in seen, "Unplanned duplicate execution"
                seen.add(rid)
                count["valid_journal_records"] += 1
                count["numerical_separate"] += int(r["numerical_failure"])
                count["collision"] += int(r["collision"])
                row = db.execute("SELECT * FROM rollout WHERE rollout_uid=?", (rid,)).fetchone()
                if row is None:
                    count["awaiting_single_merger"] += 1
                    continue
                assert row["compatibility_quality"] == "EXACT_REUSE" and not row["conflict_quarantined"]
                assert row["raw_record_hash"] == digest(r)
                for k in ("success", "deadlock", "timeout", "collision", "numerical_failure", "episode_length"):
                    assert row[k] == r[k], k
                assert row["seed_key"] == row["continuation_seed_json"] == sk
                source = uid("src", {"path": str(path.resolve())})
                assert db.execute("SELECT 1 FROM rollout_source WHERE rollout_uid=? AND source_uid=? AND source_line=?",
                                  (rid, source, number)).fetchone()
                count["journal_db_exact_match"] += 1
        for req in read(OUT / "planned_rollouts.json")["requests"]:
            name = profiles[req["controller_uid"]]["name"]
            count = stats[name]
            rows = {r["seed_key"]: r for r in db.execute(
                "SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?",
                (req["state_uid"], req["eta_uid"], req["controller_uid"]))}
            for sk in req["seed_keys"]:
                count["requested"] += 1
                row = rows.get(sk)
                if row is None:
                    count["planned_not_in_db_yet"] += 1
                    continue
                assert row["compatibility_quality"] == "EXACT_REUSE" and not row["conflict_quarantined"]
                if row["numerical_failure"]:
                    count["planned_numerical_not_imputed"] += 1
                else:
                    count["planned_exact_reusable"] += 1
                count["historical_reused"] += int(row["experiment_uid"] != EXPERIMENT)
        quarantine = db.execute("SELECT COUNT(*) FROM rollout WHERE experiment_uid=? AND conflict_quarantined=1",
                                (EXPERIMENT + "_precision_quarantine",)).fetchone()[0]
        assert quarantine == 4825
    selected = CONTROLLERS if complete == "all" else [complete] if complete else []
    for c in selected:
        assert not stats[c]["awaiting_single_merger"] and not stats[c]["planned_not_in_db_yet"], c
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "completed_controllers_required": selected,
              "controllers": {k: dict(v) for k, v in stats.items()},
              "precision_mismatch_records_quarantined": quarantine, "all_checks_passed": True,
              "scope": "full raw journals, native precision, manifest state/eta/seed/controller fingerprints, DB outcomes and provenance",
              "new_rollouts_from_this_audit": 0}
    write(OUT / f"alignment_audit_{complete or 'live'}.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--complete", choices=[*CONTROLLERS, "all"])
    main(parser.parse_args().complete)
