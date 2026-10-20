"""Read-only inventory of historical intervention evidence required by G1."""
from __future__ import annotations

import datetime
import hashlib
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DB = ROOT / "shared_rollout_db/rollout.sqlite"
RUNS = (
    "base_u_v12_world_timeout_goal_recovery_canonical_source_balanced_macflow",
    "base_u_v10_world_timeout_goal_recovery_source_balanced_wide_macflow",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    pre = HERE / "PREREGISTRATION.md"
    receipt = json.loads((HERE / "PREREGISTRATION.receipt.json").read_text())
    assert receipt["sha256"] == digest(pre)
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    runs = []
    for run in RUNS:
        ck = ROOT / "diagnostics/four_way_intersection_stage1" / run / "best.pkl"
        checkpoint_sha = digest(ck)
        rows = db.execute(
            """SELECT r.rollout_uid,r.experiment_uid,r.original_source_file,r.raw_record_hash
               FROM rollout r JOIN controller_config c ON r.controller_uid=c.controller_uid
               WHERE c.flow_checkpoint_sha256=? AND r.conflict_quarantined=0
               ORDER BY r.rollout_uid""",
            (checkpoint_sha,),
        ).fetchall()
        assert rows
        fields = {}
        missing_source = []
        source_hash_mismatch = []
        first_action_present = 0
        first_action_bytes_present = 0
        trace_present = 0
        for row in rows:
            path = Path(row["original_source_file"])
            if not path.is_file():
                missing_source.append(row["rollout_uid"])
                continue
            raw = json.loads(path.read_text().splitlines()[0])
            rec = raw.get("record", raw)
            for key in rec:
                fields[key] = fields.get(key, 0) + 1
            canonical = json.dumps(rec, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if hashlib.sha256(canonical.encode()).hexdigest() != row["raw_record_hash"]:
                source_hash_mismatch.append(row["rollout_uid"])
            first_action_present += int(rec.get("first_action") is not None)
            first_action_bytes_present += int(rec.get("first_action_bytes") is not None)
            trace_present += int(bool(rec.get("trace_path")))
        runs.append(
            {
                "run": run,
                "checkpoint_path": str(ck),
                "checkpoint_sha256": checkpoint_sha,
                "db_records": len(rows),
                "source_files_missing": len(missing_source),
                "source_hash_mismatch": len(source_hash_mismatch),
                "record_first_action_present": first_action_present,
                "record_first_action_bytes_present": first_action_bytes_present,
                "record_trace_path_present": trace_present,
                "record_fields": sorted(fields),
                "first_eight_rollout_uids": [r["rollout_uid"] for r in rows[:8]],
                "example_source_path": rows[0]["original_source_file"],
                "example_source_sha256": digest(Path(rows[0]["original_source_file"])),
            }
        )
    result = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "G1_FAILED_UNVERIFIABLE_HISTORICAL_FIRST_ACTION",
        "preregistration_sha256": digest(pre),
        "database_path": str(DB),
        "database_schema_has_first_action": any(
            x[1] == "first_action" for x in db.execute("PRAGMA table_info(rollout)").fetchall()
        ),
        "required_records_per_run": 8,
        "runs": runs,
        "new_round19_success_labels_generated": False,
        "new_round19_probes_started": False,
        "reason": "Historical intervention DB rows and source journals do not retain first-action values or bytes; the required independent byte comparison cannot be performed.",
    }
    assert all(x["db_records"] >= 8 for x in runs)
    assert not result["database_schema_has_first_action"]
    assert all(x["record_first_action_present"] == 0 and x["record_first_action_bytes_present"] == 0 for x in runs)
    out = HERE / "G1_EVIDENCE_AUDIT.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "record_counts": [x["db_records"] for x in runs]}))


if __name__ == "__main__":
    main()
