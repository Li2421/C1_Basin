"""Read-only exact-UID census for the frozen physical-only Q panels.

Do not infer reuse from nearby states, eta values, or rollout seeds.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORMAL = ROOT / "diagnostics/physical_state_critic_formal_v1"
DB = ROOT / "shared_rollout_db/rollout.sqlite"


def panels():
    for scene in ("toy_give_way", "ring_exchange"):
        for placement in ("TT", "FF"):
            for split in ("train", "validation", "test"):
                yield f"matched/{scene}/{placement}/{split}", FORMAL / "matched" / scene / placement / split / "cases.json"
    for scene in ("toy_give_way", "double_bottleneck", "four_way_intersection", "ring_exchange"):
        for split in ("train", "validation", "test"):
            yield f"loso/{scene}/{split}", FORMAL / "loso" / scene / split / "cases.json"


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("CREATE TEMP TABLE request(rollout_uid TEXT PRIMARY KEY,state_index INTEGER,eta_index INTEGER)")
    result = {"schema": "transferable_state_conditioning_exact_cache_census_v1",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "db_rollout_rows": con.execute("SELECT COUNT(*) FROM rollout").fetchone()[0],
              "panels": {}}
    for name, path in panels():
        cases = json.loads(path.read_text())
        con.execute("DELETE FROM request")
        con.executemany("INSERT INTO request VALUES(?,?,?)",
                        ((c["rollout_uid"], c["state_index"], c["eta_index"]) for c in cases))
        assert con.execute("SELECT COUNT(*) FROM request").fetchone()[0] == len(cases), name
        pair = defaultdict(dict)
        for s, j, n, success, collision, numerical in con.execute(
            """SELECT q.state_index,q.eta_index,COUNT(r.rollout_uid),
                      SUM(CASE WHEN r.success=1 THEN 1 ELSE 0 END),
                      SUM(CASE WHEN r.collision=1 THEN 1 ELSE 0 END),
                      SUM(CASE WHEN r.numerical_failure=1 THEN 1 ELSE 0 END)
               FROM request q LEFT JOIN rollout r
               ON q.rollout_uid=r.rollout_uid AND r.conflict_quarantined=0
               GROUP BY q.state_index,q.eta_index"""):
            pair[s][j] = (n, success or 0, collision or 0, numerical or 0)
        trials = 4 if name.endswith("/train") else 16
        full = {s: [j for j in range(16) if pair[s][j][0] == trials] for s in pair}
        complete_states = [s for s, js in full.items() if len(js) == 16]
        distinct = Counter()
        for s in complete_states:
            counts = [pair[s][j][1] for j in range(16)]
            distinct["nonconstant_eta_outcome"] += int(min(counts) < max(counts))
            distinct["at_least_one_success_and_failure_eta"] += int(min(counts) == 0 and max(counts) == trials)
        out = {"requested_seed_slots": len(cases), "observed_seed_slots": sum(v[0] for x in pair.values() for v in x.values()),
               "independent_states": len(pair), "complete_16_eta_states": len(complete_states),
               "complete_eta_pairs": sum(len(js) for js in full.values()),
               "observed_safety_states": sum(pair[s].get(16, (0,))[0] == trials for s in pair),
               "collisions": sum(v[2] for x in pair.values() for v in x.values()),
               "numerical_failures": sum(v[3] for x in pair.values() for v in x.values()),
               **distinct}
        result["panels"][name] = out
        print(name, out, flush=True)
    con.close()
    target = HERE / "formal_exact_db_audit.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(target, flush=True)


if __name__ == "__main__":
    main()
