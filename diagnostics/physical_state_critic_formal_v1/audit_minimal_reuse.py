"""Read-only, exhaustive exact-slot audit of the frozen formal Q requests.

This checks the entire shared rollout database after completed worker journals
have been merged. It never treats a nearby state, eta or seed as equivalent.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent
DB = ROOT.parents[1] / "shared_rollout_db/rollout.sqlite"


def panels():
    for scene in ("toy_give_way", "ring_exchange"):
        for chain in ("TT", "FF"):
            for split in ("train", "validation", "test"):
                yield f"matched/{scene}/{chain}/{split}", ROOT / "matched" / scene / chain / split / "cases.json"
    for scene in ("toy_give_way", "double_bottleneck", "four_way_intersection", "ring_exchange"):
        for split in ("train", "validation", "test"):
            yield f"loso/{scene}/{split}", ROOT / "loso" / scene / split / "cases.json"


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("CREATE TEMP TABLE request(rollout_uid TEXT PRIMARY KEY, state_index INTEGER, eta_index INTEGER)")
    results = {}
    for name, path in panels():
        con.execute("DELETE FROM request")
        cases = json.loads(path.read_text())
        con.executemany(
            "INSERT INTO request VALUES(?,?,?)",
            ((x["rollout_uid"], x["state_index"], x["eta_index"]) for x in cases),
        )
        assert con.execute("SELECT COUNT(*) FROM request").fetchone()[0] == len(cases), name
        pair = defaultdict(lambda: Counter())
        query = """SELECT q.state_index,q.eta_index,
                          COUNT(r.rollout_uid),
                          SUM(CASE WHEN r.rollout_uid IS NOT NULL AND r.numerical_failure=0 THEN 1 ELSE 0 END),
                          SUM(CASE WHEN r.numerical_failure=1 THEN 1 ELSE 0 END)
                   FROM request q LEFT JOIN rollout r
                   ON q.rollout_uid=r.rollout_uid AND r.conflict_quarantined=0
                   GROUP BY q.state_index,q.eta_index"""
        for state, eta, observed, resolved, numerical in con.execute(query):
            pair[state][eta] = (observed, resolved, numerical)
        split = name.rsplit("/", 1)[-1]
        trials = 4 if split == "train" else 16
        n_states = 512 if split == "train" else 32 if split == "validation" else 128
        eta_count = 16 if split == "train" else 17
        assert len(pair) == n_states and all(len(x) == eta_count for x in pair.values()), name
        full_candidates = [sum(pair[s][j][0] == trials for j in range(16)) for s in range(n_states)]
        states_all_eta = sum(x == 16 for x in full_candidates)
        states_all_eta_safety = sum(full_candidates[s] == 16 and pair[s][16][0] == trials
                                    for s in range(n_states)) if split != "train" else None
        stats = Counter()
        for state_rows in pair.values():
            for observed, resolved, numerical in state_rows.values():
                stats["observed"] += observed
                stats["resolved"] += resolved
                stats["numerical"] += numerical or 0
        results[name] = dict(requests=len(cases), **stats,
                             missing=len(cases)-stats["observed"],
                             states=n_states, trials_per_pair=trials,
                             states_all_16_eta=states_all_eta,
                             states_all_16_eta_and_safety=states_all_eta_safety,
                             states_at_least_one_complete_eta=sum(x > 0 for x in full_candidates),
                             complete_eta_pairs=sum(full_candidates),
                             states_with_complete_safety=(sum(pair[s][16][0] == trials for s in range(n_states))
                                                          if split != "train" else None))
        print(name, results[name], flush=True)
    con.close()
    output = ROOT / "minimal_reuse_db_audit.json"
    output.write_text(json.dumps(dict(schema="all_frozen_formal_requests_exact_db_scan_v1",
                                      database=str(DB), panels=results), indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
