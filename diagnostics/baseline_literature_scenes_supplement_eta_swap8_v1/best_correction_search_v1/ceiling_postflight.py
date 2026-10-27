"""Search postflight: every registered request is in the official DB with matching identity/outcome; new raws hash-match
their DB rows and traces; no pending journals for the search experiment. Read-only except the report file."""
from pathlib import Path
import sys
S2 = Path(__file__).resolve().parent
sys.path.insert(0, str(S2))
from ceiling_common import *  # noqa: F401,F403
from shared_rollout_db.src.rollout_db import connect, canonical as dbcanonical

KEYS = ('state_uid', 'eta_uid', 'controller_uid', 'seed_key', 'success', 'collision', 'timeout', 'numerical_failure', 'outcome')


def main():
    errors = []; seen = set(); traces = set(); new = reused = unknown = aliases = 0
    stages = sorted(p.name[:-len('_REGISTRATION.receipt.json')] for p in S2.glob('*_REGISTRATION.receipt.json'))
    with connect(True) as db:
        for stage in stages:
            for c in read(S2 / f'{stage}_CASES.json'):
                rid = c['rollout_uid']
                if rid in seen:
                    continue
                seen.add(rid)
                row = db.execute('SELECT * FROM rollout WHERE rollout_uid=?', (rid,)).fetchone()
                if not row or row['conflict_quarantined']:
                    errors.append(dict(stage=stage, rollout_uid=rid, reason='missing_or_conflicted_db')); continue
                raw = S2 / stage / 'raw' / f'{rid}.json'
                if raw.exists():
                    r = read(raw); new += 1; aliases += int('deterministic_source' in r); unknown += int(r['numerical_failure'])
                    if any(row[k] != r[k] for k in KEYS):
                        errors.append(dict(stage=stage, rollout_uid=rid, reason='identity_or_outcome_mismatch'))
                    if row['raw_record_hash'] != hashlib.sha256(dbcanonical(r).encode()).hexdigest():
                        errors.append(dict(stage=stage, rollout_uid=rid, reason='raw_hash_mismatch'))
                    if r['trace_path'] not in traces:
                        traces.add(r['trace_path'])
                        if sha(r['trace_path']) != r['trace_sha256']:
                            errors.append(dict(stage=stage, rollout_uid=rid, reason='trace_hash_mismatch'))
                else:
                    reused += 1
    pending = sorted(str(p) for p in (ROOT / 'shared_rollout_db/journals' / EXP2).glob('*.jsonl'))
    out = dict(utc=utc(), passed=not errors and not pending, stages=stages, distinct_requests=len(seen), new_records=new, deterministic_aliases=aliases,
               reused_records=reused, unknowns_preserved=unknown, traces_checked=len(traces), pending_journals=pending, errors=errors[:200], error_count=len(errors))
    write2(S2 / 'DB_POSTFLIGHT.json', out)
    print('postflight passed' if out['passed'] else 'postflight FAILED', len(seen), new, reused, aliases, unknown, len(errors), len(pending), flush=True)


if __name__ == '__main__':
    main()
