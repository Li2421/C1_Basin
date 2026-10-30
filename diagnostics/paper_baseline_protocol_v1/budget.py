"""Crash-conservative attempt reservations with the protocol's hard 50k cap."""
import guard
from guard import HERE
import datetime
import fcntl
import json
import os
import time

LIMIT = 50000
LEDGER = HERE / 'attempt_reservations.jsonl'


def _last_reservation():
    if not LEDGER.exists() or not LEDGER.stat().st_size:
        return 0
    with LEDGER.open('rb') as stream:
        stream.seek(max(0, LEDGER.stat().st_size - 16384))
        lines = stream.read().splitlines()
    # A torn final line is not silently discarded: ambiguity at the resource
    # boundary must stop submission rather than risk exceeding the limit.
    return int(json.loads(lines[-1])['ordinal'])


def used():
    path = HERE / 'BUDGET.json'
    count = json.loads(path.read_text())['actual_attempts'] if path.exists() else 0
    return max(count, _last_reservation())


def reserve(case):
    assert (HERE / 'PRIOR_TASK_DELIVERED.json').exists()
    with (HERE / 'budget.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        count = used()
        if count >= LIMIT:
            raise RuntimeError('HARD_STOP: 50000 new rollout attempts exhausted')
        record = dict(ordinal=count + 1, rollout_uid=case['rollout_uid'], job=os.environ.get('SLURM_JOB_ID'),
                      purpose=case.get('purpose'), time=datetime.datetime.now().astimezone().isoformat(), time_ns=time.time_ns())
        # Persist the append first. If the subsequent summary rename fails, the
        # ledger still reserves this ordinal and used() cannot undercount it.
        with LEDGER.open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary = HERE / 'BUDGET.json.tmp'
        temporary.write_text(json.dumps(dict(actual_attempts=count + 1, hard_limit=LIMIT,
            counting='Reservations immediately before execution; includes aborted/retry attempts and may conservatively include a crash before numerical start.'), ensure_ascii=False, indent=2) + '\n')
        temporary.replace(HERE / 'BUDGET.json')
        return count + 1
