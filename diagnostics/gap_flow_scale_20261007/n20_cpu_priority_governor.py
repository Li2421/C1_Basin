"""Give other Slurm work twelve CPU slots while using idle capacity for Gap1.

This is an operational scheduler for the current resumable Gap1 eta arrays.
Any non-Gap1 job which is running or eligible to run gets priority.  New
Gap1 jobs must be added to OWN_JOB_IDS before submission.  Canceled eta
array tasks are journaled for exact-index recovery after their parent array.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time


ROOT = Path('diagnostics/gap_flow_scale_20261007')
ARRAYS = (9055, 9602)
FIXED_ARRAY_CONCURRENCY = {11243: 8, 11378: 4, 11469: 4, 11477: 1,
                           11681: 12, 11760: 8, 11902: 4, 12023: 8,
                           12124: 8, 12187: 4}
OWN_JOB_IDS = frozenset((*ARRAYS, 9620, 9647, 10024, 10113, 10459, 10491,
                         10633, 10708, 10725, 10736, 10811, 10908, 10917,
                         10925, 10940, 10948, 10994, 11001, 11007, 11038,
                         11228, 11243, 11354, 11378, 11466, 11469, 11477,
                         11501, 11517, 11530, 11543, 11544, 11546, 11572,
                         11618, 11661, 11681, 11760, 11891, 11902, 12023,
                         12058, 12124, 12174, 12187, 12235))
TOTAL_CPUS = 24
PRIORITY_RESERVATION = 12
ARRAY_TASK = re.compile(r'^(\d+)_(\d+)$')


def command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False)


def slurm_rows() -> list[dict]:
    out = command('squeue', '-h', '-o', '%i|%t|%C|%R')
    if out.returncode:
        raise RuntimeError(f'squeue failed: {out.stderr}')
    rows = []
    for line in out.stdout.splitlines():
        job_id, state, cpus, reason = line.split('|', 3)
        base = int(job_id.split('_', 1)[0])
        rows.append(dict(job_id=job_id, base=base, state=state,
                         cpus=int(cpus), reason=reason))
    return rows


def journal(payload: dict, *, path: Path) -> None:
    payload = {'utc': datetime.now(timezone.utc).isoformat(), **payload}
    with path.open('a') as handle:
        handle.write(json.dumps(payload, sort_keys=True) + '\n')
        handle.flush()


def tick(*, path: Path, previous: tuple[int, int] | None) -> tuple[int, int]:
    rows = slurm_rows()
    priority = [row for row in rows if row['base'] not in OWN_JOB_IDS
                and row['state'] in ('R', 'PD', 'CG')
                and row['reason'] not in ('(Dependency)', '(JobHeldUser)')]
    cap = TOTAL_CPUS - PRIORITY_RESERVATION if priority else TOTAL_CPUS
    external_running = sum(row['cpus'] for row in priority if row['state'] == 'R')
    cap = min(cap, max(0, TOTAL_CPUS - external_running))
    fixed_running = sum(row['cpus'] for row in rows if row['base'] in OWN_JOB_IDS
                        and row['base'] not in ARRAYS and row['state'] == 'R')
    fixed_pending = sum(row['cpus'] for row in rows if row['base'] in OWN_JOB_IDS
                        and row['base'] not in ARRAYS
                        and row['base'] not in FIXED_ARRAY_CONCURRENCY
                        and row['state'] == 'PD'
                        and row['reason'] not in
                        ('(Dependency)', '(JobHeldUser)', '(JobArrayTaskLimit)'))
    # squeue compresses a pending array into one row, whose CPU count is one
    # task rather than the array's intended concurrency.  Reserve all of its
    # runnable slots so the lower-priority eta workers do not starve it.
    for job, concurrency in FIXED_ARRAY_CONCURRENCY.items():
        if any(row['base'] == job and row['state'] == 'PD'
               and row['reason'] not in ('(Dependency)', '(JobHeldUser)')
               for row in rows):
            running = sum(row['cpus'] for row in rows
                          if row['base'] == job and row['state'] == 'R')
            fixed_pending += max(0, concurrency - running)
    fixed = fixed_running + fixed_pending
    slots = max(0, cap - fixed)
    active10 = any(row['base'] == 9055 and row['state'] in ('R', 'PD') for row in rows)
    active2 = any(row['base'] == 9602 and row['state'] in ('R', 'PD') for row in rows)
    if active10 and active2 and slots >= 2:
        n10, n2 = slots // 2, slots - slots // 2
    elif active10 and slots:
        n10, n2 = slots, 0
    elif active2 and slots:
        n10, n2 = 0, slots
    else:
        n10 = n2 = 0
    target = (n10, n2)
    if target != previous:
        for job, throttle in zip(ARRAYS, target):
            if not any(row['base'] == job for row in rows):
                continue
            if throttle == 0:
                result = command('scontrol', 'hold', str(job))
                event = 'hold_array'
            else:
                # Releasing does not interrupt running tasks.  Slurm may
                # return an error for finished members while applying the
                # action to pending members of the same array.
                command('scontrol', 'release', str(job))
                result = command('scontrol', 'update', f'JobId={job}',
                                 f'ArrayTaskThrottle={throttle}')
                event = 'throttle'
            journal(dict(event=event, job=job, throttle=throttle,
                         exit_code=result.returncode, stderr=result.stderr.strip(),
                         priority_jobs=[row['job_id'] for row in priority]), path=path)
    rows = slurm_rows()
    running = [row for row in rows if row['base'] in OWN_JOB_IDS and row['state'] == 'R']
    allocated = sum(row['cpus'] for row in running)
    # Reserve a slot for runnable own fixed work so it can actually launch;
    # merely reducing array throttles does not evict existing array workers.
    running_limit = max(0, cap - fixed_pending)
    if allocated > running_limit:
        candidates = [row for row in running if row['base'] in ARRAYS
                      and ARRAY_TASK.fullmatch(row['job_id'])]
        # Array IDs rise as tasks launch. Cancel the newest work first; each
        # seed has an atomic journal and can be resumed under the same index.
        candidates.sort(key=lambda row: int(ARRAY_TASK.fullmatch(row['job_id']).group(2)),
                        reverse=True)
        for row in candidates:
            if allocated <= running_limit:
                break
            result = command('scancel', row['job_id'])
            journal(dict(event='cancel_for_priority', job=row['job_id'],
                         exit_code=result.returncode, stderr=result.stderr.strip(),
                         cap=cap, priority_jobs=[item['job_id'] for item in priority]),
                    path=path)
            if result.returncode == 0:
                allocated -= row['cpus']
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=5.0)
    args = parser.parse_args()
    path = ROOT / 'n20_cpu_priority_governor.jsonl'
    previous = None
    while True:
        try:
            previous = tick(path=path, previous=previous)
        except Exception as exc:
            journal(dict(event='error', error=repr(exc)), path=path)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
