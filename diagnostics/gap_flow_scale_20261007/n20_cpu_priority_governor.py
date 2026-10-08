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
OWN_JOB_IDS = frozenset((*ARRAYS, 9620, 9647, 10024, 10113))
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
    fixed = sum(row['cpus'] for row in rows if row['base'] in OWN_JOB_IDS
                and row['base'] not in ARRAYS and
                (row['state'] == 'R' or
                 (row['state'] == 'PD' and row['reason'] not in
                  ('(Dependency)', '(JobHeldUser)'))))
    slots = max(0, cap - fixed)
    n10 = max(1, slots // 2)
    n2 = max(1, slots - n10)
    target = (n10, n2)
    if target != previous:
        for job, throttle in zip(ARRAYS, target):
            # Slurm may return an error for already-finished array members
            # while still updating live members; verify via the next tick.
            result = command('scontrol', 'update', f'JobId={job}',
                             f'ArrayTaskThrottle={throttle}')
            journal(dict(event='throttle', job=job, throttle=throttle,
                         exit_code=result.returncode, stderr=result.stderr.strip(),
                         priority_jobs=[row['job_id'] for row in priority]), path=path)
    rows = slurm_rows()
    running = [row for row in rows if row['base'] in OWN_JOB_IDS and row['state'] == 'R']
    allocated = sum(row['cpus'] for row in running)
    if allocated > cap:
        candidates = [row for row in running if row['base'] in ARRAYS
                      and ARRAY_TASK.fullmatch(row['job_id'])]
        # Array IDs rise as tasks launch. Cancel the newest work first; each
        # seed has an atomic journal and can be resumed under the same index.
        candidates.sort(key=lambda row: int(ARRAY_TASK.fullmatch(row['job_id']).group(2)),
                        reverse=True)
        for row in candidates:
            if allocated <= cap:
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
