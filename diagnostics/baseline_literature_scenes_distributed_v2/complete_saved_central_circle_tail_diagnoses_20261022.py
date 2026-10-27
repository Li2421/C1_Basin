"""Complete two landed central-circle diagnoses on the reserved local CPU.

Uses saved evidence and the sealed diagnostic implementation; no research jobs,
claims, simulations, controller changes or database writes.
"""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from resources_corrections_release_20261023 import snapshot
import fcntl


def main():
    verify_receipt('CENTRAL_CIRCLE_TAIL_DIAGNOSIS_PLAN_20261022.receipt.json')
    os.sched_setaffinity(0, {read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    os.nice(10)
    with (HERE/'CENTRAL_CIRCLE_TAIL_DIAGNOSIS_20261022.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import numpy as np
        from failure_analysis import summary
        written = []
        for row in read(HERE/'CENTRAL_CIRCLE_TAIL_DIAGNOSIS_PLAN_20261022.json')['rows']:
            if any(j['name'].startswith('blv2_analysis') for j in snapshot()['jobs']):
                raise RuntimeError('Yield reserved-CPU diagnosis to Slurm analysis')
            dest = HERE/'DIAGNOSTICS'/f"{row['rollout_uid']}.json"
            if dest.exists():
                continue
            assert sha(row['path']) == row['sha256']
            raw = read(row['path'])
            assert sha(raw['trace_path']) == raw['trace_sha256']
            with np.load(raw['trace_path']) as z:
                result = summary(raw, z)
            assert not dest.exists(), 'Concurrent diagnostic writer'
            write(dest, dict(utc=utc(), rollout_uid=row['rollout_uid'], source=row,
                             trace_sha256=raw['trace_sha256'], **result))
            written.append(str(dest))
            print('Saved-trace diagnosis complete', row['rollout_uid'], flush=True)
        seal('CENTRAL_CIRCLE_TAIL_DIAGNOSIS_COMPLETE_20261022.receipt.json',
             [HERE/'CENTRAL_CIRCLE_TAIL_DIAGNOSIS_PLAN_20261022.receipt.json', *written],
             diagnoses_written=len(written), new_rollouts=0, database_writes=0,
             local_cpu_affinity=sorted(os.sched_getaffinity(0)))


if __name__ == '__main__':
    main()
