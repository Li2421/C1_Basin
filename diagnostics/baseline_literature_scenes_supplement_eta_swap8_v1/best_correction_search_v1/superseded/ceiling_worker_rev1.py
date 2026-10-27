"""Search worker: frozen runtimes, outputs and journals only under the search experiment."""
from pathlib import Path
import sys
S2 = Path(__file__).resolve().parent
if (S2 / 'FATAL_STOP.json').exists():
    raise SystemExit('Search fatal stop')
sys.path.insert(0, str(S2))
from ceiling_common import *  # noqa: F401,F403
from registry import record
from swap_runtime import run as original_run
from swap_runtime_user_fix_20261023 import run as revised_run
from swap_diagnostics import summary
import os
import signal

STOP = False


def stop(*_):
    global STOP
    STOP = True


def save2(stage, c, result, arrays):
    folder = S2 / stage; trace = folder / 'traces' / f"{c['rollout_uid']}.npz"; trace.parent.mkdir(parents=True, exist_ok=True)
    temp = trace.with_suffix('.partial.npz'); np.savez_compressed(temp, **arrays)
    with temp.open('rb') as stream:
        os.fsync(stream.fileno())
    temp.replace(trace)
    raw = folder / 'raw' / f"{c['rollout_uid']}.json"
    result = clean(record(c, dict(result, trace_path=str(trace), trace_sha256=sha(trace))))
    write2(raw, result)
    journal = cache_writer.append_journal([result], EXP2, worker_id=f'{stage}_{os.getpid()}')
    write2(folder / 'journal_receipts' / f"{c['rollout_uid']}.json", dict(journal=str(journal), raw_sha256=sha(raw)))
    return result


def repair(stage, c):
    raw = S2 / stage / 'raw' / f"{c['rollout_uid']}.json"; marker = S2 / stage / 'journal_receipts' / f"{c['rollout_uid']}.json"
    if raw.exists() and not marker.exists():
        result = read(raw); assert sha(result['trace_path']) == result['trace_sha256']
        journal = cache_writer.append_journal([result], EXP2, worker_id=f'{stage}_recovery_{os.getpid()}')
        write2(marker, dict(journal=str(journal), raw_sha256=sha(raw), recovered_existing_raw_without_new_steps=True))


def execute(c):
    revised = bool(c.get('scene_revision'))
    result, arrays = (revised_run if revised else original_run)(c)
    result.update(execution_backend='user_fix_20261023' if revised else 'unchanged_free', write_path_audit_sha256=sha(HERE / 'WRITE_PATH_AUDIT.json'), scene_revision=c.get('scene_revision'))
    return result, arrays


def main(stage):
    for name in FROZEN:
        verify_receipt(name)
    folder = S2 / 'QUEUES' / stage; proof = read(folder / 'QUEUE.receipt.json')
    for p, h in proof['files'].items():
        assert sha(p) == h, p
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGUSR1, stop)
    for entry in read(folder / 'INDEX.json'):
        if STOP:
            return
        claim = folder / 'claims' / entry['group']; claim.parent.mkdir(parents=True, exist_ok=True)
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        job = f"{os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID', 'local'))}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
        write2(claim / 'OWNER.json', dict(pid=os.getpid(), job=job, utc=utc()))
        assert sha(entry['path']) == entry['sha256']
        for c in read(entry['path']):
            if STOP:
                return
            raw = S2 / stage / 'raw' / f"{c['rollout_uid']}.json"
            if raw.exists():
                repair(stage, c); continue
            result, arrays = execute(c)
            result = save2(stage, c, result, arrays)
            write2(S2 / 'DIAGNOSTICS' / f"{c['rollout_uid']}.json", summary(result, arrays))
            print(stage, c['state']['scene'], c['controller_payload']['nominal']['controller'], c['state']['index'], c['requested_action'], c['seed'],
                  result['outcome'], round(result['elapsed_seconds'], 3), flush=True)
        write2(claim / 'COMPLETE.json', dict(utc=utc()))


if __name__ == '__main__':
    main(sys.argv[1])
