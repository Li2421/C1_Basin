"""Resumable rollout receipt gate, DB merge, and complete numerical audit."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / 'heldout_adaptive_selection_v1'
PY = Path('/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python')
sys.path.insert(0, str(REPO))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def status(phase, **kwargs):
    write(OUT / 'batch_status.json', dict(phase=phase, complete=False, video_required=False,
                                          updated=time.time(), **kwargs))


def run(name, script, action=None):
    command = [str(PY), str(HERE / script)]
    if action:
        command.append(action)
    status(name, job_id=read(OUT / 'submission.json')['job_id'])
    with (OUT / 'logs' / f'{name}.log').open('a') as stream:
        subprocess.run(command, cwd=REPO, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def receipts():
    missing = read(OUT / 'missing_cases.json')
    chunk_size = read(OUT / 'submission.json')['chunk_size']
    count = math.ceil(len(missing) / chunk_size)
    assert count == 342
    incomplete = []
    for i in range(count):
        path = OUT / 'progress' / f'audit_{i}.json'
        expected = min(chunk_size, len(missing) - i * chunk_size)
        if not path.is_file():
            incomplete.append(i)
            continue
        record = read(path)
        if (record['worker'] != i or record['workers'] != count or
                record['total'] != expected or record['completed'] != expected or
                not record['complete'] or record['limited_pilot']):
            incomplete.append(i)
    return incomplete


def wait_job(job_id):
    while True:
        queue = subprocess.run(['squeue', '-h', '-j', str(job_id)], cwd=REPO,
                               capture_output=True, text=True)
        if queue.returncode and 'Invalid job id specified' in queue.stderr:
            return
        queue.check_returncode()
        if not queue.stdout.strip():
            return
        time.sleep(25)


def resubmit(ids, attempt):
    array = ','.join(str(i) for i in ids) + '%22'
    cmd = ['sbatch', '--parsable', '-p', 'cpu', '--array=' + array,
           '--cpus-per-task=1', '--mem=3G', '--time=00:45:00',
           '--job-name=ring_q_adapt_resume',
           f'--output={OUT}/logs/resume{attempt}_%A_%a.log',
           str(HERE / 'adaptive_selection_audit_worker.sh'), '342']
    job = subprocess.check_output(cmd, cwd=REPO, text=True).strip().split(';')[0]
    write(OUT / f'resubmission_{attempt}.json',
          dict(job_id=int(job), shard_ids=ids, attempt=attempt, submitted=time.time()))
    return job


def main():
    assert (OUT / 'analysis_plan.json').is_file()
    assert (OUT / 'protocol.json').is_file()
    assert (OUT / 'prediction_audit.json').is_file()
    assert (OUT / 'sampling_audit.json').is_file()
    job = read(OUT / 'submission.json')['job_id']
    for attempt in range(3):
        status('rollouts', job_id=job,
               completed_shards=342 - len(receipts()))
        wait_job(job)
        incomplete = receipts()
        if not incomplete:
            break
        if attempt == 2:
            raise RuntimeError(f'{len(incomplete)} shards did not complete: {incomplete[:30]}')
        job = resubmit(incomplete, attempt + 1)
    assert not receipts()
    write(OUT / 'rollout_receipt_audit.json',
          dict(expected_shards=342, complete_shards=342, missing_shards=0,
               expected_requests=65536, all_worker_receipts_complete=True))
    if not (OUT / 'merge_audit.json').exists():
        run('merge', 'adaptive_selection_audit_v1.py', 'merge')
    if not (OUT / 'collection_audit.json').exists():
        run('collect', 'adaptive_selection_audit_v1.py', 'collect')
    while read(OUT / 'prediction_audit.json').get('inference_backend') != 'gpu':
        status('waiting_for_frozen_gpu_predictions', job_id=job,
               rollouts_complete=True)
        time.sleep(25)
    if not (OUT / 'audit_results.json').exists():
        run('analyze', 'adaptive_selection_analyze_v1.py')
    if not (OUT / 'verification.json').exists():
        run('verify', 'adaptive_selection_verify_v1.py')
    status('awaiting_verdict', job_id=job, rollouts_complete=True,
           analysis_complete=True, verification_complete=True)
    print('Complete numerical audit ready for final verdict.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=repr(exc))
        raise
