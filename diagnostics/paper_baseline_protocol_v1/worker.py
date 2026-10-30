"""Single-CPU workers with UID locks and a fresh interpreter for each paper局."""
import os
os.environ.pop('STB_COORDINATOR', None)
import guard
from guard import HERE
from pathlib import Path
import json
import sys
import fcntl
import subprocess
import time


def read(path):return json.loads(Path(path).read_text())


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main(queue_name, index):
    assert(HERE/'PRIOR_TASK_DELIVERED.json').exists()
    queue = read(HERE / 'queues' / (queue_name + '.json'))
    local = queue['shards'][index]
    mine = {c['rollout_uid'] for c in local}
    other = [c for shard in queue['shards'] for c in shard if c['rollout_uid'] not in mine]
    other.sort(key=lambda c: (-c['scenario_payload']['config']['max_steps'] * c['state']['N'] ** 1.5, c['rollout_uid']))
    folder = HERE / 'case_locks'
    folder.mkdir(exist_ok=True)
    numerical_runner=None
    for request in local + other:
        uid = request['rollout_uid']
        raw = HERE / 'raw' / (uid + '.json')
        if raw.exists():
            continue
        with (folder / (uid + '.lock')).open('a+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if raw.exists():
                continue
            if request['controller_payload']['engine'] == 'paper_own_algorithm':
                from execution_policy import assert_method_enabled
                assert_method_enabled(request['controller_payload']['paper_method'])
                path = HERE / 'paper_requests' / (uid + '.json')
                dump(path, request)
                # Generic vendor names and installed wrappers are isolated to
                # one episode; papers receive no C1 acceleration patches.
                child_start=time.monotonic()
                child = subprocess.run([sys.executable, str(HERE / 'paper_episode.py'), str(path)], check=False)
                child_seconds=time.monotonic()-child_start
                if raw.exists():
                    result=read(raw)
                    dump(HERE/'paper_process_timing'/(uid+'.json'),dict(
                        rollout_uid=uid,child_process_seconds=child_seconds,
                        recorded_episode_seconds=result['wall_seconds'],
                        import_and_exit_residual_seconds=max(0.,child_seconds-result['wall_seconds']),
                        purpose='Observed overhead for forecasting only; author execution unchanged.'))
                if child.returncode and not raw.exists():
                    raise RuntimeError(f'Paper child exited {child.returncode} before saving {uid}')
            else:
                if numerical_runner is None:
                    import runner as numerical_runner
                    from budget import reserve
                reserve(request)
                result = numerical_runner.work(request)
                print(queue_name, uid, result['outcome'], round(result['wall_seconds'], 3), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
