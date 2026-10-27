"""Observe the authorized workflow; never retry rollouts or modify scientific evidence."""
from pathlib import Path
import datetime, fcntl, json, os, subprocess, time

HERE = Path(__file__).resolve().parent


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def alive(pid, script):
    try:
        return str(HERE / script).encode() in Path('/proc', str(pid), 'cmdline').read_bytes()
    except FileNotFoundError:
        return False


def snapshot():
    from local_cpu_guard import pin
    pin()
    progress = read(HERE / 'PROGRESS.json')
    stage = progress.get('phase', '')
    folder = HERE / stage / 'raw' if stage else None
    count = sum(e.name.endswith('.json') for e in os.scandir(folder)) if folder and folder.is_dir() else 0
    processes = {}
    for file, script in [('DRIVER_PID.json', 'driver.py'),
                         ('RESOURCE_GUARD_PID.json', 'resources.py'),
                         ('LOCAL_CPU_GUARD_PID.json', 'local_cpu_guard.py')]:
        pid = read(HERE / file).get('pid', 0)
        processes[script] = dict(pid=pid, alive=alive(pid, script))
    from resources import jobs
    all_jobs = jobs()
    mine = [r for r in all_jobs if r['user'] == 'zhihan']
    queue = read(HERE / 'QUEUES' / stage / 'QUEUE.receipt.json') if stage else {}
    result = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  phase=stage, completed_local_rows=count,
                  stage_planned=queue.get('planned'), stage_missing_at_preflight=queue.get('missing'),
                  processes=processes, all_jobs=all_jobs,
                  own_running_worker_cpus=sum(int(r['cpus']) for r in mine if r['state'] == 'RUNNING'),
                  total_cpu_cap_including_management=18,
                  fatal_stop=read(HERE / 'FATAL_STOP.json'),
                  driver_error=read(HERE / 'DRIVER_ERROR.json'),
                  complete=(HERE / 'COMPLETE.json').exists(),
                  observational_only=True)
    incidents = []
    if result['fatal_stop']:
        incidents.append('read_only_fatal_stop')
    if result['driver_error']:
        incidents.append('driver_error_requires_agent_review')
    if not result['complete'] and not processes['driver.py']['alive']:
        incidents.append('coordinator_not_running_requires_agent_review')
    if result['own_running_worker_cpus'] + 1 > 18:
        incidents.append('cpu_cap_exceeded_resource_guard_must_handle')
    result['incidents'] = incidents
    return result


def main():
    with (HERE / 'PROCESS_WATCH.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(HERE / 'PROCESS_WATCH_PID.json', dict(pid=os.getpid(), script=str(Path(__file__).resolve())))
        previous = None
        while True:
            try:
                status = snapshot()
                write(HERE / 'PROCESS_WATCH_STATUS.json', status)
                signature = json.dumps([status['incidents'], status['driver_error'], status['fatal_stop']], sort_keys=True)
                if status['incidents'] and signature != previous:
                    with (HERE / 'PROCESS_WATCH_INCIDENTS.jsonl').open('a') as out:
                        out.write(json.dumps(status, ensure_ascii=False) + '\n')
                previous = signature
                if status['complete']:
                    return
            except Exception as exc:
                with (HERE / 'PROCESS_WATCH_ERRORS.jsonl').open('a') as out:
                    out.write(json.dumps(dict(utc=time.time(),error=repr(exc))) + '\n')
            time.sleep(20)


if __name__ == '__main__':
    main()
