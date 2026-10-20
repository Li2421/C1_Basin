"""Submit the recorded initial comparisons once the complete bundle is ready."""
from pathlib import Path
import datetime,fcntl,json,os,time
from resource_guard import queue
from submit_selector_candidates import main as submit

HERE=Path(__file__).resolve().parent


def write(name,value):
    path=HERE/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    lock=(HERE/'ADVANCE_SELECTOR.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write('ADVANCE_SELECTOR_PID.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    while True:
        ready=(HERE/'DEVELOPMENT_BUNDLE.receipt.json').exists()
        submitted=(HERE/'INITIAL_SELECTOR_COMPARISONS_SUBMISSION.json').exists()
        if ready and not submitted and queue()['reserved_cpu_shards']<=14:
            submit();submitted=True
        write('SELECTOR_LIVE_STATUS.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            bundle_ready=ready,initial_comparisons_submitted=submitted,freeze_or_locked_access_authorized_to_this_script=False))
        if submitted:return
        time.sleep(30)


if __name__=='__main__':main()
