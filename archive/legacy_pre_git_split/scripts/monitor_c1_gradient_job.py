"""Check the authorized Slurm diagnostic every 15 minutes, then summarize."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.c1.training.persistence import atomic_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, type=int)
    parser.add_argument('--folder', required=True, type=Path)
    args = parser.parse_args()
    checks = []
    # The parent has just checked; the first repeated check is 15 min later.
    next_check = time.monotonic()+900
    while True:
        time.sleep(min(30, max(0,next_check-time.monotonic())))
        if time.monotonic() < next_check:
            continue
        queue = subprocess.run(['squeue','-h','-j',str(args.job),'-o','%T'],
                               capture_output=True, text=True)
        row = dict(at=datetime.now(timezone.utc).isoformat(), job=args.job,
                   state=queue.stdout.strip(), queue_exit=queue.returncode)
        progress = args.folder/'progress.json'
        if progress.exists():
            row['progress'] = json.loads(progress.read_text())
        checks.append(row)
        atomic_save(args.folder/'monitor_checks.json',checks)
        print(json.dumps(row),flush=True)
        if (args.folder/'complete.json').exists():
            subprocess.run([sys.executable,str(ROOT/'scripts/analyze_c1_gradient_closed_loop.py'),
                            str(args.folder)],check=True)
            return
        if queue.returncode==0 and not queue.stdout.strip():
            atomic_save(args.folder/'monitor_stopped.json',dict(reason='job ended without complete result',**row))
            return
        next_check = time.monotonic()+900


if __name__=='__main__':
    main()
