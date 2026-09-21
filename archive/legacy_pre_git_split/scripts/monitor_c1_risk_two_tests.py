"""Read-only, 15-minute Slurm checks; never restarts jobs or launches training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',type=int,required=True)
    parser.add_argument('--folder',type=Path,required=True)
    args=parser.parse_args();checks=[]
    while True:
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            time.sleep(min(30,max(0,deadline-time.monotonic())))
        proc=subprocess.run(['scontrol','show','job',str(args.job)],capture_output=True,text=True)
        match=re.search(r'JobState=(\S+)',proc.stdout)
        state=match.group(1) if match else 'UNKNOWN'
        row=dict(at=datetime.now(timezone.utc).isoformat(),job=args.job,state=state,query_exit=proc.returncode)
        for filename,key in [('diagnostics.json','starts_diagnosed'),('records.json','replays')]:
            path=args.folder/filename
            if path.exists():
                try:row[key]=len(json.loads(path.read_text()))
                except (OSError,json.JSONDecodeError):row[key]='temporarily_unreadable'
        summary=args.folder/'analysis.json'
        if summary.exists():
            try:row['gates']=json.loads(summary.read_text())['gates']
            except (OSError,json.JSONDecodeError,KeyError):pass
        checks.append(row)
        target=args.folder/'monitor_checks.json';temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(checks,indent=2)+'\n');temporary.replace(target)
        print(json.dumps(row),flush=True)
        if state in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}:
            return
        if state=='UNKNOWN':
            queue=subprocess.run(['squeue','-h','-j',str(args.job),'-o','%T'],capture_output=True,text=True)
            # Absence plus a saved complete analysis suffices; transient errors
            # or missing output alone never trigger a replacement experiment.
            if queue.returncode==0 and not queue.stdout.strip() and 'gates' in row:return


if __name__=='__main__':main()
