"""Record one Slurm training pipeline every 900 seconds; stop at completion."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from single_integrator.c1.training.persistence import atomic_save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',required=True,type=int)
    p.add_argument('--folder',required=True,type=Path)
    args=p.parse_args();records=[]
    while True:
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:time.sleep(min(30,max(0,deadline-time.monotonic())))
        q=subprocess.run(['squeue','-h','-u','zhihan','-o','%i %T'],capture_output=True,text=True)
        state=next((s for s in q.stdout.splitlines() if s.split()[0]==str(args.job)),'not in queue')
        row=dict(at=datetime.now(timezone.utc).isoformat(),job=args.job,state=state)
        for name in ('history.json','validation.json','feasibility_stage.json','training_complete.json','complete.json'):
            f=args.folder/name
            if f.exists():
                v=json.loads(f.read_text());v=v[-1] if isinstance(v,list) and v else v
                if isinstance(v,dict):row[name]={k:x for k,x in v.items() if k not in ('episodes','rids')}
        records.append(row)
        if args.folder.exists():atomic_save(args.folder/'monitor_checks.json',records)
        print(json.dumps(row),flush=True)
        if (args.folder/'complete.json').exists() or (q.returncode==0 and state=='not in queue'):return


if __name__=='__main__':main()
