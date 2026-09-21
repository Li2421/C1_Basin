"""Sparse, durable observation of the two authorized Slurm jobs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from single_integrator.c1.training.persistence import atomic_save


def main():
    history=[]
    output=ROOT/'results/c1_exact_pipeline_monitor.json'
    while True:
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            time.sleep(min(30,max(0,deadline-time.monotonic())))
        q=subprocess.run(['squeue','-h','-u','zhihan','-o','%i %j %T'],capture_output=True,text=True)
        row=dict(at=datetime.now(timezone.utc).isoformat(),queue=q.stdout.strip(),queue_exit=q.returncode)
        for name in ('c1_gradient_exact_v1','c1_exact_margin_v1'):
            folder=ROOT/'results'/name
            info={}
            for file in ('progress.json','training_complete.json','complete.json'):
                p=folder/file
                if p.exists():
                    data=json.loads(p.read_text())
                    info[file]={k:v for k,v in data.items() if k not in ('variants',)}
            for file in ('history.json','validation.json'):
                p=folder/file
                if p.exists():
                    data=json.loads(p.read_text())
                    if data:info[file]={k:v for k,v in data[-1].items() if k not in ('episodes',)}
            row[name]=info
        history.append(row);atomic_save(output,history)
        print(json.dumps(row),flush=True)
        if q.returncode==0 and not any(line.split()[0] in ('20','21') for line in q.stdout.splitlines()):
            return


if __name__=='__main__':main()
