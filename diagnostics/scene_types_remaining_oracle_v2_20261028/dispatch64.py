"""Durable finite per-context searches with18 actual CPU allocations."""
from core import *
from runner import merge
from resource_watch import allowance,jobs,enforce
import os,subprocess,time,datetime

PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
def result_path(g):return HERE/'oracle64_groups'/f"{g['family']}_{g['type_id']}_{g['controller']}_{g['state_index']}.json"

def execute_groups(protocol,groups,requests):
    assert os.environ.get('STB_COORDINATOR')=='1'
    start=time.monotonic();attempts=[];errors=0
    for retry in range(4):
        pending=[g for g in groups if not result_path(g).exists()]
        if not pending:break
        other=sum(j['cpus'] for j in jobs() if not j['name'].startswith('str_') and j['state'] in ('RUNNING','CONFIGURING','COMPLETING'))
        workers=min(len(pending),max(0,allowance()-other))
        if not workers:raise RuntimeError('CPU quota currently occupied: resume later; other jobs preserved')
        shards=[[]for _ in range(workers)];loads=[0.]*workers
        def estimate(g):
            missing=[x for x in g['candidates'] if not Path(x['path']).exists()]
            if not missing:return 0.
            c=requests[missing[0]['uid']]
            return len(missing)*c['scenario_payload']['config']['max_steps']*c['state']['N']**1.5
        for g in sorted(pending,key=estimate,reverse=True):
            k=min(range(workers),key=lambda j:loads[j]);shards[k].append(g);loads[k]+=estimate(g)
        stage=f'ORACLE64_r{retry}'
        dump(HERE/'queues'/f'{stage}.json',dict(shards=shards,requests=requests))
        script=HERE/'worker64.sbatch'
        script.write_text('#!/bin/bash\nset -euo pipefail\ncd /home/zhihan/research/Basin_C1\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu\nexec '+PYTHON+' diagnostics/scene_types_remaining_oracle_v2_20261028/worker64.py "$1" "$SLURM_ARRAY_TASK_ID"\n')
        log=HERE/'slurm';log.mkdir(exist_ok=True)
        job=subprocess.check_output(['sbatch','--parsable','-p','cpu','--job-name','str_oracle64','--cpus-per-task=1','--mem=2200M','--time=06:00:00',f'--array=0-{workers-1}',f'--output={log}/%A_%a.out',str(script),stage],text=True).strip().split(';')[0]
        attempts.append(dict(job_id=job,workers=workers,pending_groups=len(pending),retry=retry));dump(HERE/'ORACLE64_JOBS.json',attempts)
        print('ORACLE64 SUBMITTED',job,'workers',workers,'groups',len(pending),flush=True)
        last=0.
        while True:
            if(HERE/'PROTECTION_VIOLATION.jsonl').exists():
                subprocess.run(['scancel',job],check=True);raise RuntimeError('protection violation')
            current=subprocess.check_output(['squeue','-h','-j',job,'-o','%T|%C'],text=True).splitlines()
            if time.monotonic()-last>=30:
                resource=enforce()
                completed=[read(result_path(g)) for g in groups if result_path(g).exists()]
                out=dict(time=datetime.datetime.now().astimezone().isoformat(),complete_groups=len(completed),total_groups=len(groups),new_raw_total=len(list((HERE/'raw').glob('*.json'))),new_attempts_total=len(list((HERE/'attempts').glob('*.json'))),actual_cpu=sum(int(x.split('|')[1])for x in current if x.startswith('RUNNING|')),elapsed_seconds=time.monotonic()-start,per_family={f['index']:dict(complete=sum(r['family']==f['index']for r in completed),success=sum(r['family']==f['index']and r['first_success']is not None for r in completed))for f in protocol['families']})
                dump(HERE/'ORACLE64_LIVE.json',out)
                print('ORACLE64',out['complete_groups'],'/',len(groups),'raw',out['new_raw_total'],'cpu',out['actual_cpu'],'elapsed',round(out['elapsed_seconds']),flush=True);last=time.monotonic()
            if not current:break
            time.sleep(10)
        # Single coordinator ingestion also preserves partial batch evidence.
        merge(stage,requests.values())
    missing=[dict(family=g['family'],type_id=g['type_id'],controller=g['controller'],state_index=g['state_index']) for g in groups if not result_path(g).exists()]
    dump(HERE/'ORACLE64_EXECUTION.json',dict(complete=not missing,groups=len(groups),missing=missing,jobs=attempts,elapsed_seconds=time.monotonic()-start))
    if missing:raise RuntimeError('Infrastructure retries exhausted; resume only missing contexts')

