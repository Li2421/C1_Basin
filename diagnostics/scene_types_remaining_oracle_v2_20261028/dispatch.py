"""Durable stage execution; unfinished exact requests are the only retry unit."""
import os
os.environ['STB_COORDINATOR']='1'
from runner import *
import datetime
from resource_watch import allowance,jobs

PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'

def status(stage,complete,total,extra=''):
    current=datetime.datetime.now().astimezone().isoformat()
    rows=list((HERE/'raw').glob('*.json'))if(HERE/'raw').exists()else[]
    dump(HERE/'LIVE_STATUS.json',dict(time=current,stage=stage,stage_complete=complete,stage_total=total,new_complete=len(rows),resource_limit=allowance(),extra=extra))
    (HERE/'STATUS.md').write_text(f'# 当前状态\n\n{current}：{stage}，完成 {complete}/{total}，本轮已有 {len(rows)} 局完整结果。CPU 用户总上限 {allowance()}，GPU 0。\n\n{extra}\n\n旧实验目录只读。数值 unknown 保留，不自动重跑；基础设施失败仅补交未完成 UID，每阶段最多三次。补查为事后救援筛选，保留原留出结论。\n')

def execute(cases,stage,worker_entry='runner.py'):
    if (HERE/'PROTECTION_VIOLATION.jsonl').exists():raise RuntimeError('HARD_STOP protection event')
    cases,todo,done=register(cases,stage)
    if worker_entry=='runner.py' and cases and all(c['controller_payload'].get('exact_acceleration') for c in cases):worker_entry='worker_accel.py'
    execution=HERE/f'{stage}_execution.json'
    if execution.exists()and read(execution).get('complete'):
        return [read(HERE/'raw'/f"{c['rollout_uid']}.json")if(HERE/'raw'/f"{c['rollout_uid']}.json").exists()else done[c['rollout_uid']]for c in cases]
    if len(list((HERE/'attempts').glob('*.json'))) + len(todo)>256:raise RuntimeError('HARD_STOP rollout budget')
    t0=time.monotonic();attempts=[]
    for retry in range(4):
        todo=[c for c in todo if not(HERE/'raw'/f"{c['rollout_uid']}.json").exists()]
        if not todo:break
        other=sum(j['cpus']for j in jobs()if not j['name'].startswith('str_')and j['state']in('RUNNING','CONFIGURING','COMPLETING'))
        workers=min(len(todo),max(0,allowance()-other))
        if not workers:
            status(stage,len(cases)-len(todo),len(cases),'等待用户总 CPU 额度释放；其他作业不撤销。');time.sleep(30);continue
        shards=[[]for _ in range(workers)];load=[0.]*workers
        ordered=sorted(todo,key=lambda c:(-c['scenario_payload']['config']['max_steps']*c['state']['N']**1.5,c['rollout_uid']))
        for c in ordered:
            k=min(range(workers),key=lambda i:load[i]);shards[k].append(c)
            load[k]+=c['scenario_payload']['config']['max_steps']*c['state']['N']**1.5
        qname=f'{stage}_retry{retry}';dump(HERE/'queues'/f'{qname}.json',dict(shards=shards,longest_first=True))
        logs=HERE/'slurm';logs.mkdir(exist_ok=True)
        script=HERE/'worker.sbatch'
        script.write_text('#!/bin/bash\nset -euo pipefail\ncd /home/zhihan/research/Basin_C1\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu\nexec '+PYTHON+' diagnostics/scene_types_remaining_oracle_v2_20261028/'+worker_entry+' "$1" "$SLURM_ARRAY_TASK_ID"\n')
        name=f'str_{stage[:18]}_r{retry}'
        result=subprocess.check_output(['sbatch','--parsable','-p','cpu','--job-name',name,'--cpus-per-task=1','--mem=1800M','--time=06:00:00',f'--array=0-{workers-1}',f'--output={logs}/%A_%a.out',str(script),qname],text=True)
        jobid=result.strip().split(';')[0];attempts.append(dict(retry=retry,jobid=jobid,workers=workers,missing=len(todo)))
        dump(HERE/f'{stage}_jobs.json',attempts);print(stage,'SUBMITTED',jobid,'workers',workers,'missing',len(todo),flush=True)
        pending_since=time.monotonic();observed_running=False;last_status=0.
        while True:
            if(HERE/'PROTECTION_VIOLATION.jsonl').exists():
                subprocess.run(['scancel','--name',name],check=True);raise RuntimeError('HARD_STOP protection event')
            current=subprocess.check_output(['squeue','-h','-j',jobid,'-o','%T|%C'],text=True).splitlines()
            complete=sum((HERE/'raw'/f"{c['rollout_uid']}.json").exists()for c in cases)
            running=sum(int(line.split('|')[1])for line in current if line.split('|')[0]=='RUNNING')
            if running and not observed_running:
                print(stage,'RUNNING',running,'CPUs',flush=True);observed_running=True
            if time.monotonic()-last_status>60:
                status(stage,complete,len(cases),f'作业 {jobid}：运行 {running} CPU，重试 {retry}/3；事后筛选不修改原维度结论。')
                print(stage,'progress',complete,'/',len(cases),'running',running,'elapsed',round(time.monotonic()-t0),flush=True);last_status=time.monotonic()
            if not current:break
            if not observed_running and time.monotonic()-pending_since>900:
                subprocess.run(['scancel',jobid],check=True)
                print(stage,'pending >15min; cancelled own job for shorter resubmission',flush=True);break
            time.sleep(10)
    merge(stage,cases)
    missing=[c['rollout_uid']for c in cases if not(HERE/'raw'/f"{c['rollout_uid']}.json").exists()and c['rollout_uid']not in done]
    dump(execution,dict(complete=not missing,requests=len(cases),new=len(cases)-len(done),elapsed=time.monotonic()-t0,missing=missing,attempts=attempts))
    rows=[read(HERE/'raw'/f"{c['rollout_uid']}.json")if(HERE/'raw'/f"{c['rollout_uid']}.json").exists()else done[c['rollout_uid']]for c in cases if c['rollout_uid']not in missing]
    with(HERE/'candidates.jsonl').open('a')as f:
        seen=set()
        # The authoritative unique stream is rebuilt by final analysis; this is
        # an append-only per-stage stream whose UID explicitly permits dedup.
        for r in rows:
            if r['rollout_uid']in seen:continue
            f.write(json.dumps(clean(r),ensure_ascii=False)+'\n');seen.add(r['rollout_uid'])
    status(stage,len(rows),len(cases),'阶段已合并并读回核验。'if not missing else'达到三次补交上限；未完成 UID 单列为证据缺口。')
    return rows
