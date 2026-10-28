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
    eta=read(HERE/'ETA.json')if(HERE/'ETA.json').exists()else{}
    estimate=eta.get('description','预计剩余时间：等待主线吞吐样本；成功即停使实际运行量尚未确定。')
    (HERE/'STATUS.md').write_text(f'# 当前状态\n\n{current}：{stage}，完成 {complete}/{total}，本轮已有 {len(rows)} 局完整结果。CPU 用户总上限 {allowance()}，GPU 0。\n\n{extra}\n\n{estimate}\n\n旧实验目录只读。数值 unknown 保留，不自动重跑；基础设施失败仅补交未完成 UID，每阶段最多三次。不选族、不判维度；用户追加的论文对照排在四列阶段之后。\n')

def execute(cases,stage,worker_entry='worker_recorded.py'):
    if (HERE/'PROTECTION_VIOLATION.jsonl').exists():raise RuntimeError('HARD_STOP protection event')
    cases,todo,done=register(cases,stage)
    if worker_entry in ('runner.py','worker_recorded.py') and cases and all(c['controller_payload'].get('exact_acceleration') for c in cases):worker_entry='worker_accel.py'
    execution=HERE/f'{stage}_execution.json'
    if execution.exists()and read(execution).get('complete'):
        return [read(HERE/'raw'/f"{c['rollout_uid']}.json")if(HERE/'raw'/f"{c['rollout_uid']}.json").exists()else done[c['rollout_uid']]for c in cases]
    from budget import used
    if used()>=90000 and todo:raise RuntimeError('HARD_STOP actual new rollout budget exhausted (90000)')
    jobs_path=HERE/f'{stage}_jobs.json'
    previous=read(jobs_path)if jobs_path.exists()else[]
    first_new=previous[0]['missing']if previous else len(todo)
    invocation_new=len(todo)
    t0=time.monotonic();attempts=list(previous)
    for retry in range(len(previous),4):
        if not todo:break
        if retry:
            # Recover a completed journal even if the worker died before its
            # final raw-file rename. Never spend another rollout on that label.
            merge(stage+'_infra_recovery',cases)
            cases,new_todo,new_done=register(cases,stage)
            done.update(new_done);todo=new_todo
        todo=[c for c in todo if not(HERE/'raw'/f"{c['rollout_uid']}.json").exists()]
        if not todo:break
        remaining=90000-used()
        if remaining<=0:raise RuntimeError('HARD_STOP actual new rollout budget exhausted (90000)')
        submission_todo=todo[:remaining]
        # Count concurrent stages in this experiment as well as other tasks.
        # This coordinator has no live stage workers at the submission point.
        other=sum(j['cpus']for j in jobs()if j['state']in('RUNNING','CONFIGURING','COMPLETING'))
        workers=min(len(submission_todo),max(0,allowance()-other))
        if not workers:
            while not workers:
                status(stage,len(cases)-len(todo),len(cases),'等待用户总 CPU 额度释放；其他作业不撤销；等待不计为基础设施重试。')
                time.sleep(30)
                other=sum(j['cpus']for j in jobs()if j['state']in('RUNNING','CONFIGURING','COMPLETING'))
                workers=min(len(submission_todo),max(0,allowance()-other))
        shards=[[]for _ in range(workers)];load=[0.]*workers
        ordered=sorted(submission_todo,key=lambda c:(-c['scenario_payload']['config']['max_steps']*c['state']['N']**1.5,c['rollout_uid']))
        for c in ordered:
            k=min(range(workers),key=lambda i:load[i]);shards[k].append(c)
            load[k]+=c['scenario_payload']['config']['max_steps']*c['state']['N']**1.5
        qname=f'{stage}_retry{retry}';dump(HERE/'queues'/f'{qname}.json',dict(shards=shards,longest_first=True))
        # Keep the complete archive, but each worker parses only its own shard.
        # This changes neither candidate order nor numerical execution.
        for shard_index,shard in enumerate(shards):
            private_shards=[[]for _ in shards];private_shards[shard_index]=shard
            dump(HERE/'queues'/f'{qname}_{shard_index}.json',dict(shards=private_shards,longest_first=True))
        logs=HERE/'slurm';logs.mkdir(exist_ok=True)
        script=HERE/'worker.sbatch'
        script.write_text('#!/bin/bash\nset -euo pipefail\ncd /home/zhihan/research/Basin_C1\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu\nexec '+PYTHON+' diagnostics/scene_types_oracle_coverage_v3/'+worker_entry+' "${1}_${SLURM_ARRAY_TASK_ID}" "$SLURM_ARRAY_TASK_ID"\n')
        name=f'stb3_{stage[:18]}_r{retry}'
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
                new_complete=max(0,complete-len(done));elapsed=time.monotonic()-t0
                if new_complete>=20 and elapsed>=30:
                    rate=new_complete/elapsed
                    budget_hours=max(0,90000-used())/rate/3600
                    dump(HERE/'ETA.json',dict(stage=stage,new_per_second=rate,hours_to_budget_at_current_rate=budget_hours,
                        description=f'预计本批剩余约{(len(cases)-complete)/rate/60:.1f}分钟；按本批吞吐耗尽剩余新rollout预算约{budget_hours:.1f}小时。后续场景成本不同，成功即停与旧证据复用会改变实际总量，完整交付时间尚待收敛。'))
                status(stage,complete,len(cases),f'作业 {jobid}：运行 {running} CPU，重试 {retry}/3；覆盖数据随搜索更新，尚未完成项不算64次失败。')
                print(stage,'progress',complete,'/',len(cases),'running',running,'elapsed',round(time.monotonic()-t0),flush=True);last_status=time.monotonic()
            if not current:break
            if not observed_running and time.monotonic()-pending_since>900:
                subprocess.run(['scancel',jobid],check=True)
                print(stage,'pending >15min; cancelled own job for shorter resubmission',flush=True);break
            time.sleep(10)
    merge(stage,cases)
    missing=[c['rollout_uid']for c in cases if not(HERE/'raw'/f"{c['rollout_uid']}.json").exists()and c['rollout_uid']not in done]
    dump(execution,dict(complete=not missing,requests=len(cases),new=first_new,
        new_requested_at_first_submission=first_new,new_unique_completed=max(0,first_new-len(missing)),
        new_requested_this_invocation=invocation_new,
        elapsed=time.monotonic()-t0,missing=missing,attempts=attempts))
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
