"""Durable preflighted stages, global CPU accounting and bounded infra retries."""
import os
os.environ['STB_COORDINATOR'] = '1'
from runner import *
from budget import LIMIT, used
from resource_watch import allowance, jobs, cancel_own, enforce, available as available_cpus, task_allowance
import datetime
import shlex
import subprocess

PYTHON = '/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'


class BudgetExhausted(RuntimeError):
    pass


class DeadlineReached(RuntimeError):
    pass


def status(stage, complete, total, note=''):
    current = datetime.datetime.now().astimezone().isoformat()
    active = sum(row['cpus'] for row in jobs() if row['state'] in ('RUNNING', 'CONFIGURING', 'COMPLETING'))
    dump(HERE / 'LIVE_STATUS.json', dict(time=current, stage=stage, stage_complete=complete, stage_total=total,
        actual_attempts=used(), hard_limit=LIMIT, active_user_cpus=active, cpu_limit=allowance(), gpu=0, note=note))
    estimate = read(HERE / 'ETA.json') if (HERE / 'ETA.json').exists() else {}
    warning = '主线预计超过30小时；按协议继续运行，等待用户决定是否调整份额。\n\n' if estimate.get('main_hours', 0) > 30 else ''
    (HERE / 'STATUS.md').write_text(warning + '# 当前状态\n\n' + current + f'：{stage}，{complete}/{total}。新增尝试{used()}/{LIMIT}，全部用户任务CPU{active}/{allowance()}，GPU0。\n\n' + note + '\n\n' + estimate.get('description', '预计总时长将在各方法训练场景试跑后按实测成本更新。') + '\n\n缺失、求解器异常和超时分开记录；不因测试结果重跑。\n')


def wait_job(jobid, cases, done, stage, retry, deadline=None):
    last = 0.
    while True:
        enforce()
        if deadline is not None and time.time()>=deadline:
            cancel_own(jobid)
            dump(HERE/'deadline_events'/(stage+'.json'),dict(time=time.time(),deadline=deadline,jobid=jobid,
                 reason='User conditional timing supplement limit; incomplete queries remain missing, not failures.'))
            return False
        if (HERE / 'PROTECTION_VIOLATION.jsonl').exists():
            cancel_own(jobid)
            raise RuntimeError('HARD_STOP: protection violation')
        live = [row for row in jobs() if row['jobid'] == str(jobid) or row['jobid'].split('_')[0] == str(jobid)]
        assert all(row['ours'] for row in live), ('Stage points at another task', jobid)
        complete = sum(c['rollout_uid'] in done or (HERE / 'raw' / (c['rollout_uid'] + '.json')).exists() for c in cases)
        if time.monotonic() - last >= 30:
            status(stage, complete, len(cases), f'作业{jobid}，基础设施补交{retry}/3。只提交缺失UID；已有数值结局不重跑。')
            print(stage, 'progress', complete, '/', len(cases), 'job', jobid, flush=True)
            last = time.monotonic()
        if not live:
            return True
        time.sleep(10)


def execute(cases, stage, deadline=None):
    if (HERE / 'PROTECTION_VIOLATION.jsonl').exists():
        raise RuntimeError('HARD_STOP: protection violation')
    cases, todo, done = register(cases, stage)
    execution = HERE / 'plans' / (stage + '_execution.json')
    jobs_path = HERE / 'plans' / (stage + '_jobs.json')
    attempts = read(jobs_path) if jobs_path.exists() else []
    start = time.monotonic()
    # On coordinator restart, adopt existing stage workers before considering a
    # retry. UID locks are a second guard, not a license for duplicate submission.
    current = jobs()
    deadline_stop=False
    for attempt in attempts:
        if any(row['jobid'].split('_')[0] == str(attempt['jobid']) for row in current):
            if not wait_job(attempt['jobid'], cases, done, stage, attempt['retry'],deadline):deadline_stop=True
    merge(cases)
    cases, todo, done = register(cases, stage)
    budget_stop = False
    for retry in range(len(attempts), 4):
        if not todo:
            break
        if deadline_stop or(deadline is not None and time.time()>=deadline):
            deadline_stop=True
            break
        remaining = LIMIT - used()
        if remaining <= 0:
            budget_stop = True
            break
        submitted = todo[:remaining]
        available = 0
        while available <= 0:
            if deadline is not None and time.time()>=deadline:
                deadline_stop=True
                break
            other = sum(row['cpus'] for row in jobs() if row['state'] in ('RUNNING', 'CONFIGURING', 'COMPLETING'))
            available = available_cpus()
            if not available:
                status(stage, len(done), len(cases), '等待全部用户任务的CPU余量，不撤销其他作业。')
                time.sleep(30)
        if deadline_stop:break
        count = min(len(submitted), available)
        shards = [[] for _ in range(count)]
        loads = [0.] * count
        ordered = sorted(submitted, key=lambda c: (-c['scenario_payload']['config']['max_steps'] * c['state']['N'] ** 1.5, c['rollout_uid']))
        for request in ordered:
            target = min(range(count), key=lambda i: loads[i])
            shards[target].append(request)
            loads[target] += request['scenario_payload']['config']['max_steps'] * request['state']['N'] ** 1.5
        queue_name = stage + '_retry' + str(retry)
        dump(HERE / 'queues' / (queue_name + '.json'), dict(shards=shards, longest_first=True))
        logs = HERE / 'slurm'
        logs.mkdir(exist_ok=True)
        script = HERE / 'worker.sbatch'
        script.write_text('#!/bin/bash\nset -euo pipefail\ncd ' + shlex.quote(str(ROOT)) + '\n'
                          'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu\n'
                          'exec ' + shlex.quote(PYTHON) + ' ' + shlex.quote(str(HERE / 'worker.py')) + ' "$1" "$SLURM_ARRAY_TASK_ID"\n')
        job_name = 'pbp1_' + digest(dict(stage=stage))[:12] + '_r' + str(retry)
        answer = subprocess.check_output(['sbatch', '--parsable', '-p', 'cpu', '--job-name', job_name,
            '--cpus-per-task=1', '--mem=2200M', '--time=06:00:00', '--array=0-' + str(count - 1),
            '--output=' + str(logs / '%A_%a.out'), str(script), queue_name], text=True)
        jobid = answer.strip().split(';')[0]
        attempts.append(dict(retry=retry, jobid=jobid, workers=count, missing=len(todo), submitted=len(submitted)))
        dump(jobs_path, attempts)
        print(stage, 'SUBMITTED', jobid, 'workers', count, 'requests', len(submitted), flush=True)
        if not wait_job(jobid, cases, done, stage, retry,deadline):deadline_stop=True
        merge(cases)
        cases, todo, done = register(cases, stage)
        if deadline_stop:break
        if todo and used() >= LIMIT:
            budget_stop = True
            break
    rows = [done[c['rollout_uid']] for c in cases if c['rollout_uid'] in done]
    dump(execution, dict(complete=not todo, requests=len(cases), results=len(rows),
                         missing=[c['rollout_uid'] for c in todo], budget_stop=budget_stop,
                         user_deadline_stop=deadline_stop,
                         attempts=attempts, elapsed=time.monotonic() - start))
    status(stage, len(rows), len(cases), '阶段已合并并读回核验。' if not todo else '未完成项单列缺口，不当作失败。')
    if budget_stop:
        raise BudgetExhausted('HARD_BUDGET_REACHED:' + stage)
    if deadline_stop:
        raise DeadlineReached('USER_TIMING_SUPPLEMENT_DEADLINE:'+stage)
    return rows
