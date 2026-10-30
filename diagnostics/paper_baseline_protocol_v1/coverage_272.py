"""User-prioritized historical-cohort coverage; frozen paper numerics, SSD outputs.

This is not the preregistered fresh-state generalization test. The only runtime
changes are a narrowly scoped artifact-write permission and paper runtime path.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import guard
from guard import HERE, ROOT
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

CONFIG = HERE / 'COVERAGE_272_PARAMETERS_FROZEN.json'
POLICY = json.loads(CONFIG.read_text())
ART = Path(POLICY['artifacts'])
assert Path('/mnt/ssd').is_mount(), 'No SSD mount: refuse home fallback'
assert ART == Path('/mnt/ssd/zhihan/Basin_C1/diagnostics/paper_baseline_protocol_v1/coverage_272_artifacts')
assert (HERE / 'coverage_272_artifacts').is_symlink()
assert (HERE / 'coverage_272_artifacts').resolve() == ART and ART.is_dir()
assert ART.stat().st_dev != HERE.stat().st_dev
_original_allowed = guard.allowed


def storage_allowed(path):
    if isinstance(path, int):
        return True
    resolved = Path(os.fsdecode(path)).resolve()
    return resolved.is_relative_to(ART) or _original_allowed(path)


guard.allowed = storage_allowed


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE) or path.resolve().is_relative_to(ART)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_policy():
    assert sha(HERE / 'EXECUTION_FROZEN.json') == POLICY['numeric_release_sha256']
    for method, choice in POLICY['methods'].items():
        assert sha(ROOT / choice['training_source']) == choice['training_sha256']
        assert load(ROOT / choice['training_source'])['global_index'] == choice['global_index']
    frozen = HERE / 'COVERAGE_272_CODE_FROZEN.json'
    receipt = load(frozen)
    assert receipt['parameters_sha256'] == sha(CONFIG)
    assert receipt['source_sha256'] == sha(Path(__file__))


def requests():
    from request_factory import paper
    from scene_catalog import catalog
    representative = load(HERE / POLICY['scenes_source'])
    scenes = catalog('B')
    cases = []
    for method, choice in POLICY['methods'].items():
        for entry in representative['selected']:
            scene = scenes[entry['scene']]
            for state in POLICY['states']:
                request = paper('B', scene, state, method, choice['options'],
                    phase='exploratory_historical_coverage', purpose='user_priority_fixed_tuned_historical272')
                cases.append(request)
    assert len(cases) == 272
    assert len({(r['scene'], r['state_index']) for r in cases}) == 136
    return cases


def child(request_path):
    os.environ.pop('STB_COORDINATOR', None)
    verify_policy()
    import runner
    from budget import reserve
    from core import clean, np, cache_writer
    import paper_episode
    # This module global is used only to select the runtime output directory.
    # Adapter paths, controller, solver, task, state and random seed stay frozen.
    paper_episode.HERE = ART
    request = load(request_path)
    reserve(request)
    start = time.monotonic()
    result, arrays = runner.execute_numerical(request)
    cfg = request['scenario_payload']['config']
    count = int(result['episode_length'])
    result = dict(result, completion_time=count * cfg['dt'] if result['success'] else None,
        truncated_time=count * cfg['dt'] if result['success'] else cfg['max_steps'] * cfg['dt'],
        horizon_time=cfg['max_steps'] * cfg['dt'])
    trace = ART / 'traces' / (request['rollout_uid'] + '.npz')
    trace.parent.mkdir(exist_ok=True)
    temporary = trace.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(trace)
    for key in ('scenario_uid', 'state_uid', 'controller_uid', 'eta_uid', 'seed_key', 'rollout_uid',
                'scene', 'type_id', 'state_index', 'seed', 'spec', 'purpose', 'solo', 'benchmark', 'phase'):
        result[key] = request[key]
    result.update(controller=request['controller_payload']['nominal']['type_code'], eta=request['eta_a'],
        future_index=request['seed'], cache_identity_schema='registered_exact_uid_v1',
        initial_positions=request['state']['positions'], initial_velocities=request['state']['velocities'],
        goals=request['state']['goals'], deadlock=0, source='new_rollout', trace=str(trace), trace_sha256=sha(trace),
        worker_cpu=list(os.sched_getaffinity(0)), wall_seconds=time.monotonic() - start,
        coverage_policy_sha256=sha(CONFIG), coverage_code_freeze_sha256=sha(HERE/'COVERAGE_272_CODE_FROZEN.json'))
    result = clean(result)
    cache_writer.append_journal([result], runner.EXP)
    save(ART / 'raw' / (request['rollout_uid'] + '.json'), result)
    print(request['controller_payload']['paper_method'], request['scene'], request['state_index'], result['outcome'], flush=True)


def worker(index):
    import fcntl
    queue = load(ART / 'queue.json')
    local = queue['shards'][index]
    mine = {r['rollout_uid'] for r in local}
    other = [r for shard in queue['shards'] for r in shard if r['rollout_uid'] not in mine]
    (ART / 'locks').mkdir(exist_ok=True)
    for request in local + other:
        uid = request['rollout_uid']
        if (ART / 'raw' / (uid + '.json')).exists():
            continue
        with (ART / 'locks' / (uid + '.lock')).open('a+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if (ART / 'raw' / (uid + '.json')).exists():
                continue
            target = ART / 'requests' / (uid + '.json')
            save(target, request)
            proc = subprocess.run([sys.executable, '-B', str(Path(__file__)), 'child', str(target)])
            if proc.returncode and not (ART / 'raw' / (uid + '.json')).exists():
                raise RuntimeError(f'Infrastructure child failure before result: {uid}, code={proc.returncode}')


def summarize(cases, reused):
    from method_names import NAMES
    rows = []
    for request in cases:
        uid = request['rollout_uid']
        path = ART / 'raw' / (uid + '.json')
        result = load(path) if path.exists() else reused.get(uid)
        rows.append(dict(method=request['controller_payload']['paper_method'], scene=request['scene'],
            state=request['state_index'], type_id=request['type_id'], uid=uid, result=result))
    summaries = {}
    for method in POLICY['methods']:
        selected = [r for r in rows if r['method'] == method]
        values = [r['result'] for r in selected if r['result'] is not None]
        successful = [r for r in values if r['success']]
        by_scene = {}
        for scene in sorted({r['scene'] for r in selected}):
            vv = [r['result'] for r in selected if r['scene'] == scene and r['result'] is not None]
            by_scene[scene] = dict(complete=len(vv), expected=8, success=sum(int(v['success']) for v in vv))
        summaries[method] = dict(name=NAMES[method], selected_index=POLICY['methods'][method]['global_index'],
            complete=len(values), expected=136, success=len(successful),
            success_rate=len(successful)/136 if len(values)==136 else None,
            collision=sum(int(v.get('collision',0)) for v in values),
            timeout=sum(int(v.get('timeout',0)) for v in values),
            numerical_unknown=sum(int(v.get('numerical_failure',0)) for v in values),
            native_unhandled_episodes=sum(bool(v.get('paper_original',{}).get('numerical_unknown')) for v in values),
            success_mean_seconds=sum(v['completion_time'] for v in successful)/len(successful) if successful else None,
            all_queue_capped_mean_seconds=sum(v['truncated_time'] for v in values)/136 if len(values)==136 else None,
            by_scene=by_scene)
    out=dict(time=datetime.datetime.now().astimezone().isoformat(),complete=all(v['complete']==136 for v in summaries.values()),
        physical_states=136, corresponding_c1_nominal_cases=272, methods=summaries,
        parameters_sha256=sha(CONFIG),generalization_claim=False,
        caveat='Fixed tuned external policies versus previously explored historical cohort. Not external per-case Oracle and not a fresh-state generalization test.')
    save(HERE / 'COVERAGE_272_RESULTS.json', out)
    lines=['# 已调参外部方法：历史272案例对应物理初态覆盖', '', out['caveat'], '',
        '每种外部方法136个物理初态；C1的272回合包含两个名义控制器，不将外部结果复制成272次独立运行。', '',
        '| 方法 | 已完成 | 成功 | 成功率 | 成功平均时间/s | 全队列截断时间/s | 碰撞 | 超时 | 终局数值未知 | 曾未处理异常 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v): return '待完成' if v is None else f'{v:.3f}'
    for v in summaries.values():
        sr='待完成' if v['success_rate'] is None else f"{100*v['success_rate']:.2f}%"
        lines.append(f"| {v['name']} | {v['complete']}/136 | {v['success']} | {sr} | {fmt(v['success_mean_seconds'])} | {fmt(v['all_queue_capped_mean_seconds'])} | {v['collision']} | {v['timeout']} | {v['numerical_unknown']} | {v['native_unhandled_episodes']} |")
    (HERE/'COVERAGE_272_REPORT.md').write_text('\n'.join(lines)+'\n')
    return out


def coordinator():
    os.environ['STB_COORDINATOR']='1'
    verify_policy()
    from runner import register, merge
    from dispatch import status
    from resource_watch import jobs, available, cancel_own, enforce
    cases, todo, done = register(requests(), 'user_priority_historical272_fixed_tuned')
    manifest = dict(requests=[{k:r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key')} for r in cases], exact_configs=cases)
    save(ART / 'planned.json',manifest)
    # Merge durable prior child journals on restart, before consulting the DB.
    merge([])
    cases,todo,done=register(cases,'user_priority_historical272_fixed_tuned')
    active_path=HERE/'COVERAGE_272_ACTIVE_JOB.json'
    prior=load(active_path) if active_path.exists() else dict(attempts=[])
    for retry in range(4):
        remaining=[r for r in todo if not (ART/'raw'/(r['rollout_uid']+'.json')).exists()]
        if not remaining:break
        old=[row for row in jobs() if any(row['jobid'].split('_')[0]==a['jobid'] for a in prior['attempts'])]
        if old:
            jobid=old[0]['jobid'].split('_')[0]
        else:
            workers=min(17,len(remaining),available())
            assert workers>0, 'No CPU allocation available'
            shards=[[] for _ in range(workers)];loads=[0]*workers
            for r in sorted(remaining,key=lambda x:-x['scenario_payload']['config']['max_steps']*x['state']['N']**1.5):
                k=min(range(workers),key=lambda i:loads[i]);shards[k].append(r)
                loads[k]+=r['scenario_payload']['config']['max_steps']*r['state']['N']**1.5
            save(ART/'queue.json',dict(shards=shards))
            (ART/'slurm').mkdir(exist_ok=True)
            answer=subprocess.check_output(['sbatch','--parsable','-p','cpu','--job-name=pbp1_cov272_workers',
                '--cpus-per-task=1','--mem=2200M','--time=02:00:00',f'--array=0-{workers-1}',
                '--output='+str(ART/'slurm/%A_%a.out'),str(HERE/'coverage_272_worker.sbatch')],text=True)
            jobid=answer.strip().split(';')[0]
            prior['attempts'].append(dict(jobid=jobid,retry=retry,workers=workers,submitted=len(remaining)))
            save(active_path,prior)
        while any(r['jobid'].split('_')[0]==jobid for r in jobs()):
            enforce()
            if (HERE/'PROTECTION_VIOLATION.jsonl').exists():
                cancel_own(jobid);raise RuntimeError('HARD_STOP: protection violation')
            report=summarize(cases,done)
            n=sum(v['complete'] for v in report['methods'].values())
            status('user_priority_historical272_fixed_tuned',n,len(cases),'已暂停C1调参与泛化测试；固定已选外部参数，历史136物理初态/方法；全新大型产物在SSD。')
            print('coverage272',n,'/',len(cases),'job',jobid,flush=True)
            time.sleep(20)
        merge([])
        cases,todo,done=register(cases,'user_priority_historical272_fixed_tuned')
    report=summarize(cases,done)
    missing=[r['rollout_uid'] for r in cases if not (ART/'raw'/(r['rollout_uid']+'.json')).exists() and r['rollout_uid'] not in done]
    save(HERE/'COVERAGE_272_COMPLETE.json',dict(time=report['time'],complete=report['complete'],missing=missing,
        frozen_numeric_sources_unchanged=all(sha(HERE/name)==expected for name,expected in load(HERE/'EXECUTION_FROZEN.json')['files'].items()),
        other_pipeline_remains_paused=True,raw=str(ART/'raw'),traces=str(ART/'traces')))
    status('user_priority_historical272_fixed_tuned_complete',272-len(missing),272,'历史覆盖结果已保存；不自动恢复原泛化流程。')
    return report


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='coordinator':coordinator()
    elif mode=='worker':worker(int(sys.argv[2]))
    elif mode=='child':child(sys.argv[2])
    else:raise ValueError(mode)
