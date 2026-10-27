"""Authorized autonomous completion of part-one computation, never starts part two."""
from pathlib import Path
import json,os,sys,subprocess,time,fcntl
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
PY=sys.executable
def write(name,value):
    p=HERE/name;p.parent.mkdir(exist_ok=True,parents=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def call(script,*args):
    path=HERE/'driver_actions'/f'{time.time_ns()}_{script}.log';path.parent.mkdir(exist_ok=True)
    with path.open('w') as stream:
        result=subprocess.run([PY,'-B',str(HERE/script),*map(str,args)],cwd=HERE.parents[1],stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError((script,args,result.returncode,str(path),path.read_text()[-2500:]))
    return path
def guards(label):
    try:
        call('resume_audit.py',label);call('other_readonly_guard.py',label)
    except Exception as exc:
        write('RESUME_FATAL_STOP.json',dict(utc=datetime.now(timezone.utc).isoformat(),reason=str(exc)))
        for r in jobs():subprocess.run(['scancel',r['id']],check=True)
        raise
def jobs():
    sys.path.insert(0,str(HERE))
    from resume_resources import jobs as all_jobs
    return [r for r in all_jobs() if r['user']==os.environ.get('USER','zhihan') and r['name']=='lit_resume_science']
def progress(phase,stages):
    entries={}
    for stage in stages:
        root=HERE/stage/'raw';entries[stage]=sum(1 for p in root.glob('*.json')) if root.exists() else 0
    write('PART1_PROGRESS.json',dict(utc=datetime.now(timezone.utc).isoformat(),phase=phase,logical_rows=entries,jobs=jobs(),part2_started=False))
    (HERE/'STATUS.md').write_text('# 第一部分恢复执行中\n\n'+f'阶段：{phase}。逻辑结果量：{entries}。\n\n用户接受既往查询缓存限定例外；新库32条逐位回放通过，只读目录在各批前后核查。旧停机和失败审计保留。COMPLETE.json之前不启动第二部分。\n')
def wait_phase(phase,stages):
    folder=HERE/'RESUME_QUEUES'/phase
    for attempt in range(5):
        while jobs():
            progress(phase,stages);time.sleep(30)
        guards('after_'+phase+f'_{attempt}')
        unresolved=[]
        for item in json.loads((folder/'INDEX.json').read_text()):
            rows=json.loads(Path(item['path']).read_text())
            if any(not (HERE/c['output_stage']/'raw'/f"{c['rollout_uid']}.json").exists() for c in rows):unresolved.append(item)
        if not unresolved:progress(phase+'_complete',stages);return
        # Only unfinished infrastructure work is recovered; existing raw,
        # including every unknown, is retained and never executed again.
        archive=folder/'aborted_claims';archive.mkdir(exist_ok=True)
        for item in unresolved:
            claim=folder/'claims'/item['group']
            if claim.exists():claim.rename(archive/(item['group']+'_'+str(time.time_ns())))
        with (HERE/'ITERATION_LOG.md').open('a') as f:f.write(f'\n基础设施恢复：{phase}第{attempt+1}次，{len(unresolved)}个未完成组；已有raw/unknown不重跑。\n')
        call('dispatch_resume.py',phase)
    raise RuntimeError('Five infrastructure recoveries exhausted; inspect worker logs')
def main():
    with (HERE/'PART1_DRIVER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write('PART1_DRIVER_PID.json',dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat()))
        stages=['lit_narrow_passage_initial','lit_crowded_random_initial','lit_circle_five_fallback']
        wait_phase('pilot',stages)
        selected={};fallback=[]
        for stage in stages:
            call('resume_pilot_summary.py',stage);r=json.loads((HERE/f'{stage}_PILOT_RESULTS_RESUME.json').read_text());scene=stage.rsplit('_',1)[0]
            if r['passed']:selected[scene]=dict(stage=stage,status='accepted_pilot',pilot=r)
            elif scene=='lit_circle_five':selected[scene]=dict(stage=None,status='GAP',reason='圆形唯一离散回退试跑仍有碰撞',pilot=r)
            else:
                gamma=.25 if scene=='lit_narrow_passage' else .15
                call('plan_resume.py',scene,'fallback',gamma);fallback.append(scene+'_fallback')
        if fallback:
            # The fixed reduced rates already passed the original head-on
            # grid; support-halfspace proof holds for all gamma in [0,1].
            call('resume_queue.py','pilot_fallback',*fallback);guards('before_pilot_fallback');call('dispatch_resume.py','pilot_fallback')
            wait_phase('pilot_fallback',fallback)
            for stage in fallback:
                call('resume_pilot_summary.py',stage);r=json.loads((HERE/f'{stage}_PILOT_RESULTS_RESUME.json').read_text());scene=stage.rsplit('_',1)[0]
                selected[scene]=dict(stage=stage if r['passed'] else None,status='accepted_pilot' if r['passed'] else 'GAP',reason=None if r['passed'] else '唯一gamma降档回退仍碰撞',pilot=r)
        write('PART1_STAGE_SELECTIONS.json',selected)
        call('merge_fix.py')
        accepted=[r['stage'] for r in selected.values() if r['stage']]
        if accepted:
            call('resume_queue.py','full',*accepted);guards('before_full');call('dispatch_resume.py','full');wait_phase('full',accepted);call('merge_fix.py')
        write('PART1_COMPUTATION_READY.json',dict(utc=datetime.now(timezone.utc).isoformat(),stage_selections=selected,
              remaining='Saved-trace diagnosis completeness, 5000 paired bootstrap, DB postflight, reports, final integrity, COMPLETE. Part two has not begun.'))
        progress('computation_ready',accepted)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        write('PART1_DRIVER_ERROR.json',dict(utc=datetime.now(timezone.utc).isoformat(),error=str(exc)))
        raise
