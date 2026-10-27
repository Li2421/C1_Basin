"""Finish required analyses and first COMPLETE; never starts supplement work."""
from pathlib import Path
import subprocess,sys,json,time,os,fcntl
HERE=Path(__file__).resolve().parent
PY=sys.executable
def call(script,*args):
    p=HERE/'finish_actions'/f'{time.time_ns()}_{script}.log';p.parent.mkdir(exist_ok=True)
    with p.open('w') as log:r=subprocess.run([PY,'-B',str(HERE/script),*map(str,args)],cwd=HERE.parents[1],stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError((script,r.returncode,str(p),p.read_text()[-2000:]))
def jobs():
    return subprocess.check_output(['squeue','-u',os.environ.get('USER','zhihan'),'-h','-r','-o','%j'],text=True).splitlines()
def main():
    with (HERE/'PART1_FINISH_DRIVER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (HERE/'PART1_FINISH_DRIVER_PID.json').write_text(json.dumps(dict(pid=os.getpid())))
        while not (HERE/'PART1_COMPUTATION_READY.json').exists():
            if (HERE/'PART1_DRIVER_ERROR.json').exists() or (HERE/'RESUME_FATAL_STOP.json').exists():raise RuntimeError('Computation driver requires review')
            time.sleep(30)
        call('backfill_part1_diagnostics.py')
        call('resume_audit.py','before_statistics');call('other_readonly_guard.py','before_statistics')
        before=subprocess.check_output(['squeue','-h','-r','-o','%i|%u|%T|%C'],text=True)
        assert not any(n.startswith('lit_resume_') for n in jobs()),before
        job=subprocess.check_output(['sbatch','--parsable','--array=0-2',str(HERE/'part1_statistics.sbatch')],text=True).strip()
        (HERE/'STATISTICS_SUBMISSION_RESUME.json').write_text(json.dumps(dict(job=job,all_user_jobs_before=before)))
        while any(n=='lit_resume_stats' for n in jobs()):time.sleep(30)
        for scene,r in json.loads((HERE/'PART1_STAGE_SELECTIONS.json').read_text()).items():
            if r['stage']:assert (HERE/f'STATISTICS_{scene}.json').exists(),scene
        call('resume_audit.py','after_statistics');call('other_readonly_guard.py','after_statistics')
        call('postflight_resume.py');call('finalize_part1.py')
if __name__=='__main__':
    try:main()
    except Exception as exc:
        (HERE/'PART1_FINISH_DRIVER_ERROR.json').write_text(json.dumps(dict(error=str(exc)),ensure_ascii=False,indent=2)+'\n')
        raise
