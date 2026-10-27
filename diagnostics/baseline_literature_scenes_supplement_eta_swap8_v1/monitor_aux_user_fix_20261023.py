from pathlib import Path
import subprocess,sys,time,json,os
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
if __name__=='__main__':
    jid=sys.argv[1]
    while any(x==jid or x.startswith(jid+'_') for x in subprocess.check_output(['squeue','-u','zhihan','-h','-r','-o','%i'],text=True).splitlines()):
        if (HERE/'FATAL_STOP.json').exists():subprocess.run(['scancel',jid],check=True)
        time.sleep(5)
    from local_cpu_guard import pin
    pin()
    with (HERE/'USER_FIX_AUX_MONITOR_AFTER_GUARD.log').open('w') as log:r=subprocess.run([sys.executable,'-B',str(HERE/'guard_fast.py'),'after_user_fix_aux_monitor'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:
        (HERE/'FATAL_STOP.json').write_text(json.dumps(dict(reason='Auxiliary read-only postcheck failed',job=jid)))
        for line in subprocess.check_output(['squeue','-u','zhihan','-h','-r','-o','%i|%j'],text=True).splitlines():
            job,name=line.split('|')
            if name.startswith('lit_supp_'):subprocess.run(['scancel',job],check=True)
        raise RuntimeError('Read-only postcheck failed')
    (HERE/'USER_FIX_AUX_EXITED.json').write_text(json.dumps(dict(utc=time.time(),job=jid,read_only_postcheck_passed=True)))
