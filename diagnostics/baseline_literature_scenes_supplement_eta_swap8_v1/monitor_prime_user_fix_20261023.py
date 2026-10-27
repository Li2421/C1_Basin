from pathlib import Path
import os,sys,time,json,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
if __name__=='__main__':
    jid=sys.argv[1]
    while any(x==jid or x.startswith(jid+'_') for x in subprocess.check_output(['squeue','-u','zhihan','-h','-r','-o','%i'],text=True).splitlines()):time.sleep(20)
    with (HERE/'USER_FIX_PRIME_AFTER_GUARD.log').open('w') as log:r=subprocess.run([sys.executable,'-B',str(HERE/'guard_fast.py'),'after_user_fix_prime'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:
        (HERE/'FATAL_STOP.json').write_text(json.dumps(dict(reason='Prime post-batch read-only guard failed',job=jid)))
        for line in subprocess.check_output(['squeue','-u','zhihan','-h','-r','-o','%i|%j'],text=True).splitlines():
            job,name=line.split('|')
            if name.startswith('lit_supp_'):subprocess.run(['scancel',job],check=True)
        raise RuntimeError('Read-only guard failed')
    count=sum(1 for _ in (HERE/'capability_user_fix_20261023/raw').glob('*.json'))
    (HERE/'USER_FIX_PRIME_DONE.json').write_text(json.dumps(dict(utc=time.time(),job=jid,local_rows=count,passed_readonly_postcheck=True,reason='Five-minute authorized initial shard; partial claims recovered by regular batch, unknown raw never rerun')))
