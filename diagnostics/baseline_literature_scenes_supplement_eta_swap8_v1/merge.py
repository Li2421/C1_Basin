import bootstrap
from experiment import *
from registry import EXP
import subprocess,fcntl,time
if __name__=='__main__':
    with (HERE/'OFFICIAL_MERGER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        other=subprocess.run(['pgrep','-af','[s]hared_rollout_db.src.merger'],capture_output=True,text=True);assert other.returncode!=0,other.stdout
        p=subprocess.run([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,capture_output=True,text=True)
        write(HERE/f'OFFICIAL_MERGE_{time.time_ns()}.json',dict(utc=utc(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,p.stderr
        print(p.stdout,flush=True)
