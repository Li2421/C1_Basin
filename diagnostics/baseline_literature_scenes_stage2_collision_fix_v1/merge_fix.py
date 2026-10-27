from experiment import *
from registry_fix import EXP
import subprocess
import fcntl
import time

def main():
    with (HERE/'OFFICIAL_MERGER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        other=subprocess.run(['pgrep','-af','[s]hared_rollout_db.src.merger'],capture_output=True,text=True)
        assert other.returncode!=0,('Another official merger is active',other.stdout)
        result=subprocess.run([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,capture_output=True,text=True)
        path=HERE/f'OFFICIAL_MERGE_{time.time_ns()}.json'
        write(path,dict(utc=utc(),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        assert result.returncode==0,str(path)
        print(result.stdout,flush=True)
if __name__=='__main__':main()
