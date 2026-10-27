"""One official merger; expose completed journals before dependent preflight."""
from experiment import *
import subprocess
import fcntl
import time

def main():
    with (HERE/'OFFICIAL_MERGER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        other=subprocess.run(['pgrep','-af','[s]hared_rollout_db.src.merger'],capture_output=True,text=True)
        assert other.returncode!=0,('Another official merger is active',other.stdout)
        result=subprocess.run([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment','exp_baseline2_literature_scenes_v1'],cwd=ROOT,capture_output=True,text=True)
        p=HERE/f'OFFICIAL_MERGE_{time.time_ns()}.json';write(p,dict(utc=utc(),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,command='python -B -m shared_rollout_db.src.merger --experiment exp_baseline2_literature_scenes_v1'))
        assert result.returncode==0,str(p)
        print(p.name,result.stdout[-500:],flush=True)

if __name__=='__main__':main()
