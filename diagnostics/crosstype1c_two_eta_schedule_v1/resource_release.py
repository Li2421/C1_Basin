"""Independent persistent watchdog: cancel only this round's temporary jobs."""
from resource_guard import *
import time,fcntl
def main():
    with (HERE/'RESOURCE_RELEASE.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while datetime.now(timezone.utc).timestamp()<DEADLINE:time.sleep(max(0,min(30,DEADLINE-datetime.now(timezone.utc).timestamp())))
        cancelled=[]
        for p in HERE.glob('*_SUBMISSION.json'):
            r=json.loads(p.read_text())
            if not r.get('temporary'):continue
            job=r['job_id'].split(';')[0]
            current=subprocess.run(['squeue','-h','-j',job,'-o','%j'],text=True,capture_output=True)
            names=current.stdout.splitlines()
            if names:
                assert all(n.startswith('crosstype1c_') for n in names)
                result=subprocess.run(['scancel',job],capture_output=True,text=True)
                cancelled.append(dict(job=job,returncode=result.returncode,stderr=result.stderr))
        (HERE/'RESOURCE_RELEASE_COMPLETE.json').write_text(json.dumps(dict(utc=utc(),temporary_job_cancellations=cancelled,limits_after_release=limits(),queue=queue()),indent=2)+'\n')
if __name__=='__main__':main()
