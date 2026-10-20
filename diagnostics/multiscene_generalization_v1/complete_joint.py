"""Parent-only adaptive validity design, official merger, then demonstration stage."""
from common import *
from submit_stage import completed,merge,submit
from catalog import accept,retry,finalize
import time

def main():
    phase='joint_3';number=3
    while True:
        sub=read(HERE/f'{phase}_SUBMISSION.json')
        while not completed(sub['job']):time.sleep(3)
        merge(phase);failed=accept(phase)
        write(HERE/'ACTIVE_STAGE.json',dict(utc=utc(),phase=phase,unresolved=len(failed)))
        print(phase,'unresolved',len(failed),flush=True)
        if not failed:break
        number+=1;rows=retry(number,failed)
        if not rows:break
        phase='joint_'+str(number)
        while submit(phase,18) is None:time.sleep(5)
    finalize()
    from demonstrations import prepare,freeze
    prepare()
    job=None
    while job is None:job=submit('demonstrations',18);time.sleep(1)
    while not completed(job):time.sleep(5)
    merge('demonstrations');freeze()
    job=None
    while job is None:job=submit('policy_training',2,True,'train_policies.py');time.sleep(1)
    write(HERE/'ACTIVE_STAGE.json',dict(utc=utc(),phase='policy_training',job=job))
    print('Policy training submitted',job,flush=True)

if __name__=='__main__':main()
