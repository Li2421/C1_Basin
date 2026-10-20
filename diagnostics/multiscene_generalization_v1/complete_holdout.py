from common import *
from submit_stage import completed,merge,submit
from holdout_correction import accept,retry,finalize
from capability import navigation,navigation_result,pilot,pilot_result
import time

def run_stage(phase):
    job=None
    while job is None:job=submit(phase,18);time.sleep(1)
    while not completed(job):time.sleep(3)
    merge(phase)

def main():
    number=0
    while True:
        phase='holdout_joint_'+str(number);run_stage(phase);failed=accept(phase)
        print(phase,'unresolved',len(failed),flush=True)
        if not failed:break
        number+=1;retry(number,failed)
    finalize()
    for family in 'ABCDEFG':
        navigation(family,True);phase='navigation_valid_'+family;run_stage(phase)
        result=navigation_result(family,True)
        print(family,'valid navigation',[(v['name'],v['success'],v['unknown']) for v in result],flush=True)
    write(HERE/'ALL_VALID_NAVIGATION_GATES.json',dict(families={f:read(HERE/f'NAVIGATION_VALID_CAPABILITY_{f}.json') for f in 'ABCDEFG'}))
    seal('ALL_VALID_NAVIGATION_GATES.receipt.json',[HERE/'ALL_VALID_NAVIGATION_GATES.json',HERE/'CATALOG_FINAL.json'],same_frozen_policies=True,no_retraining=True,no_replacement=True)
    for family in 'ABCDEFG':
        nav=read(HERE/f'NAVIGATION_VALID_CAPABILITY_{family}.json')
        if not nav['family_passed']:
            write(HERE/f'MULTIAGENT_VALID_CAPABILITY_{family}.json',dict(family=family,status='NOT_RUN',reason='Fewer than two instances passed valid navigation capability gate',eligible=[],family_passed=False,instances=[]));continue
        rows=pilot(family)
        if not rows:continue
        phase='pilot_gate_'+family;run_stage(phase);eligible=pilot_result(family)
        write(HERE/f'MULTIAGENT_VALID_CAPABILITY_{family}.json',read(HERE/f'MULTIAGENT_CAPABILITY_{family}.json'))
        print(family,'valid multi-agent eligible',eligible,flush=True)
    write(HERE/'ALL_CAPABILITY_FINAL.json',dict(families={f:read(HERE/f'MULTIAGENT_VALID_CAPABILITY_{f}.json') for f in 'ABCDEFG'}))
    seal('ALL_CAPABILITY_FINAL.receipt.json',[HERE/'ALL_CAPABILITY_FINAL.json'],before_target_correction_labels=True)

if __name__=='__main__':main()
