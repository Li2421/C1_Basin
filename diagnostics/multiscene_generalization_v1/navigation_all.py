"""Run each family's gate once after its four checkpoints are immutable."""
from common import *
from capability import navigation,navigation_result,checkpoints,pilot,pilot_result
from submit_stage import submit,completed,merge
import time

def main():
    for family in 'ABCDEFG':
        while not all((p.parent/'COMPLETE.receipt.json').exists() for p in checkpoints(family)):time.sleep(5)
        phase='navigation_'+family
        if not (HERE/f'{phase}_CASES.json').exists():navigation(family)
        if not (HERE/f'{phase}_SUBMISSION.json').exists():
            while submit(phase,18) is None:time.sleep(5)
        job=read(HERE/f'{phase}_SUBMISSION.json')['job']
        while not completed(job):time.sleep(3)
        if not (HERE/f'{phase}_POSTFLIGHT.json').exists():merge(phase)
        result=navigation_result(family)
        write(HERE/'ACTIVE_STAGE.json',dict(utc=utc(),phase=phase,passed=sum(v['passed'] for v in result)))
        print(family,[(v['name'],v['success'],v['unknown']) for v in result],flush=True)
    write(HERE/'ALL_NAVIGATION_GATES.json',dict(families={f:read(HERE/f'NAVIGATION_CAPABILITY_{f}.json') for f in 'ABCDEFG'}))
    seal('ALL_NAVIGATION_GATES.receipt.json',[HERE/'ALL_NAVIGATION_GATES.json'],no_replacements=True,no_retraining_after_gate=True)
    for family in 'ABCDEFG':
        nav=read(HERE/f'NAVIGATION_CAPABILITY_{family}.json')
        if not nav['family_passed']:
            write(HERE/f'MULTIAGENT_CAPABILITY_{family}.json',dict(family=family,status='NOT_RUN',reason='Fewer than two instances passed navigation capability gate',eligible=[],family_passed=False,instances=[]));continue
        rows=pilot(family)
        if not rows:continue
        phase='pilot_gate_'+family
        job=None
        while job is None:job=submit(phase,18);time.sleep(1)
        while not completed(job):time.sleep(5)
        merge(phase);eligible=pilot_result(family)
        print(family,'multi-agent eligible',eligible,flush=True)
    write(HERE/'ALL_CAPABILITY_GATES.json',dict(families={f:read(HERE/f'MULTIAGENT_CAPABILITY_{f}.json') for f in 'ABCDEFG'}))
    seal('ALL_CAPABILITY_GATES.receipt.json',[HERE/'ALL_CAPABILITY_GATES.json'],before_target_labels=True)

if __name__=='__main__':main()
