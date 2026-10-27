from experiment import *
from unit_gates import GRID,head_on
trials=[]
for gamma in reversed(GRID):
    r=head_on('lit_circle_five',gamma,circle_discrete=True);trials.append(r)
    write(HERE/'CIRCLE_FALLBACK_HEAD_ON.json',dict(trials=trials,gamma=gamma if r['passed'] else None,passed=r['passed']))
    print(gamma,r['passed'],flush=True)
    if r['passed']:break
