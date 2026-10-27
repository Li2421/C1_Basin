import resume_bootstrap
from experiment import *
from collections import Counter

def summarize(stage):
    cases=read(HERE/f'{stage}_PILOT_CASES_RESUME.json');rows=[]
    for c in cases:
        p=HERE/stage/'raw'/f"{c['rollout_uid']}.json"
        if p.exists():rows.append(read(p))
    out=dict(utc=utc(),stage=stage,planned=len(cases),completed=len(rows),complete=len(rows)==len(cases),
        collisions=sum(r['collision'] for r in rows),unknown=sum(r['numerical_failure'] for r in rows),
        outcomes=dict(Counter(r['outcome'] for r in rows)),
        elapsed_cpu_seconds=sum(r['elapsed_seconds'] for r in rows),
        collision_examples=[dict(rollout_uid=r['rollout_uid'],controller=r['case']['controller_payload']['nominal']['controller'],state=r['case']['state']['index'],N=r['case']['state']['N'],method=r['case']['method'],clearance=r['minimum_swept_clearance'],steps=r['episode_length'],violations=r['constraint_violation_steps'],trace=r['trace_path']) for r in rows if r['collision']])
    out['passed']=out['complete'] and not out['collisions']
    write(HERE/f'{stage}_PILOT_RESULTS_RESUME.json',out)
    print({k:v for k,v in out.items() if k!='collision_examples'},flush=True)
    return out
if __name__=='__main__':summarize(sys.argv[1])
