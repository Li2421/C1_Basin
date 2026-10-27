from experiment import *
from collections import Counter

def summarize(stage):
    cases=read(HERE/f'{stage}_PILOT_CASES.json');rows=[];missing=[]
    for c in cases:
        p=HERE/stage/'raw'/f"{c['rollout_uid']}.json"
        if not p.exists():missing.append(c['rollout_uid']);continue
        r=read(p);rows.append(r)
    result=dict(utc=utc(),stage=stage,planned=len(cases),completed=len(rows),
        collisions=sum(r['collision'] for r in rows),unknown=sum(r['numerical_failure'] for r in rows),
        success=sum(r['success'] for r in rows),outcomes=dict(Counter(r['outcome'] for r in rows)),missing=missing,
        elapsed_cpu_seconds=sum(r['elapsed_seconds'] for r in rows),
        complete=len(rows)==len(cases),
        passed=len(rows)==len(cases) and not any(r['collision'] or r['numerical_failure'] for r in rows),
        collision_examples=[dict(rollout_uid=r['rollout_uid'],controller=r['case']['controller_payload']['nominal']['controller'],state=r['case']['state']['index'],method=r['case']['method'],clearance=r['minimum_swept_clearance'],steps=r['episode_length'],violations=r['constraint_violation_steps'],trace=r['trace_path']) for r in rows if r['collision']])
    write(HERE/f'{stage}_PILOT_RESULTS.json',result)
    print({k:v for k,v in result.items() if k not in ('collision_examples','missing')},flush=True)
    return result
if __name__=='__main__':summarize(sys.argv[1])
