"""Use the paused interval for controls and port smoke tests, no heldout data."""
from dispatch import *
from acceleration import accelerated_case
import time

def main():
    while not(HERE/'SCENE2_COORDINATOR_PAUSED.json').exists():time.sleep(10)
    entries=[e for e in read(HERE/'SCENES_FROZEN.json')['entries']if 'scene'in e and e['type_id']!=2]
    cases=[accelerated_case(e['scene'],c,i,dict(eta=[.875,0.,.375],coefficients={}),purpose='matched_midpoint_control')for e in entries for c in controllers()for i in range(4)]
    rows=execute(cases,'matched_midpoint')
    dump(HERE/'MATCHED_MIDPOINT_RESULT.json',dict(cases=len(rows),success=sum(r['success']for r in rows),description='Additional conservative parameter-control point G0.875/P0/R0.375, no new field. Needed to separate new-direction contributions from the changed original-three-column settings. No heldout outcome used. All work recorded and counted against the hard budget.'))
    from papers import prepare,make,OLD
    prepare();options=read(OLD/'paper/FINAL_FROZEN.json')['selected'];smoke=[]
    for e in entries:
        if e['type_id']not in(1,8,14,16):continue
        for method,opts in options.items():smoke.append(make(e['scene'],0,method,opts,'paper_smoke'))
    execute(smoke,'paper_smoke',worker_entry='worker_papers.py')

if __name__=='__main__':main()
