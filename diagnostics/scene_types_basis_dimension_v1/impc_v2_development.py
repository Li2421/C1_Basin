"""Validate and screen the corrected occupancy adapter on development only."""
from dispatch import *
from papers import prepare,make,OLD
import time

def main():
    while not(HERE/'paper_development_other15_execution.json').exists():time.sleep(10)
    assert read(HERE/'paper_development_other15_execution.json')['complete']
    prepare();entries=[e for e in read(HERE/'SCENES_FROZEN.json')['entries']if 'scene'in e and e['type_id']!=2]
    variants=read(OLD/'paper/DEVELOPMENT_FROZEN.json')['variants']['chen_impc_ob_si_tracking'];pilot=[]
    for e in entries:
        if e['type_id']not in(1,8,14,16):continue
        for opts in [variants[0],variants[-1]]:pilot.append(make(e['scene'],0,'chen_impc_ob_si_tracking',opts,'paper_development'))
    rr=execute(pilot,'impc_v2_smoke',worker_entry='worker_papers.py')
    interface=[r for r in rr if (r.get('error')or{}).get('type')in('IndexError','ImportError','ModuleNotFoundError','AttributeError','TypeError')]
    if interface:raise RuntimeError('Corrected IMPC adapter still has interface errors; preserve pilot and do not scale.')
    cases=[make(e['scene'],i,'chen_impc_ob_si_tracking',opts,'paper_development')for e in entries for i in range(4)for opts in variants]
    execute(cases,'impc_v2_development_other15',worker_entry='worker_papers.py')

if __name__=='__main__':main()
