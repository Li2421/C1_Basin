"""Supplemental queue consumers; the frozen rollout runner remains byte unchanged."""
from core import *
import importlib.util
spec=importlib.util.spec_from_file_location('round15_frozen_rollout_runner',HERE/'run.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
original_queue=runner.queued_cases
def bounded_queue(rows,phase):
    stream=original_queue(rows,phase)
    try:
        for c in stream:
            if datetime.now().astimezone().hour*60+datetime.now().astimezone().minute>=505:return
            yield c
    finally:stream.close()
runner.queued_cases=bounded_queue
def main():
    phase=sys.argv[1];index=int(sys.argv[2]);assert phase.startswith('wave') and 18<=index<24
    if datetime.now().astimezone().hour*60+datetime.now().astimezone().minute>=505:
        write(HERE/phase/f'BOOST_worker_{index}.json',dict(utc=utc(),started_after_0825=True,new=0,clean_exit=True));return
    rows=read(HERE/f'{phase}_MISSING_CASES.json')
    try:runner.execute(rows,phase,index)
    except RuntimeError as exc:
        if 'Wave incomplete before 10:00 cutoff' in str(exc) and datetime.now().astimezone().hour*60+datetime.now().astimezone().minute>=505:
            write(HERE/phase/f'BOOST_worker_{index}.json',dict(utc=utc(),stopped_claiming_at_0825=True,clean_exit=True,unfinished_cases_remain_in_shared_queue=True));return
        raise
if __name__=='__main__':main()
