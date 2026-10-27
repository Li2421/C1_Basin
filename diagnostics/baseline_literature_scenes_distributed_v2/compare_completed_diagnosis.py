"""Check early completed-scope diagnosis figures against the final main figures."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import time
import fcntl


def main():
    with (HERE/'COMPLETED_DIAGNOSIS_COMPARISON.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_receipt('COMPLETED_DIAGNOSIS_COMPARISON_CODE.receipt.json')
        verify_receipt('COMPLETED_DIAGNOSIS_NUMBERS.receipt.json')
        write(HERE/'COMPLETED_DIAGNOSIS_COMPARISON_PID.json',dict(utc=utc(),pid=os.getpid()))
        while not (HERE/'MAIN_ADDITIONAL_METRICS.receipt.json').exists():time.sleep(30)
        verify_receipt('MAIN_ADDITIONAL_METRICS.receipt.json')
        verify_receipt('MIGRATION_NUMBERS.receipt.json')
        old=read(HERE/'COMPLETED_DIAGNOSIS_RESULTS.json')
        final=read(HERE/'MAIN_ADDITIONAL_METRICS.json')
        census=read(HERE/'CONTROLLER_LEVEL_FAILURES.json')
        assert old['controller_census']==census,'Controller census differs'
        checks=0
        for old_key,new_key in [('weighted_rates','weighted_outcome_and_diagnosis_rates'),('per_layer_excluding_definite_controller_failures','per_layer_excluding_definite_controller_failure_rates')]:
            for scene,arms in old[old_key].items():
                for arm,value in arms.items():
                    assert value==final[new_key][scene][arm],(old_key,scene,arm)
                    checks+=1
        write(HERE/'COMPLETED_DIAGNOSIS_COMPARISON.json',dict(utc=utc(),passed=True,controller_census_identical=True,shared_scope_weighted_summaries_identical=checks,new_rollouts=0))
        seal('COMPLETED_DIAGNOSIS_COMPARISON.receipt.json',[HERE/n for n in ('COMPLETED_DIAGNOSIS_COMPARISON.json','COMPLETED_DIAGNOSIS_NUMBERS.receipt.json','MAIN_ADDITIONAL_METRICS.receipt.json','MIGRATION_NUMBERS.receipt.json','COMPLETED_DIAGNOSIS_COMPARISON_CODE.receipt.json')])


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        write(HERE/'COMPLETED_DIAGNOSIS_COMPARISON_ERROR.json',dict(utc=utc(),error=repr(exc),traceback=traceback.format_exc()))
        raise
