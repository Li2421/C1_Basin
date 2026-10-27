"""Compare serialized fixed-menu results; normalize tuple/list representation."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import time
import signal
import fcntl


def main():
    with (HERE/'FIXED_CROSSFIT_SAVED_COMPARISON.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_receipt('FIXED_CROSSFIT_SAVED_COMPARISON_CODE.receipt.json')
        write(HERE/'FIXED_CROSSFIT_SAVED_COMPARISON_PID.json',dict(utc=utc(),pid=os.getpid()))
        while not (HERE/'COMPLETED_FIXED_CROSSFIT_NUMBERS.receipt.json').exists():time.sleep(10)
        verify_receipt('COMPLETED_FIXED_CROSSFIT_NUMBERS.receipt.json')
        # The numerical producer has finished all eight tables at this point.
        # Its in-memory future comparison retains tuples; final JSON has lists.
        # Retain the sealed figures and compare serialized representations.
        pid=read(HERE/'COMPLETED_FIXED_CROSSFIT_PID.json')['pid']
        cmd=Path(f'/proc/{pid}/cmdline')
        if cmd.exists():
            assert b'completed_fixed_crossfit.py' in cmd.read_bytes()
            os.kill(pid,signal.SIGTERM)
        write(HERE/'FIXED_CROSSFIT_COMPARISON_HANDOFF.json',dict(utc=utc(),finished_numerical_producer_pid=pid,numerical_outputs_retained=True,recomputed_statistics=False,reason='Compare saved JSON to saved JSON, avoiding in-memory tuple/list representation mismatch'))
        while not (HERE/'MIGRATION_NUMBERS.receipt.json').exists():time.sleep(30)
        verify_receipt('MIGRATION_NUMBERS.receipt.json')
        early=read(HERE/'COMPLETED_FIXED_CROSSFIT.json')['results']
        final=read(HERE/'MIGRATION_RESULTS.json')['results']
        for scene,by_layer in early.items():
            for layer,result in by_layer.items():assert canonical(result)==canonical(final[scene]['crossfit_fixed_oracle'][layer]),(scene,layer)
        write(HERE/'COMPLETED_FIXED_CROSSFIT_COMPARISON.json',dict(utc=utc(),passed=True,identical_scopes=8,new_rollouts=0))
        seal('COMPLETED_FIXED_CROSSFIT_COMPARISON.receipt.json',[HERE/n for n in ('COMPLETED_FIXED_CROSSFIT_COMPARISON.json','COMPLETED_FIXED_CROSSFIT_NUMBERS.receipt.json','MIGRATION_NUMBERS.receipt.json','FIXED_CROSSFIT_SAVED_COMPARISON_CODE.receipt.json')],passed=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        write(HERE/'FIXED_CROSSFIT_SAVED_COMPARISON_ERROR.json',dict(utc=utc(),error=repr(exc),traceback=traceback.format_exc()))
        raise
