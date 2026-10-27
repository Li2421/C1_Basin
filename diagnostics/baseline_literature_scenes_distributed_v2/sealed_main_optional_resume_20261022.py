"""Resume optional delivery directly after immutable main numbers exist."""
from experiment import *
from registry import verify_receipt
from atomic_write import write
from coordinator_spatial_mask_20261022 import action,status
import fcntl

READY='SEALED_MAIN_OPTIONAL_RESUME_READY_20261022.receipt.json'


def main():
    verify_receipt(READY);verify_receipt('MIGRATION_NUMBERS.receipt.json')
    with (HERE/'CORRECTION_CONTINUATION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(HERE/'CORRECTION_CONTINUATION_PID_20261021.json',dict(utc=utc(),pid=os.getpid(),sealed_main_resume=True))
        entry=read(HERE/'ACTIVE_ENTRYPOINT.json');target=entry['optional'];proof=verify_receipt(entry['optional_receipt'])
        assert Path(target).name==target and str(HERE/target) in proof['files']
        status('resuming_optional_after_sealed_main',main_rollouts_not_repeated=True)
        action('resume_optional_after_sealed_main',target)


if __name__=='__main__':main()
