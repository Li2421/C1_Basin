"""Restart-safe root for a completed, sealed main experiment."""
from experiment import *
from registry import verify_receipt
from atomic_write import write
from coordinator_spatial_mask_20261022 import action
import fcntl


def main():
    ready='SEALED_MAIN_OPTIONAL_RESUME_READY_20261022.receipt.json'
    verify_receipt(ready);verify_receipt('MIGRATION_NUMBERS.receipt.json')
    with (HERE/'CORRECTION_COORDINATOR.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        entry=read(HERE/'ACTIVE_ENTRYPOINT.json');target='sealed_main_optional_resume_20261022.py'
        entry.update(utc=utc(),pid=os.getpid(),coordinator=Path(__file__).name,downstream=target,downstream_receipt=ready)
        write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
        write(HERE/'COORDINATOR_PID.json',dict(utc=utc(),pid=os.getpid(),entrypoint=Path(__file__).name,sealed_main_resume=True))
        # The ordinary optional driver still verifies all original science
        # receipts and resumes only missing preflighted requests.
        action('sealed_main_optional_resume',target)


if __name__=='__main__':main()
