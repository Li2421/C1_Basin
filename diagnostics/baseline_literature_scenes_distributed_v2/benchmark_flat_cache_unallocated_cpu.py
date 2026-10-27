"""Repeat timing off all Slurm CPUs; pause only the engineering replay briefly."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import signal
import time


def main():
    verify_receipt('CBS_FLAT_CACHE_CODE.receipt.json')
    from local_cpu_guard import allocated_cpus
    occupied,_=allocated_cpus();affinity=os.sched_getaffinity(0)
    assert len(affinity)==1 and not affinity&occupied
    pid=177744
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
    assert b'replay_flat_cache_saved.py' in cmd
    started=time.monotonic();os.kill(pid,signal.SIGSTOP)
    try:
        import test_cbs_flat_cache_saved as test
        def save_isolated(path,result):
            assert path.name=='CBS_FLAT_CACHE_SAVED_QUERY.json'
            write(HERE/'CBS_FLAT_CACHE_UNALLOCATED_CPU_QUERY.json',dict(result,physical_cpu=sorted(affinity),slurm_allocated_cpus=sorted(occupied),validation_replay_temporarily_paused=True,scope='Runtime timing without contention with any Slurm worker; no research worker paused'))
        test.write=save_isolated
        test.main()
    finally:
        os.kill(pid,signal.SIGCONT)
        write(HERE/'CBS_FLAT_CACHE_BENCHMARK_REPLAY_PAUSE.json',dict(utc=utc(),engineering_replay_pid=pid,pause_seconds=time.monotonic()-started,research_workers_paused=0,research_rollouts_cancelled=0))


if __name__=='__main__':main()
