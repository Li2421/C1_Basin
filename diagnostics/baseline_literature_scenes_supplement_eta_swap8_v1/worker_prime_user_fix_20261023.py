"""Bounded first capability shard; cooperative stop after current saved episode."""
import os,signal,sys
from worker_user_fix_20261023 import main
signal.signal(signal.SIGALRM,lambda *_:os.kill(os.getpid(),signal.SIGUSR1))
signal.alarm(300)
main(sys.argv[1])
