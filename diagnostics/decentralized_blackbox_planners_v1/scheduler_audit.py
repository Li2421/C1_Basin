from common import *
import subprocess
G=guard()
cmd=['sacct','-j','22825,22826,22827,22845,22863,22883','--format=JobID,JobName,State,Elapsed,Start,End,AllocCPUS','-P'];q=subprocess.run(cmd,capture_output=True,text=True)
write(HERE/'SCHEDULER_AUDIT.json',dict(utc=utc(),available=q.returncode==0,returncode=q.returncode,stdout=q.stdout,stderr=q.stderr,command=cmd,evidence_if_unavailable='Scheduler accounting is disabled; rely on sealed submission squeue snapshots, worker guards, periodic queue monitor, frozen pre-rollout amendment record and immutable trajectory timestamps.',initial_replay_job='22825 canceled while pending; replacement22826 completed. No scientific result was rerun due to this scheduler correction.',guard=guard()))
