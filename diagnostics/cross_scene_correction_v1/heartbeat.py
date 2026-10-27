"""Read-only worker stack observation; never counts as a completed result."""
import json
import os
from pathlib import Path
import sys
import threading
import time
from datetime import datetime, timezone


def start():
    owner=threading.get_ident()
    root=Path(__file__).resolve().parent
    job=os.environ.get('SLURM_ARRAY_JOB_ID',os.environ.get('SLURM_JOB_ID','local'))
    task=os.environ.get('SLURM_ARRAY_TASK_ID','0')
    path=root/'worker_status'/f'{job}_{task}_{os.getpid()}.json'
    path.parent.mkdir(parents=True,exist_ok=True)

    def observe():
        while True:
            try:
                frame=sys._current_frames().get(owner)
                stack=[];request=None;step=None
                while frame is not None:
                    filename=frame.f_code.co_filename
                    stack.append(dict(file=filename,line=frame.f_lineno,function=frame.f_code.co_name))
                    if Path(filename)in(root/'runtime_new.py',root/'runtime_checkpointed.py')and frame.f_code.co_name=='run':
                        values=frame.f_locals
                        candidate=values.get('c')
                        if isinstance(candidate,dict):request=candidate.get('rollout_uid')
                        value=values.get('t')
                        if isinstance(value,int):step=value
                    frame=frame.f_back
                memo=sys.modules.get('pa_query_memo_candidate_20261022')
                cache=None if memo is None else dict(counters=dict(memo.COUNTERS),
                    entries=len(memo.CACHE),estimated_payload_bytes=memo.BYTES)
                payload=dict(utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
                    rollout_uid=request,simulation_step=step,stack=stack,
                    process_cpu_seconds=time.process_time(),astar_memo=cache,
                    diagnostic_only=True,counts_as_completed_result=False)
                tmp=path.with_suffix('.json.tmp')
                tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
                tmp.replace(path)
            except Exception:
                # Monitoring failure must never alter a scientific execution.
                pass
            finally:
                frame=None
            time.sleep(20)

    threading.Thread(target=observe,name='read-only-stack-observer',daemon=True).start()
