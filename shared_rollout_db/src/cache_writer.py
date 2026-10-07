from __future__ import annotations
import json, os, socket, time, uuid
from pathlib import Path
from .rollout_db import ROOT,canonical

def append_journal(records,experiment_uid,worker_id=None):
    """Workers append locally; only merger writes SQLite."""
    d=ROOT/'journals'/experiment_uid; d.mkdir(parents=True,exist_ok=True)
    worker_id=worker_id or f'{socket.gethostname()}_{os.getpid()}'
    p=d/f'{worker_id}_{int(time.time()*1e6)}_{uuid.uuid4().hex}.jsonl.tmp'
    with p.open('x') as f:
        for r in records:f.write(canonical(r)+'\n')
        f.flush();os.fsync(f.fileno())
    final=p.with_suffix('');p.rename(final);return final

