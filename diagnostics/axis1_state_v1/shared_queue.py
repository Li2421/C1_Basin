"""Only assigns preregistered, unexecuted cases; no simulator changes."""
from common import *
import fcntl
class SharedCases:
 def __init__(self):self.rows=read(HERE/'SPEEDUP_CASES.json')
 def __len__(self):return len(self.rows)
 def __iter__(self):
  while True:
   with (HERE/'SPEEDUP_CURSOR').open('r+') as f:
    fcntl.flock(f,fcntl.LOCK_EX);i=int(f.read());
    if i>=len(self.rows):return
    f.seek(0);f.write(str(i+1));f.truncate();f.flush();os.fsync(f.fileno())
   yield self.rows[i]
def reserved_or_missing(path):
 if not path.exists():return True
 r=read(path)
 return r.get('schema')=='round32_work_reservation_v1'
