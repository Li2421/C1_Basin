"""Administrative resource guard; never cancels a job outside brd1_."""
from resource_watch import enforce
from pathlib import Path
import time,json
HERE=Path(__file__).resolve().parent
while True:
 try:
  data=enforce()
  if(HERE/'COMPLETE.json').exists()and not any(j['name'].startswith('brd1_')for j in data['observed_jobs']):break
 except Exception as exc:
  with(HERE/'resource_events.jsonl').open('a')as f:f.write(json.dumps(dict(time=time.time(),error=repr(exc)))+'\n')
 time.sleep(5)
