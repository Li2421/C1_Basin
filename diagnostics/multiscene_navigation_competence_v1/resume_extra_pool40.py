"""Keep the single merger coordinator waiting until every supplemental worker exits."""
from pathlib import Path
import subprocess,time,json,os,signal,datetime
H=Path(__file__).resolve().parent
state=json.loads((H/'CURRENT_EXTRA_POOL40.json').read_text())
for job in [state['original_job'],state['additional_job']]:
    while subprocess.check_output(['squeue','-h','-j',job],text=True).strip():time.sleep(5)
pid=state['primary_coordinator_pid'];command=Path(f'/proc/{pid}/cmdline').read_bytes()
assert b'pilot_training40.py' in command
os.kill(pid,signal.SIGCONT)
(H/'EXTRA_POOL40_PRIMARY_RESUMED.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs=[state['original_job'],state['additional_job']],pid=pid,status='all physical workers complete; sole primary merger coordinator resumed'),indent=2)+'\n')
