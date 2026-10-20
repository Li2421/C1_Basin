from common import *
import time,subprocess
while not (HERE/'EXPERT_EXECUTION_COMPLETE.json').exists():time.sleep(5)
from expert_results import analyze
analyze()
r=read(HERE/'EXPERT_GATE_RESULTS.json')
if all(not x['passed'] for x in r['families'].values()):
 subprocess.run([sys.executable,'-B',str(HERE/'audit_resume.py'),'RESUME_END'],check=True)
 from deliver_failed_expert_pilot import main
 main()
else:
 write(HERE/'NEXT_TRAINING_STAGE.json',dict(utc=utc(),families=[f for f,x in r['families'].items() if x['passed']],status='expert passed; run frozen round0-3 pipeline before navigation gate'))
