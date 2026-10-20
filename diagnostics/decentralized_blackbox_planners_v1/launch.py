from common import *
import subprocess
G=guard()
seal('IMPLEMENTATION_FROZEN.receipt.json',[p for p in HERE.glob('*.py') if p.name not in ['analyze.py','delivery.py','plots.py']]+[HERE/'COPY_PROVENANCE.json'],before_probes=True,before_evaluation_labels=True,guard=guard())
with (HERE/'coordinator.log').open('w') as f:
 p=subprocess.Popen([sys.executable,'-B',str(HERE/'coordinator.py')],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','JAX_PLATFORMS':'cpu'},start_new_session=True)
write(HERE/'COORDINATOR_PID.json',dict(pid=p.pid,utc=utc()))
