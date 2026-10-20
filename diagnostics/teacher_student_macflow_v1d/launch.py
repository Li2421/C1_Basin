from core import *
import subprocess
from audit_utils import source_inventory
assert read(HERE/'COMPONENT_TESTS.json')['status']=='PASS'
files=[p for p in HERE.glob('*.py') if p.name not in ['analysis.py','finish.py','launch.py']]+[HERE/'rollout.sbatch',HERE/'COMPONENT_TESTS.json',HERE/'PREREGISTRATION.receipt.json']
seal('IMPLEMENTATION_FROZEN.receipt.json',files,before_new_rollouts=True)
py='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
for script,log,pid in [('orchestrate.py','orchestration.log','COORDINATOR_PID'),('quota_watch.py','quota_watch.log','QUOTA_WATCH_PID')]:
    if script=='quota_watch.py':
        import time
        while not (HERE/'RUN_STATUS.json').exists():time.sleep(.2)
    with (HERE/log).open('w') as f:
        p=subprocess.Popen([py,'-B',str(HERE/script)],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','JAX_PLATFORMS':'cpu'},start_new_session=True)
    (HERE/pid).write_text(str(p.pid)+'\n')
source_inventory('imports_launch.json')
