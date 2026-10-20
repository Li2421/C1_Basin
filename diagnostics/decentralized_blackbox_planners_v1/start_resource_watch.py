from common import *
import subprocess
G=guard()
with (HERE/'resource_watch.log').open('w') as f:
 p=subprocess.Popen([sys.executable,'-B',str(HERE/'resource_watch.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','JAX_PLATFORMS':'cpu'})
write(HERE/'RESOURCE_WATCH_PID.json',dict(utc=utc(),pid=p.pid))
