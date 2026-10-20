"""Synchronous isolated inference: no generator imports in the controller process."""
from common import *
import subprocess
class FrozenMean:
    def __init__(self):
        env=dict(os.environ);env.update(JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1');self.process=subprocess.Popen([sys.executable,'-B',str(HERE/'requery_helper.py')],cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        line=self.process.stdout.readline()
        if not line or not json.loads(line).get('ready'):raise RuntimeError('Frozen reference helper failed startup')
    def query(self,state_uid,physical):
        self.process.stdin.write(json.dumps(clean(dict(state_uid=state_uid,physical=physical)))+'\n');self.process.stdin.flush();line=self.process.stdout.readline()
        if not line:raise RuntimeError('Frozen reference helper exited')
        reply=json.loads(line)
        if not reply['ok']:raise RuntimeError(reply['error'])
        return reply['result']
    def close(self):
        if self.process.stdin:self.process.stdin.close()
        self.process.wait(timeout=30)
