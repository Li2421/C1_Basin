"""Operational source snapshot; scientific worker receipts stay immutable."""
from experiment import *
import ast
import signal
import subprocess

def main():
    text=(HERE/'infrastructure_recovery_v3.py').read_text()
    lines=text.splitlines()
    lines=['    own_trace="Traceback (most recent call last)" in tail and str(HERE) in tail'
           if line.lstrip().startswith('own_trace=') else line for line in lines]
    text='\n'.join(lines)+'\n'
    text=text.replace("and 'worker_reservation_certified.sbatch' in", "and 'worker_reservation' in")
    text=text.replace('min(32,old*2)','min(96,old*2)')
    text=text.replace("'node failure'","'node failure','std::bad_alloc','cannot allocate memory'")
    (HERE/'infrastructure_recovery_current.py').write_text(text)
    coor=(HERE/'coordinator_v4.py').read_text().replace('from infrastructure_recovery_v3 import','from infrastructure_recovery_current import').replace('downstream_v4.py','downstream_current.py')
    (HERE/'coordinator_current.py').write_text(coor)
    down=(HERE/'downstream_v4.py').read_text().replace('from coordinator_v4 import','from coordinator_current import').replace('from critic_supervisor import','from critic_supervisor_current import')
    (HERE/'downstream_current.py').write_text(down)
    critic=(HERE/'critic_supervisor.py').read_text().replace('from infrastructure_recovery_v3 import','from infrastructure_recovery_current import')
    (HERE/'critic_supervisor_current.py').write_text(critic)
    names=('infrastructure_recovery_current.py','coordinator_current.py','downstream_current.py','critic_supervisor_current.py')
    for name in names:ast.parse((HERE/name).read_text(),filename=name)
    write(HERE/'ORCHESTRATOR_CURRENT_DEPLOYMENT.json',dict(utc=utc(),operational_source_at_start={n:sha(HERE/n) for n in names},science_frozen_receipts=['IMPLEMENTATION_FROZEN.receipt.json','RESERVATION_V2_EQUIVALENCE.receipt.json','RESERVATION_V2_WORKER_DEPLOYMENT.receipt.json'],reason='Generic worker-name memory recovery and scontrol fallback; physical/controller parameters unchanged'))
    owner=read(HERE/'COORDINATOR_PID.json')['pid']
    os.kill(owner,signal.SIGTERM)
    with (HERE/'coordinator.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-B',str(HERE/'coordinator_current.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('Current coordinator',child.pid,flush=True)

if __name__=='__main__':main()
