"""Durable audit/finalization child with an actual OS exit-code record."""
from native_io import *
import subprocess
if __name__=='__main__':
    name=sys.argv[1];assert name in ['audit','finalize'];os.sched_setaffinity(0,{23})
    started=time.time();command=[sys.executable,'-u','-B',str(D/(name+'.py'))]
    p=subprocess.run(command,stdin=subprocess.DEVNULL)
    dump(D/(name.upper()+'_EXIT.json'),dict(exit_code=p.returncode,started=started,finished=time.time(),command=command))
    sys.exit(p.returncode)
