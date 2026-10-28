import runner
from acceleration import enable
import sys
enable(check=sys.argv[1].startswith('accel_paired'))
from recording_fix import install
install()
if __name__=='__main__':
    from flexible_worker import worker_main
    worker_main(sys.argv[1],int(sys.argv[2]))
