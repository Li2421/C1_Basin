"""Witness-only temporary goals; never used as a comparison controller."""
import runner,runtime,sys
from witnesses import run
original_run=runtime.run
runtime.run=lambda c:run(c,original_run) if c['controller_payload'].get('witness') else original_run(c)
if __name__=='__main__':runner.worker_main(sys.argv[1],int(sys.argv[2]))
