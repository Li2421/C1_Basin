from pathlib import Path
if (Path(__file__).resolve().parent/'READONLY_BOUNDARY_STOP.json').exists():
    raise SystemExit('STOPPED: protected v2 cache write; unsafe native cache must not be imported')
from registry_fix import verify_receipt
from experiment import prereg,sys
verify_receipt('IMPLEMENTATION_FROZEN.receipt.json')
verify_receipt('WORKER_PATH_CORRECTION.receipt.json')
prereg()
from worker_corrected import main
main(*sys.argv[1:])
