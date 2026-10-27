from registry_fix import verify_receipt
from experiment import prereg,sys
verify_receipt('IMPLEMENTATION_FROZEN.receipt.json')
prereg()
from worker_fix import main
main(*sys.argv[1:])
