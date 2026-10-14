"""IO-only ownership adapter; execute the exact frozen numerical runner."""
from run import *
index=int(sys.argv[1]);owner=100+index;prereg();labels_allowed();gate();p=read(HERE/'protocol.json');plan=read(HERE/'RESOURCE_BOOST_PLAN.json');receipt=read(HERE/'RESOURCE_BOOST.receipt.json');assert sha(HERE/'RESOURCE_BOOST_PLAN.json')==receipt['plan_sha256'];assert sha(__file__)==receipt['helper_sha256'];assert sha(HERE/'run.py')==p['io_binding_sha256'];assert sha(R9/'run.py')==p['runner_sha256']
rows=plan['cases'][index::6];owned={str((HERE/'main/raw'/f"{c['rollout_uid']}.json").resolve()) for c in rows};original_exists=Path.exists
# Reservations remain visible to original workers until the genuine raw is
# atomically written. Only the owning helper treats its marker as absent.
def owned_exists(path):
    found=original_exists(path)
    if found and str(path.resolve()) in owned:
        r=read(path)
        if r.get('schema')=='RESERVATION_NOT_EVIDENCE':
            assert r['reservation_owner']==owner and r['reservation_plan_sha256']==receipt['plan_sha256'];return False
    return found
Path.exists=owned_exists
write(HERE/'main'/f'START_AUTHORIZATION_{owner}.json',dict(utc=utc(),preregistration_sha256=sha(HERE/'PREREGISTRATION.md'),probe_receipt_sha256=sha(HERE/'R5_FROZEN.receipt.json'),selector_receipt_sha256=sha(HERE/'SELECTOR_FROZEN.receipt.json'),resource_receipt_sha256=sha(HERE/'RESOURCE_BOOST.receipt.json'),phase='main',worker=owner,before_any_worker_rollout=True))
frozen.execute(rows,'main',owner)
