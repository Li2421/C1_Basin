"""Transferred reservation IO only; unchanged frozen numerical execution."""
from run import *
wave=int(sys.argv[1]);index=int(sys.argv[2]);plan=read(HERE/f'RESOURCE_WAVE{wave}_PLAN.json');receipt=read(HERE/f'RESOURCE_WAVE{wave}.receipt.json');owner=wave*1000+index;prereg();labels_allowed();gate();p=read(HERE/'protocol.json');assert sha(HERE/f'RESOURCE_WAVE{wave}_PLAN.json')==receipt['plan_sha256'];assert sha(__file__)==receipt['helper_sha256'];assert sha(HERE/'run.py')==p['io_binding_sha256'];assert sha(R9/'run.py')==p['runner_sha256']
rows=plan['cases'][index::plan['helpers']];owned={str((HERE/'main/raw'/f"{c['rollout_uid']}.json").resolve()) for c in rows};original_exists=Path.exists
schema=plan['reservation_schema']
def owned_exists(path):
    found=original_exists(path)
    if found and str(path.resolve()) in owned:
        r=read(path)
        if r.get('schema')==schema:
            assert r['reservation_owner']==owner and r['reservation_plan_sha256']==receipt['plan_sha256'];return False
    return found
Path.exists=owned_exists
write(HERE/'main'/f'START_AUTHORIZATION_{owner}.json',dict(utc=utc(),preregistration_sha256=sha(HERE/'PREREGISTRATION.md'),probe_receipt_sha256=sha(HERE/'R5_FROZEN.receipt.json'),selector_receipt_sha256=sha(HERE/'SELECTOR_FROZEN.receipt.json'),resource_receipt_sha256=sha(HERE/f'RESOURCE_WAVE{wave}.receipt.json'),phase='main',worker=owner,before_any_worker_rollout=True))
frozen.execute(rows,'main',owner)
