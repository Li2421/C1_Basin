"""IO-only case ownership; call the same frozen execute function."""
from core import *
import run as bound
index=int(sys.argv[1]);owner=100+index;prereg();gate();plan=read(HERE/'RESOURCE_BOOST_PLAN.json');receipt=read(HERE/'RESOURCE_BOOST.receipt.json');assert sha(HERE/'RESOURCE_BOOST_PLAN.json')==receipt['plan_sha256'];assert sha(__file__)==receipt['helper_sha256'];assert sha(HERE/'run.py')==read(HERE/'protocol.json')['run_sha256'];rows=plan['cases'][index::6];owned={str((HERE/'main/raw'/f"{c['rollout_uid']}.json").resolve()) for c in rows};original_exists=Path.exists;original_read=bound.read;original_write=bound.write;original_append=bound.append_journal
# Original workers see markers as existing. Only the owning helper treats its
# own non-evidence marker as absent; no physical call or seed rule is changed.
def exists(path):
    found=original_exists(path)
    if found and str(path.resolve()) in owned:
        r=original_read(path)
        if r.get('schema')=='RESERVATION_NOT_EVIDENCE':assert r['reservation_owner']==owner;return False
    return found
Path.exists=exists
def subset_read(path):return rows if Path(path).resolve()==HERE/'main_MISSING_CASES.json' else original_read(path)
def owner_write(path,obj):
    path=Path(path)
    if path==HERE/'main/START_AUTHORIZATION_0.json':path=HERE/'main'/f'START_AUTHORIZATION_{owner}.json';obj=dict(obj,worker=owner,resource_receipt_sha256=sha(HERE/'RESOURCE_BOOST.receipt.json'),helper_adapter_sha256=sha(__file__))
    elif path==HERE/'main/worker_0.json':path=HERE/'main'/f'worker_{owner}.json'
    return original_write(path,obj)
def owner_append(records,experiment,tag):return original_append(records,experiment,tag.replace('main_0',f'main_{owner}'))
bound.read=subset_read;bound.write=owner_write;bound.append_journal=owner_append
bound.execute('main',0,1)
