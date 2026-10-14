"""Bind round9's immutable numerical rollout loop to round10 artifact IO."""
from core import *
import importlib.util
from shared_rollout_db.src import cache_writer
_append=cache_writer.append_journal
def normalized_append(records,*args,**kwargs):return _append(clean(records),*args,**kwargs)
cache_writer.append_journal=normalized_append
spec=importlib.util.spec_from_file_location('round9_frozen_runner',R9/'run.py')
frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
if __name__=='__main__':
    phase=sys.argv[1];index=int(sys.argv[2]);workers=int(sys.argv[3]);prereg();p=read(HERE/'protocol.json');assert sha(__file__)==p['io_binding_sha256']
    if phase=='gate':assert read(HERE/'STATE_DEDUP_AUDIT_V2.json')['passed']
    rows=read(HERE/f'{phase}_MISSING_CASES.json');frozen.execute(rows[index::workers],phase,index)
