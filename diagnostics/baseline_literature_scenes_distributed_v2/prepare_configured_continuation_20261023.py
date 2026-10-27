"""Select sealed continuation entrypoints from operational metadata; no runtime patch."""
from experiment import *
from atomic_write import write
import ast,difflib
files=[Path(__file__)];changes=[]
def output(old,new,after):
 before=(HERE/old).read_text();ast.parse(after);target=HERE/new;assert not target.exists();target.write_text(after);diff=HERE/(new+'.diff');diff.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=old,tofile=new)));files.extend([target,diff]);changes.append(dict(source=old,source_sha256=sha(HERE/old),target=new,target_sha256=sha(target)))
old='coordinator_correction_tail_backfill_20261023.py';s=(HERE/old).read_text();start=s.index("  while not (HERE/'CORRECTION_CONTINUATION_TAIL_BACKFILL_READY_20261023.receipt.json')")
stop=s.index("if __name__=='__main__':main()",start)
newtail='''  while True:
   entry=read(HERE/'ACTIVE_ENTRYPOINT.json');proof=entry.get('downstream_receipt');target=entry['downstream']
   assert Path(target).name==target and target.endswith('.py'),target
   if proof and (HERE/proof).exists():
    assert Path(proof).name==proof and proof.endswith('.receipt.json'),proof
    ready=verify_receipt(proof);assert str(HERE/target) in ready['files'],target
    break
   status('revised_Dfixed_complete_awaiting_continuation_implementation',required_next='D learning3 -> Cfixed4 -> circle learning last',required_receipt=proof);time.sleep(10)
  action('correction_after_Dfixed',target)
'''
s=s[:start]+newtail+s[stop:];s=s.replace('CORRECTION_CONTINUATION_TAIL_BACKFILL_READY_20261023.receipt.json','CORRECTION_CONFIGURED_CONTINUATION_READY_20261023.receipt.json').replace('coordinator_correction_tail_backfill_20261023.py','coordinator_correction_configured_20261023.py')
output(old,'coordinator_correction_configured_20261023.py',s)
old='correction_after_distributed_fixed_tail_backfill_20261023.py';s=(HERE/old).read_text();pairs=[('from coordinator_correction_tail_backfill_20261023 import','from coordinator_correction_configured_20261023 import'),('CORRECTION_CONTINUATION_TAIL_BACKFILL_READY_20261023.receipt.json','CORRECTION_CONFIGURED_CONTINUATION_READY_20261023.receipt.json'),('CORRECTION_MAIN_RESULTS_READY_20261021.receipt.json','CORRECTION_PARALLEL_NUMERICS_READY_20261023.receipt.json'),('statistics_correction_revised_20261021.py','statistics_parallel_correction_20261023.py')]
for a,b in pairs:assert a in s,a;s=s.replace(a,b)
output(old,'correction_after_distributed_fixed_parallel_20261023.py',s)
old='scheduler_watchdog_correction_tail_backfill_20261023.py';s=(HERE/old).read_text();pairs=[('coordinator_correction_tail_backfill_20261023.py','coordinator_correction_configured_20261023.py'),("str(HERE/'correction_after_distributed_fixed_tail_backfill_20261023.py')","str(HERE/entry['downstream'])"),('CORRECTION_CONTINUATION_TAIL_BACKFILL_READY_20261023.receipt.json','CORRECTION_CONFIGURED_CONTINUATION_READY_20261023.receipt.json'),('scheduler_watchdog_correction_tail_backfill_20261023.py','scheduler_watchdog_correction_configured_20261023.py')]
for a,b in pairs:assert a in s,a;s=s.replace(a,b)
output(old,'scheduler_watchdog_correction_configured_20261023.py',s)
write(HERE/'CONFIGURED_CONTINUATION_SOURCE_DERIVATIONS_20261023.json',dict(utc=utc(),changes=changes,scientific_worker_and_request_identity_unchanged=True,strict_phase_order_unchanged=True,statistical_change='Independent unchanged native calls in27 tasks; diagnostic reader avoids unused census JSON parse, retaining original full-file hashes',entrypoint_selection='Only presealed .py entries underHERE whose hashes appear in the declared ready receipt; normal process dispatch, no runtime module patch'))
files += [HERE/'CONFIGURED_CONTINUATION_SOURCE_DERIVATIONS_20261023.json',HERE/'CORRECTION_PARALLEL_NUMERICS_READY_20261023.receipt.json',HERE/'CORRECTION_CONTINUATION_TAIL_BACKFILL_READY_20261023.receipt.json']
seal('CORRECTION_CONFIGURED_CONTINUATION_READY_20261023.receipt.json',files,numerical_equivalence_verified=True,science_unchanged=True,worker_processes_not_interrupted=True,scheduler_owner_guard_preserved=True)
print('Configured corrected continuation sealed; native numerical parallelism ready')
