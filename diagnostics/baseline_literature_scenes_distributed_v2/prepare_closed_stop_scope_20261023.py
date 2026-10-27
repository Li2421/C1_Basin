from experiment import *
from atomic_write import write
import ast,difflib
files=[Path(__file__),HERE/'closed_old_circle_scope_20261023.py'];changes=[]
def output(old,new,after):
 before=(HERE/old).read_text();ast.parse(after);target=HERE/new;assert not target.exists();target.write_text(after);diff=HERE/(new+'.diff');diff.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=old,tofile=new)));files.extend([target,diff]);changes.append(dict(source=old,source_sha256=sha(HERE/old),target=new,target_sha256=sha(target)))
old='statistics_parallel_correction_20261023.py';s=(HERE/old).read_text();s=s.replace('controller_census,complete_original_landed_label_diagnoses,save_or_assert','controller_census,save_or_assert')
s=s.replace('from statistics_parallel_tasks_20261023 import task_manifest','from statistics_parallel_tasks_20261023 import task_manifest\nfrom closed_old_circle_scope_20261023 import complete_original_landed_label_diagnoses')
s=s.replace("'INSTANCE_CLASSES_REVISED.receipt.json',CODE_PROOF)","'INSTANCE_CLASSES_REVISED.receipt.json',CODE_PROOF,'CORRECTION_SCOPED_NUMERICS_READY_20261023.receipt.json')")
s=s.replace("'statistics_parallel_correction_20261023.py'","'statistics_parallel_scoped_20261023.py','CORRECTION_SCOPED_NUMERICS_READY_20261023.receipt.json'")
a="fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_receipt(CODE_PROOF)";assert a in s;s=s.replace(a,a+";verify_receipt('CORRECTION_SCOPED_NUMERICS_READY_20261023.receipt.json')")
output(old,'statistics_parallel_scoped_20261023.py',s)
# Postflight respects the immutable old raw scope even if exact eta0 identities now exist in the new stage/cache.
old='postflight_correction_revised_20261021.py';s=(HERE/old).read_text().replace('from result_inventory import find_raw','from result_inventory import find_raw\nfrom closed_old_circle_scope_20261023 import preserved_scope,OLD_STAGE')
a="            cases=read(HERE/f'{stage}_CASES.json');complete=0;absent=[]";b=a+"\n            closed=preserved_scope() if stage==OLD_STAGE else None";assert a in s;s=s.replace(a,b)
a="                rid=c['rollout_uid']\n                try:path=find_raw(rid,stage,c['state']['scene'])";b="""                rid=c['rollout_uid']
                if closed is not None and rid not in closed:
                    absent.append(rid);continue
                try:path=Path(closed[rid]['raw']) if closed is not None else find_raw(rid,stage,c['state']['scene'])""";assert a in s;s=s.replace(a,b)
a="            stages.append(dict(stage=stage,planned=len(cases),completed=complete,missing=len(absent),main=main_stage,user_stopped=user_stopped,unwritten_not_unknown=user_stopped))"
b="""            later_exact_eta0_cache_hits=0
            if closed is not None:
                for c in cases:
                    if c['rollout_uid'] in closed:continue
                    row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
                    if row is not None:
                        assert c.get('menu_index')==0 and not any(c['eta_a']) and not any(c['eta_b']),'Unexpected old nonzero execution after stop'
                        later_exact_eta0_cache_hits+=1
            stages.append(dict(stage=stage,planned=len(cases),completed=complete,missing=len(absent),main=main_stage,user_stopped=user_stopped,unwritten_not_unknown=user_stopped,later_exact_eta0_cache_hits_not_old_stage_completion=later_exact_eta0_cache_hits,closed_original_raw_scope=closed is not None))"""
assert a in s;s=s.replace(a,b).replace("HERE/'postflight_correction_revised_20261021.py'","HERE/'postflight_correction_closed_scope_20261023.py'")
output(old,'postflight_correction_closed_scope_20261023.py',s)
# Model and numerical calls are unchanged; new receipts cover only the closed-scope adapter.
write(HERE/'CLOSED_STOP_SCOPE_SOURCE_DERIVATIONS_20261023.json',dict(utc=utc(),changes=changes,scope_rule='Old circle is exactly its immutable2044 raw manifest. New revised eta0 retains same execution UID by requirement; later global cache availability does not expand old stopped labels or count as old resumption.',new_rollouts=0,math_unchanged=True))
files.append(HERE/'CLOSED_STOP_SCOPE_SOURCE_DERIVATIONS_20261023.json')
from closed_old_circle_scope_20261023 import preserved_scope
closed=preserved_scope();planned=read(HERE/'learning_labels_lit_circle_five_CASES.json');all_uids={c['rollout_uid'] for c in planned};assert len(all_uids)==6144 and len(all_uids-set(closed))==4100
later_zero={c['rollout_uid'] for c in planned if c.get('menu_index')==0}-set(closed)
# Simulate every old identity becoming globally discoverable; the explicit stop predicate still selects only2044.
selected=[c for c in planned if c['rollout_uid'] in closed];assert len(selected)==2044 and all(c['rollout_uid'] not in later_zero for c in selected)
write(HERE/'CLOSED_STOP_SCOPE_EQUIVALENCE_20261023.json',dict(utc=utc(),passed=True,old_planned=6144,old_preserved=2044,old_stopped=4100,potential_later_authorized_eta0_identities=len(later_zero),global_cache_filled_simulation_does_not_expand_old_scope=True,new_rollouts=0))
files += [HERE/'CLOSED_STOP_SCOPE_EQUIVALENCE_20261023.json',HERE/'CORRECTION_PARALLEL_NUMERICS_READY_20261023.receipt.json',HERE/'OLD_CIRCLE_LEARNING_STOPPED_20261021.receipt.json']
seal('CORRECTION_SCOPED_NUMERICS_READY_20261023.receipt.json',files,same_native_math_and27tasks=True,old_stopped_scope_exact=True,authorized_new_eta0_semantic_reuse_preserved=True,no_old_nonzero_resumption=True)
print('Closed old scope adapter sealed; later new eta0 potential IDs',len(later_zero))
