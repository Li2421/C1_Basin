"""Static worker/claim changes for revised cases; no sealed source edits."""
from experiment import *
from atomic_write import write
import ast,difflib
os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
changes=[]
def add(old,new,replacements):
 source=(HERE/old).read_text();updated=source
 for a,b in replacements:assert a in updated,(old,a);updated=updated.replace(a,b)
 normalized=updated
 for a,b in reversed(replacements):normalized=normalized.replace(b,a)
 assert normalized==source,old
 ast.parse(updated) if new.endswith('.py') else None
 assert not (HERE/new).exists();(HERE/new).write_text(updated)
 diff=HERE/(new+'.diff');diff.write_text(''.join(difflib.unified_diff(source.splitlines(True),updated.splitlines(True),fromfile=old,tofile=new)))
 changes.append(dict(original=old,original_sha256=sha(HERE/old),new=new,new_sha256=sha(HERE/new),diff=diff.name,all_outside_explicit_replacements_byte_identical=True))
add('worker_flat_static.py','worker_correction_revised_20261021.py',[
 ("def main(stage):\n", "def main(stage):\n    assert stage.startswith(('distributed_revised_fixed_','central_revised_fixed_','learning_labels_revised_20261021_','learning_eval_revised_20261021_')),stage\n    verify_receipt('CORRECTION_REVISION.receipt.json');verify_receipt('PREREGISTRATION_ADDENDUM_20261021.receipt.json');verify_receipt('CORRECTION_FIXED_PIPELINE_IMPLEMENTATION_20261021.receipt.json')\n"),
 ('from runtime_flat_static import run as distributed_run','from runtime_correction_revised_20261021 import run as distributed_run'),
 ('from central_reservation_flat_static import run as historical_run','from central_correction_revised_20261021 import run as historical_run'),
 ("result['pure_geometry_cache_equivalence_receipt_sha256']=", "result['correction_revision_receipt_sha256']=sha(HERE/'CORRECTION_REVISION.receipt.json')\n                result['pure_geometry_cache_equivalence_receipt_sha256']="),
 ("actual='central_reservation_flat_static.py' if run is historical_run else 'runtime_flat_static.py'","actual='central_correction_revised_20261021.py' if run is historical_run else 'runtime_correction_revised_20261021.py'"),
 ("str(HERE/'worker_flat_static.py')","str(HERE/'worker_correction_revised_20261021.py')")])
add('worker_flat_static.sbatch','worker_correction_revised_20261021.sbatch',[
 ('--job-name=blv2_flow','--job-name=blv2_revision'),('logs/worker_%j.log','logs/revised_%j.log'),('worker_flat_static.py','worker_correction_revised_20261021.py')])
add('fine_dispatch_flat_static_allcpu.py','fine_dispatch_correction_20261021.py',[
 ('from resources_idle_cpu import','from resources_corrections_20261021 import'),
 ("return stage.startswith(('learning_labels_','learning_eval_'))","return stage.startswith(('distributed_revised_fixed_','central_revised_fixed_','learning_labels_revised_20261021_','learning_eval_revised_20261021_'))"),
 ("def submit(stage):\n","def submit(stage):\n    assert eligible(stage),stage\n    verify_receipt('CORRECTION_REVISION.receipt.json');verify_receipt('PREREGISTRATION_ADDENDUM_20261021.receipt.json');verify_receipt('CORRECTION_FIXED_PIPELINE_IMPLEMENTATION_20261021.receipt.json')\n"),
 ("HERE/'worker_flat_static.sbatch'","HERE/'worker_correction_revised_20261021.sbatch'")])
write(HERE/'CORRECTION_FIXED_PIPELINE_SOURCE_DIFF_20261021.json',dict(utc=utc(),changes=changes,science_change_only='Documented Brel contact gate; canonical method identity allows exact fixed/menu reuse',sealed_old_sources_preserved=True))
print('Prepared revised worker and fine dispatch; old circle stages rejected')
