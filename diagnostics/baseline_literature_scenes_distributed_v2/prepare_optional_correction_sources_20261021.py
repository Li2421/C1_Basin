"""Static optional adapters; original files stay immutable and eta0 identities stay exact."""
from experiment import *
from atomic_write import write
import ast,difflib
os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
changes=[];files=[Path(__file__)]
def save(old,new,changeset):
 original=(HERE/old).read_text();updated=original
 for a,b in changeset:
  assert a in updated,(old,a);updated=updated.replace(a,b)
 ast.parse(updated) if new.endswith('.py') else None
 target=HERE/new
 if target.exists():assert target.read_text()==updated,new
 else:target.write_text(updated)
 diff=HERE/(new+'.diff');diff.write_text(''.join(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),fromfile=old,tofile=new)))
 changes.append(dict(source=old,source_sha256=sha(HERE/old),target=new,target_sha256=sha(target),documented_replacements=changeset));files.extend([target,diff])
save('runtime_local_progress_flat_static.py','runtime_local_progress_correction_revised_20261021.py',[
 ('from central_reference import progress','from central_reference import progress\nfrom correction_revision_20261021 import distributed_fields,central_fields,contact_radius'),
 ('basis.compute(env.positions,env.goals,v,cfg.max_speed).correction(eta)','central_fields(env.positions,env.goals,v,cfg.max_speed,cfg.goal_tolerance,contact_radius(cfg),basis).correction(eta)'),
 ('local_basis(env.positions,env.goals,v,i,np.inf,cfg.max_speed)','distributed_fields(env.positions,env.goals,v,i,cfg.max_speed,cfg.goal_tolerance,contact_radius(cfg))')])
# Eta0 and nominal cases call the exact original conversion; only revised nonzero gets new execution identities.
conversion='''    zero=not any(base['eta_a']) and not any(base['eta_b'])
    if zero:return original_converted(base,engine,extra,scenario,version)
    extra=dict(extra,correction_semantics='contact_conflict_relative_gate_20261021',correction_revision_receipt_sha256=sha(HERE/'CORRECTION_REVISION.receipt.json'),arrived_flag_broadcast=True,revision_factory_sha256=sha(HERE/'optional_cases_correction_revised_20261021.py'))
'''
save('optional_cases.py','optional_cases_correction_revised_20261021.py',[
 ('from registry import case,register,verify_receipt','from registry import case,register,verify_receipt\nfrom optional_cases import converted as original_converted'),
 ('def converted(base,engine,extra,scenario=None,version=None):\n','def converted(base,engine,extra,scenario=None,version=None):\n'+conversion),
 ("prereg();verify_receipt('MIGRATION_NUMBERS.receipt.json')","prereg();verify_receipt('MIGRATION_NUMBERS.receipt.json');verify_receipt('CORRECTION_OPTIONAL_EXECUTION_20261021.receipt.json');verify_receipt('CORRECTION_REVISION.receipt.json')"),
 ("sha(HERE/'runtime_local_progress.py')","sha(HERE/'runtime_local_progress_correction_revised_20261021.py')")])
save('optional_worker_flat_static.py','optional_worker_correction_revised_20261021.py',[
 ("def main(stage):\n","def main(stage):\n    verify_receipt('CORRECTION_OPTIONAL_EXECUTION_20261021.receipt.json');verify_receipt('CORRECTION_REVISION.receipt.json')\n"),
 ('from dynamics_runtime_flat_static import run','from dynamics_correction_revised_20261021 import run'),
 ('from runtime_local_progress_flat_static import run','from runtime_local_progress_correction_revised_20261021 import run'),
 ('from runtime_flat_static import run','from runtime_correction_revised_20261021 import run'),
 ("result['jax_enable_x64']=bool(jax.config.x64_enabled)","result['jax_enable_x64']=bool(jax.config.x64_enabled)\n                result['correction_revision_receipt_sha256']=sha(HERE/'CORRECTION_REVISION.receipt.json')\n                result['optional_revision_execution_receipt_sha256']=sha(HERE/'CORRECTION_OPTIONAL_EXECUTION_20261021.receipt.json')"),
 ("'dynamics_runtime_flat_static.py' if engine=='paper_dynamics' else 'runtime_local_progress_flat_static.py' if engine=='local_progress' else 'runtime_flat_static.py'","'dynamics_correction_revised_20261021.py' if engine=='paper_dynamics' else 'runtime_local_progress_correction_revised_20261021.py' if engine=='local_progress' else 'runtime_correction_revised_20261021.py'"),
 ("str(HERE/'optional_worker_flat_static.py')","str(HERE/'optional_worker_correction_revised_20261021.py')")])
save('optional_flat_static.sbatch','optional_correction_revised_20261021.sbatch',[
 ('optional_worker_flat_static.py','optional_worker_correction_revised_20261021.py')])
save('optional_fine_dispatch_flat_static_allcpu.py','optional_fine_dispatch_correction_revised_20261021.py',[
 ('from infrastructure_recovery_active import classify','from infrastructure_recovery_correction_20261021 import classify'),
 ('from resources_idle_cpu import','from resources_corrections_release_20261023 import'),
 ("def submit(stage):\n","def submit(stage):\n    verify_receipt('CORRECTION_OPTIONAL_EXECUTION_20261021.receipt.json')\n"),
 ("str(HERE/'optional_flat_static.sbatch')","str(HERE/'optional_correction_revised_20261021.sbatch')")])
save('optional_diagnosis_backfill_allcpu.py','optional_diagnosis_backfill_correction_revised_20261021.py',[
 ('from resources_idle_cpu import','from resources_corrections_release_20261023 import'),
 ('from coordinator_flat_static_allcpu import','from coordinator_correction_release_20261023 import'),
 ('from infrastructure_recovery_active import','from infrastructure_recovery_correction_20261021 import')])
write(HERE/'OPTIONAL_CORRECTION_SOURCE_DIFF_20261021.json',dict(utc=utc(),changes=changes,science_changes='Only contact gate on original Brel; eta0 and nominal identities retain original conversion; original tracking/projection/physics/progress unchanged',historical_optional_evidence='No optional CASES or PREFLIGHT exists; all preflight/register deferred until sealed combined main numbers',research_rollouts=0));files.append(HERE/'OPTIONAL_CORRECTION_SOURCE_DIFF_20261021.json')
# Cover the new dynamics execution file after its unchanged safety/tracking and revised core gates passed.
files += [HERE/'dynamics_correction_revised_20261021.py',HERE/'CORRECTION_REVISION.receipt.json',HERE/'CORRECTION_REVISION_SOURCE_DIFF_20261021.json',HERE/'CPU_RELEASE_CODE_20261023.receipt.json',HERE/'DYNAMICS_IMPLEMENTATION.receipt.json']
seal('CORRECTION_OPTIONAL_EXECUTION_20261021.receipt.json',files,after_revision_unit_gates=True,second_stage_nonmain=True,exact_original_eta0_identities=True,local_progress_basis_only_change=True)
print('Sealed optional corrected numerical sources; no research run or registration',flush=True)
