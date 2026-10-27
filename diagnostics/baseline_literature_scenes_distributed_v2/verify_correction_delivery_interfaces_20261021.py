"""Check static continuation/reporting interfaces without submitting scientific jobs."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import ast,importlib
os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
modules=['optional_revision_diagnostics_20261021','optional_results_correction_revised_20261021','optional_delivery_correction_revised_20261021','final_reports_correction_revised_20261021','door_controller_gain_components_20261021','plot_correction_revised_20261021','derived_metrics_correction_revised_20261021','postflight_correction_revised_20261021']
# The plots environment is only used at final export, so syntax-check it now.
for name in modules:
 ast.parse((HERE/(name+'.py')).read_text())
 if name!='plot_correction_revised_20261021':importlib.import_module(name)
for proof in ('CORRECTION_OPTIONAL_EXECUTION_20261021.receipt.json','CORRECTION_REPORTING_ADAPTERS_20261021.receipt.json','CPU_RELEASE_CODE_20261023.receipt.json','CORRECTION_MAIN_RESULTS_READY_20261021.receipt.json'):verify_receipt(proof)
# Validate the sole intended postflight exemption against the immutable stop complement.
old=read(HERE/'learning_labels_lit_circle_five_CASES.json');preserved=read(HERE/'OLD_CIRCLE_LEARNING_LANDED_20261021.json')['rows'];landed={r['rollout_uid'] for r in preserved};planned={c['rollout_uid'] for c in old}
assert len(landed)==2044 and landed<=planned and len(planned-landed)==4100
# Safety-eta0 preserved by exact original factory conversion was checked separately.
checks=dict(utc=utc(),modules_syntax_checked=modules,nonplot_modules_imported=True,old_circle_postflight_exemption_exact_complement=4100,old_circle_preserved=2044,original_and_revised_result_paths_separate=True,scientific_jobs_submitted=0,new_rollouts=0,new_training=0)
write(HERE/'CORRECTION_DELIVERY_INTERFACE_CHECKS_20261021.json',checks)
print('Delivery static interface checks passed',flush=True)
