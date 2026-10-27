from experiment import *
from atomic_write import write
import ast,difflib
files=[Path(__file__)];changes=[]
def build(old,new,pairs):
 before=(HERE/old).read_text();after=before
 for a,b in pairs:assert a in after,(old,a);after=after.replace(a,b)
 ast.parse(after)
 target=HERE/new
 if target.exists():assert target.read_text()==after,new
 else:target.write_text(after)
 diff=HERE/(new+'.diff');diff.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=old,tofile=new)))
 files.extend([target,diff]);changes.append(dict(source=old,source_sha256=sha(HERE/old),target=new,target_sha256=sha(target)))
# All numerical/statistical helpers remain original. Only revision interfaces and class authority differ.
build('derived_metrics.py','derived_metrics_correction_revised_20261021.py',[
 ('from learn_critic import source_key','from learn_critic_revised_20261021 import source_key'),
 ("HERE/'critic_fits'","HERE/'critic_fits_revised_20261021'"),
 ('LEARNING_BUNDLE_{scene}','LEARNING_BUNDLE_REVISED_20261021_{scene}'),
 ('FROZEN_SCENE_CRITIC_{scene}','FROZEN_SCENE_CRITIC_REVISED_20261021_{scene}'),
 ('LEARNED_EVAL_CHOICES_{scene}','LEARNED_EVAL_CHOICES_REVISED_20261021_{scene}'),
 ('COMPACT_MAIN_RESULTS.json','COMPACT_REVISED_MAIN_RESULTS_20261021.json'),
 ("read(HERE/'CONTROLLER_LEVEL_FAILURES.json')","read(HERE/'INSTANCE_CLASSES_REVISED.json')['instances']"),
 ("r['controller_failure_definite']","r['remove_in_revised_exclusion']"),
 ('MAIN_ADDITIONAL_METRICS','MAIN_ADDITIONAL_METRICS_REVISED_20261021'),
 ('GEOMETRY_CENSUS.json','GEOMETRY_CENSUS_REVISED_20261021.json'),
 ('FAILURE_SENSITIVITY_SUMMARY.json','FAILURE_SENSITIVITY_SUMMARY_REVISED_20261021.json'),
 ("'derived_metrics.py'","'derived_metrics_correction_revised_20261021.py'"),
 ('twelve','fifteen')])
# Expected absent old-circle requests are exactly the sealed user-stop complement, never silently ignored.
block="""            user_stopped=False
            if stage=='learning_labels_lit_circle_five':
                verify_receipt('OLD_CIRCLE_LEARNING_STOPPED_20261021.receipt.json')
                preserved=read(HERE/'OLD_CIRCLE_LEARNING_LANDED_20261021.json')['rows']
                old_uids={r['rollout_uid'] for r in preserved}
                expected_absent={c['rollout_uid'] for c in cases}-old_uids
                assert set(absent)==expected_absent and len(absent)==4100,(stage,len(absent))
                assert complete==2044
                for r in preserved:assert sha(r['raw'])==r['raw_sha256'],r['raw']
                user_stopped=True
"""
build('postflight.py','postflight_correction_revised_20261021.py',[
 ("            main_stage=not stage.startswith('optional_')",block+"            main_stage=not stage.startswith('optional_') and not user_stopped"),
 ('main=main_stage))','main=main_stage,user_stopped=user_stopped,unwritten_not_unknown=user_stopped))'),
 ("HERE/'postflight.py'","HERE/'postflight_correction_revised_20261021.py'")])
build('learning_label_support.py','learning_label_support_correction_revised_20261021.py',[
 ('LEARNING_LABEL_SUPPORT','LEARNING_LABEL_SUPPORT_REVISED_20261021'),
 ('LEARNING_BUNDLE_{scene}','LEARNING_BUNDLE_REVISED_20261021_{scene}'),
 ('LEARNED_EVAL_CHOICES_{scene}','LEARNED_EVAL_CHOICES_REVISED_20261021_{scene}'),
 ('range(12)','range(15)'),
 ("'learning_label_support.py'","'learning_label_support_correction_revised_20261021.py'")])
write(HERE/'CORRECTION_REPORTING_SOURCE_DERIVATIONS_20261021.json',dict(utc=utc(),changes=changes,scientific_parameters_unchanged=True,old_circle_postflight_exemption='Only exact4100 user-stopped unwritten identities; verify sealed2044 preserved hashes and no other missing main request exemption'))
files.append(HERE/'CORRECTION_REPORTING_SOURCE_DERIVATIONS_20261021.json')
seal('CORRECTION_REPORTING_ADAPTERS_20261021.receipt.json',files,new_rollouts=0)
print('Reporting adapters sealed')
