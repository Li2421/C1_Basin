"""Static15-action learning variants; model/optimizer/folds/seeds stay byte-identical."""
from experiment import *
from atomic_write import write
import ast,difflib
os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
changes=[]
def add(old,new,replacements,custom=None):
 original=(HERE/old).read_text();text=original
 for a,b in replacements:
  assert a in text,(old,a);text=text.replace(a,b)
 if custom:text=custom(text)
 ast.parse(text) if new.endswith('.py') else None
 
 if (HERE/new).exists():assert (HERE/new).read_text()==text,new
 else:(HERE/new).write_text(text)
 diff=HERE/(new+'.diff');diff.write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=old,tofile=new)))
 changes.append(dict(original=old,original_sha256=sha(HERE/old),new=new,new_sha256=sha(HERE/new),diff=diff.name))
path_replacements=[('LEARNING_BUNDLE_','LEARNING_BUNDLE_REVISED_20261021_'),('CRITIC_TRAINING_PLAN_','CRITIC_TRAINING_PLAN_REVISED_20261021_'),('FROZEN_SCENE_CRITIC_','FROZEN_SCENE_CRITIC_REVISED_20261021_'),("'critic_fits'","'critic_fits_revised_20261021'"),('learn_critic.py','learn_critic_revised_20261021.py')]
add('learn_critic.py','learn_critic_revised_20261021.py',path_replacements)
# Only storage paths change inside the frozen numerical training implementation.
text=(HERE/'learn_critic_revised_20261021.py').read_text()
for a,b in reversed(path_replacements):text=text.replace(b,a)
assert text==(HERE/'learn_critic.py').read_text()
add('learning_design.py','learning_design_revised_20261021.py',[
 ('from study_cases import make','from revised_case_factory_20261021 import make'),
 ("method='safety' if index==0 else f'learning_action_{index:02d}',seed=seed,eta_a=a['eta_a'],eta_b=a['eta_b'],theta=a['theta']", "method='safety' if index==0 else f'learning_action_{index:02d}',seed=seed,eta_a=a['eta_a'],eta_b=a['eta_b'],theta=a['theta'],menu_index=index"),
 ('LEARNING_GAP_','LEARNING_GAP_REVISED_20261021_'),('LEARNING_MENU.json','LEARNING_MENU_REVISED.json'),('LEARNING_SPLIT_','LEARNING_SPLIT_REVISED_20261021_'),('LEARNING_BUNDLE_','LEARNING_BUNDLE_REVISED_20261021_'),('learning_design.py','learning_design_revised_20261021.py'),
 ("'learning_labels_'","'learning_labels_revised_20261021_'"),("f'learning_labels_{scene}_CASES.json'","f'learning_labels_revised_20261021_{scene}_CASES.json'"),('assert len(cases)==6144','assert len(cases)==7680'),('32,12),-1','32,15),-1'),('actions.shape==(12,19)','actions.shape==(15,19)')])
add('freeze_scene_choices.py','freeze_scene_choices_revised_20261021.py',[
 ('from learning_design import','from learning_design_revised_20261021 import'),('from learn_critic import','from learn_critic_revised_20261021 import'),('FROZEN_SCENE_CRITIC_','FROZEN_SCENE_CRITIC_REVISED_20261021_'),('LEARNING_SPLIT_','LEARNING_SPLIT_REVISED_20261021_'),('LEARNING_MENU.json','LEARNING_MENU_REVISED.json'),('LEARNED_EVAL_CHOICES_','LEARNED_EVAL_CHOICES_REVISED_20261021_'),('freeze_scene_choices.py','freeze_scene_choices_revised_20261021.py'),("'learning_eval_'","'learning_eval_revised_20261021_'")])
add('critic_jobs_released.py','critic_jobs_revised_20261021.py',[
 ('from resources_released import','from resources_corrections_20261021 import'),('LEARNING_BUNDLE_','LEARNING_BUNDLE_REVISED_20261021_'),('critic.sbatch','critic_revised_20261021.sbatch'),("stage='critic_'+scene","stage='critic_revised_20261021_'+scene")])
add('critic.sbatch','critic_revised_20261021.sbatch',[
 ('--job-name=blv2_critic','--job-name=blv2_revcritic'),('learn_critic.py','learn_critic_revised_20261021.py')])
# Independent scenes within a requested learning phase progress as before;
# this scheduler is called only after all required prior fixed stages finish.
add('scene_learning_scheduler_flat_static.py','scene_learning_scheduler_revised_20261021.py',[
 ('from coordinator_pipelined import','from coordinator_correction_recovery_20261021 import'),('from resources_released import','from resources_corrections_20261021 import'),('from infrastructure_recovery_active import','from infrastructure_recovery_correction_20261021 import'),('import fine_dispatch_flat_static as fine_dispatch','import fine_dispatch_correction_recovery_20261021 as fine_dispatch'),('from critic_jobs_released import','from critic_jobs_revised_20261021 import'),('from status_released import main as status','from coordinator_correction_recovery_20261021 import status'),
 ("'critic_'","'critic_revised_20261021_'"),('LEARNING_GAP_','LEARNING_GAP_REVISED_20261021_'),('FROZEN_SCENE_CRITIC_','FROZEN_SCENE_CRITIC_REVISED_20261021_'),('LEARNING_BUNDLE_','LEARNING_BUNDLE_REVISED_20261021_'),("'learning_labels_'","'learning_labels_revised_20261021_'"),("'learning_eval_'","'learning_eval_revised_20261021_'"),('learning_design.py','learning_design_revised_20261021.py'),('freeze_scene_choices.py','freeze_scene_choices_revised_20261021.py'),
 ("originals=[f'distributed_baseline_{s}' for s in NAMES]\n            originals += [f'{layer}_fixed_{s}' for layer in ('distributed','central') for s in NAMES]", "originals=[f'distributed_revised_fixed_{s}' for s in NAMES]\n            originals += [f'central_revised_fixed_{s}' for s in NAMES if stage_done(f'central_revised_fixed_{s}')]")])
write(HERE/'REVISED_LEARNING_SOURCE_CHANGES_20261021.json',dict(utc=utc(),changes=changes,training_numerical_source_after_path_normalization_byte_identical=True,architecture_and_hyperparameters_unchanged=True,action_count=15,action_encoding_dimension=19,seeds=[17,23,41],steps=2000,source_fits_per_scene=36,new_rollouts=0))
print('Prepared15-action learner variants; training arithmetic unchanged')
