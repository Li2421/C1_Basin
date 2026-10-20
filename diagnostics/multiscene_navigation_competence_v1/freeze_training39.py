from common import *
from dagger39 import round_specs

def main():
 assert read(HERE/'TRAINING39_UNIT_TESTS.json')['passed'];assert read(HERE/'NAVIGATION_UNIT_TESTS.json')['passed']
 config=dict(observation_dimensions=39,hidden=[256,256,256],kinds=['deterministic','mixture'],mixture_components=5,seeds=[17,23],rounds=[0,1,2,3],final_round=3,steps_per_round=60000,batch=256,adam_learning_rate=.0003,development_samples=2048,development_rng='202610173900+policy_seed',development_check_interval=250,checkpoint_selection='earliest minimum fixed development loss',initialization='fresh from same policy seed each round',training_geometry_only=True,development_fixed_after_round0=True,label_selection='all finite pre-terminal expert-labeled states including collision-causing step; no post-terminal state; failed unknown collections remain in provenance and contribute no unverified labels',normalization='recomputed from cumulative training data only each round; constant columns unit scale',tasks={f:round_specs(f) for f in 'ABCDEFG'},training_family_navigation_gate_requirement=2,initial_expert_gate=dict(success=.99,zero_collisions=True),no_navigation_holdout_choice=True,no_retraining_after_final_gate=True)
 write(HERE/'TRAINING39_CONFIG_FROZEN.json',config)
 seal('TRAINING_PIPELINE39.receipt.json',[HERE/n for n in ['TRAINING39_CONFIG_FROZEN.json','training39.py','policies39.py','dagger39.py','collect39.py','test_training39.py','TRAINING39_UNIT_TESTS.json','observation39.py','navigation_field.py','test_navigation_field.py','NAVIGATION_UNIT_TESTS.json']],before_demonstration_collection=True,before_any_training=True)
 write(HERE/'SOURCE_BOUNDARY_RECOVERY_STATUS.json',dict(utc=utc(),old_failure_retained=True,status='RECOVERED_WITH_USER_AUTHORIZATION',authorization_prompt=str(ROOT/'docs/prompt_round40_multiscene_navigation_competence_20261017.md'),baseline=str(HERE/'INTEGRITY_RESUME_BEGIN.json'),no_forbidden_content_used=True,no_forbidden_write=True))
 print('39-dimensional training pipeline frozen')
if __name__=='__main__':main()
