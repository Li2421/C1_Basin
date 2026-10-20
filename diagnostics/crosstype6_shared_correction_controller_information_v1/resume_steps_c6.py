"""Resume numerical steps by sealed completion artifacts, never redo finished science."""
from common import *
import subprocess
def run(script,*args):subprocess.run([sys.executable,'-B',str(HERE/script),*args],check=True,cwd=ROOT)
def main(name):
    if name=='finish_training':
        for artifact,script,args in [('training_COMPLETE.receipt.json','finish_phase_c6.py',['training']),('TRAINING_BUNDLE.receipt.json','assemble_training_c6.py',[]),('MODEL_TESTS.receipt.json','test_models_c6.py',[]),('CRITIC_TRAINING_PLAN.receipt.json','train_critic.py',['plan'])]:
            if not (HERE/artifact).exists():run(script,*args)
    elif name=='freeze_choices':
        for artifact,script,args in [('FROZEN_INFERENCE_TESTS.receipt.json','inference.py',[]),('FROZEN_PIPELINE.receipt.json','evaluation_c6.py',['freeze']),('FULL_MENU_DEVELOPMENT_VALIDATION.receipt.json','fullmenu_development_c6.py',[]),('NEW_EVAL_CHOICES.receipt.json','evaluation_c6.py',['choose'])]:
            if not (HERE/artifact).exists():run(script,*args)
    elif name=='merge_safety':
        output=subprocess.check_output([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,text=True);write(HERE/'evaluation_SAFETY_MERGE.json',dict(utc=utc(),official_merger_output=output,physical_sources_only=True));seal('evaluation_SAFETY_MERGED.receipt.json',[HERE/'evaluation_SAFETY_MERGE.json',HERE/'evaluation_EXECUTION_FROZEN.receipt.json'])
    elif name=='numeric_T1_T7':
        for artifact,script,args in [('evaluation_COMPLETE.receipt.json','finish_phase_c6.py',['evaluation']),('NEW_EVAL_BUNDLE.receipt.json','evaluation_c6.py',['assemble']),('T1_T7_STATISTICS.receipt.json','statistics_c6.py',[])]:
            if not (HERE/artifact).exists():run(script,*args)
    else:raise ValueError(name)
if __name__=='__main__':main(sys.argv[1])
