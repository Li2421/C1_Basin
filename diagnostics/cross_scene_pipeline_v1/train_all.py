"""Train/select generators first, then critics; no held-out scoring or rollouts."""
from common import *
from queueing import training_tasks
from schedule import run,jobs,cancel

def select_models(stage):
    groups={}
    for task in training_tasks(stage):
        p=HERE/'folds'/task['fold']/task['mode']/f"seed{task['seed']}"/'FIT.receipt.json';r=read(p)
        assert sha(r['checkpoint'])==r['sha256'];groups.setdefault((task['fold'],task['mode']),[]).append(r)
    for(fold,mode),rows in groups.items():
        chosen=min(rows,key=lambda r:(r['validation_loss'],r['seed']))
        write(HERE/'folds'/fold/mode/'SELECTED.json',dict(results=rows,selected=chosen,selection='Smallest source-only internal validation loss; ties smaller seed.'))

def main():
    assert read(HERE/'MODEL_INTERFACE_CHECK.json')['passed']
    if not(HERE/'TRAINING_FROZEN.receipt.json').exists():
        seal('TRAINING_FROZEN.receipt.json',[HERE/n for n in ('networks.py','train_model.py','prepare_inputs.py','FOLDS.json','INPUTS.json','SCENE_SET.json','MODEL_SOURCE_COPY.receipt.json')],
            source_only=True,seed_runs=210,all_hyperparameters_prespecified=True,heldout_labels_used=False)
    for stage in ('train_generator','train_critic'):
        if not(HERE/f'{stage}_COMPLETE.json').exists():
            run(stage);select_models(stage);write(HERE/f'{stage}_COMPLETE.json',dict(utc=utc(),stage=stage))
            print('COMPLETED',stage,flush=True)
    print('ALL_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(HERE/'TRAINING_FATAL.json',dict(utc=utc(),type=type(e).__name__,message=str(e),traceback=__import__('traceback').format_exc()))
        for r in jobs():
            if r['name'].startswith('xpl_'):cancel(r['id'],True)
        raise
