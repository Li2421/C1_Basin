"""Freeze the original six recipes with only the specified reward change."""
from common import *
import traceback


def design():
    prereg();old=D/'crosstype1_learned_zoo_selector_v1';source=old/'IPPO_TRAINING_CONFIGS.json'
    configs=read(source)
    for cfg in configs:
        original=dict(cfg['reward']);cfg['name']=cfg['name'].replace('ring_ippo','ring_ippo_conservative')
        cfg['family']='ring_ippo_conservative';cfg['locked']=False
        cfg['reward'].update(own_collision=-60,other_collision=-20,proximity_gap=.2,proximity_coefficient=.05)
        cfg['original_reward']=original
        cfg['reward_only_change_note']='Original recipe, seeds, initial states, normalization, network, optimizer and budget retained. Swept nearest robot/obstacle/outer-wall clearance used for per-step proximity; inactive robot pairs omitted.'
        for p,digest in cfg['input_sha256'].items():assert sha(p)==digest
    write(HERE/'IPPO_TRAINING_CONFIGS.json',configs)
    seal('CONSERVATIVE_TRAINING_CONFIGS.receipt.json',[HERE/'IPPO_TRAINING_CONFIGS.json',HERE/'training_env.py',HERE/'conservative_reward.py',HERE/'train_ippo.py',HERE/'ippo.py'],before_training=True,source_config=str(source),source_config_sha256=sha(source),no_training_outcomes_read=True)


def run(worker):
    from train_ippo import train
    for index in range(worker,6,2):
        name=read(HERE/'IPPO_TRAINING_CONFIGS.json')[index]['name'];folder=HERE/'training_attempts';folder.mkdir(exist_ok=True)
        claim=folder/f'{name}.claim'
        fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644);os.close(fd)
        write(folder/f'{name}.json',dict(utc=utc(),status='STARTED',index=index,worker=worker))
        try:
            train(index)
            write(folder/f'{name}.json',dict(utc=utc(),status='TRAINED',index=index,checkpoint_selection_pending=True))
        except Exception as exc:
            write(folder/f'{name}.failure.json',dict(utc=utc(),status='GAP',error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),retry_allowed=False))
            raise
    write(HERE/f'CONSERVATIVE_WORKER_{worker}_COMPLETE.json',dict(utc=utc(),trained=3))


if __name__=='__main__':
    if sys.argv[1]=='run':run(int(sys.argv[2]))
    else:globals()[sys.argv[1]]()
