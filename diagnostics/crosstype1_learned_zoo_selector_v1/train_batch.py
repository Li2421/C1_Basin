"""Two fixed GPU workers. Failed instances are recorded, never silently retrained."""
from common import *


def main(worker):
    configs=read(HERE/'TRAINING_CONFIGS.json')
    have_ppo=(HERE/'IPPO_TRAINING_CONFIGS.json').exists()
    jobs=[('bc',i,c['name']) for i,c in enumerate(configs)]
    if have_ppo:
        jobs += [('ippo',i,c['name']) for i,c in enumerate(read(HERE/'IPPO_TRAINING_CONFIGS.json'))]
    for slot,(kind,index,name) in enumerate(jobs):
        if slot%2!=worker:continue
        receipt=HERE/'training_attempts'/f'{name}.json'
        if receipt.exists():
            print('Retaining prior attempt',name,flush=True);continue
        if name.startswith('implicit') and not read(HERE/'IMPLICIT_SYNTHETIC_TEST.json')['passed']:
            write(receipt,dict(utc=utc(),controller=name,status='GAP',reason='Required synthetic two-mode test failed'))
            continue
        write(receipt,dict(utc=utc(),controller=name,status='STARTED',worker=worker,locked=name.startswith(('implicit','ring_ippo'))))
        try:
            if kind=='bc':
                from train_bc import train
            else:
                from train_ippo import train
            train(index)
            write(receipt,dict(utc=utc(),controller=name,status='TRAINED',worker=worker,
                               checkpoint_selection_pending=kind=='ippo',locked_arm_labels_opened=False))
        except Exception as exc:
            import traceback
            write(HERE/'training_attempts'/f'{name}.failure.json',dict(utc=utc(),controller=name,
                  status='GAP',error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
                  retry_allowed=False,partial_training_retained=True))
            print(name,'GAP',type(exc).__name__,str(exc),flush=True)
    write(HERE/f'TRAINING_WORKER_{worker}_COMPLETE.json',dict(utc=utc(),jobs=len(jobs),locked_arm_labels_opened=False))

if __name__=='__main__':main(int(sys.argv[1]))
