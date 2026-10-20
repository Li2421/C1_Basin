"""Three extra GPU workers own pre-reserved PPO instances; never duplicate BC workers."""
from common import *
import traceback


def main(worker):
    reservation=read(HERE/'IPPO_PARALLEL_RESERVATION.json')
    for index in reservation['workers'][str(worker)]:
        name=f'ring_ippo{index}';path=HERE/'training_attempts'/f'{name}.json'
        prior=read(path)
        assert prior['status']=='RESERVED' and prior['reservation']==reservation['id']
        write(path,dict(utc=utc(),controller=name,status='STARTED',worker=f'ippo_extra_{worker}',
                        reservation=reservation['id'],locked=True))
        try:
            from train_ippo import train
            train(index)
            write(path,dict(utc=utc(),controller=name,status='TRAINED',worker=f'ippo_extra_{worker}',
                            checkpoint_selection_pending=True,locked_arm_labels_opened=False))
        except Exception as exc:
            write(HERE/'training_attempts'/f'{name}.failure.json',dict(utc=utc(),controller=name,
                status='GAP',error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
                retry_allowed=False,partial_training_retained=True))
            print(name,'GAP',type(exc).__name__,str(exc),flush=True)
    write(HERE/f'IPPO_EXTRA_WORKER_{worker}_COMPLETE.json',dict(utc=utc(),locked_arm_labels_opened=False))


if __name__=='__main__':main(int(sys.argv[1]))
