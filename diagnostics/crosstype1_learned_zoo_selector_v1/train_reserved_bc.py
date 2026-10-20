"""Extra BC workers execute disjoint reserved instances once."""
from common import *
import traceback

def main(worker):
    reservation=read(HERE/'BC_PARALLEL_RESERVATION.json')
    for index,name in reservation['workers'][str(worker)]:
        path=HERE/'training_attempts'/f'{name}.json';prior=read(path)
        assert prior['status']=='RESERVED' and prior['reservation']==reservation['id']
        write(path,dict(utc=utc(),controller=name,status='STARTED',worker=f'bc_extra_{worker}',reservation=reservation['id']))
        try:
            from train_bc import train
            train(index)
            write(path,dict(utc=utc(),controller=name,status='TRAINED',worker=f'bc_extra_{worker}',locked_arm_labels_opened=False))
        except Exception as exc:
            write(HERE/'training_attempts'/f'{name}.failure.json',dict(utc=utc(),controller=name,status='GAP',
                  error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),retry_allowed=False,partial_training_retained=True))
            print(name,'GAP',type(exc).__name__,str(exc),flush=True)
    write(HERE/f'BC_EXTRA_WORKER_{worker}_COMPLETE.json',dict(utc=utc(),locked_arm_labels_opened=False))

if __name__=='__main__':main(int(sys.argv[1]))
