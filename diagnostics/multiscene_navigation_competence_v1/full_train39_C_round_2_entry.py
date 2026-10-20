from common import *
from training39 import train
for task in read(HERE/'full_train39_C_round_2_TASKS.json')[int(sys.argv[1])::2]:
 try:
  train(**task)
 except Exception as error:
  tag="{family}_{kind}_{seed}_round_{round_index}".format(**task)
  write(HERE/"training_errors"/(tag+".json"),dict(task=task,type=type(error).__name__,message=str(error)))
