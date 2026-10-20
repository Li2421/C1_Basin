from support import *
from train_gaussian import train
import subprocess
G=guard()
if __name__=='__main__':
 prereg();assert (HERE/'INTEGRITY_BEGIN.json').exists();i=int(sys.argv[1]);source_inventory(f'imports_training_before_{i}.json')
 for j in range(i,4,2):train(j)
 write(HERE/f'train_worker{i}.json',dict(utc=utc(),complete=True,guard=guard(),backend=jax.default_backend(),devices=[str(d) for d in jax.devices()]));source_inventory(f'imports_training_after_{i}.json')
