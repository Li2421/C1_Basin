from common import *
import subprocess,shlex
prereg();assert read(HERE/'REPRODUCTION_GATE.json')['passed'];assert read(HERE/'FRAME_TESTS.json')['passed']
q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C|%b'],text=True);assert not q.strip(),'Account all concurrent reservations before training submission'
cmd=f'JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=.40 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {shlex.quote(sys.executable)} -B {shlex.quote(str(HERE/"train_policies.py"))} $SLURM_ARRAY_TASK_ID 2'
args=['sbatch','--parsable','-p','gpu','--gres=shard:1','--cpus-per-task=1','--mem=20G','--time','24:00:00','--array','0-1%2','-J','r38_train','-o',str(HERE/'training_%A_%a.log'),'--wrap',cmd]
seal('TRAINING_SUBMISSION.receipt.json',[HERE/'TRAINING_CONFIGS.json',HERE/'train_policies.py',HERE/'learned.py',HERE/'FRAME_TESTS.json'],before_training=True,queue=q,CPU=2,GPU_shards=2)
job=subprocess.check_output(args,text=True).strip();write(HERE/'TRAINING_SUBMISSION.json',dict(utc=utc(),job=job,args=args));print(job)
