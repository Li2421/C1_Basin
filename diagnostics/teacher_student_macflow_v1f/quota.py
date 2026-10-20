"""Read-only Slurm census; submission is refused without free user-wide quota."""
import subprocess,json,pathlib,datetime,os,sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]))
import shared_rollout_db
import shared_rollout_db.src.cache_writer as _writer
assert pathlib.Path(shared_rollout_db.__file__).resolve().is_relative_to(pathlib.Path(__file__).resolve().parents[2]/'shared_rollout_db')
assert (_writer.ROOT/'journals').resolve()==pathlib.Path(__file__).resolve().parents[2]/'shared_rollout_db/journals'
H=pathlib.Path(__file__).resolve().parent
def census():
 raw=subprocess.check_output(['squeue','-u',os.environ.get('USER','zhihan'),'-h','-r','-o','%i|%T|%C|%b|%j'],text=True);rows=[]
 for line in raw.splitlines():
  jid,state,cpu,tres,name=line.split('|');gpu=0
  for t in tres.split(','):
   if t.startswith('gres/shard:') or t.startswith('shard:'):gpu+=int(t.rsplit(':',1)[1])
   elif t.startswith('gres/gpu:') or t.startswith('gpu:'):gpu+=2*int(t.rsplit(':',1)[1])
  rows.append(dict(job=jid,state=state,cpus=int(cpu),gpu_shards=gpu,tres=tres,name=name))
 running=[r for r in rows if r['state']!='PENDING'];reserved=rows
 return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,cpu=sum(r['cpus'] for r in reserved),gpu=sum(r['gpu_shards'] for r in reserved),running_cpu=sum(r['cpus'] for r in running),running_gpu=sum(r['gpu_shards'] for r in running),raw=raw)
def submit(args,gpu=0):
 c=census()
 if c['cpu']+1>18 or c['gpu']+gpu>2:return None
 result=subprocess.check_output(['sbatch','--parsable',*args],text=True).strip().split(';')[0];c.update(submission=args,job=result,requested_cpu=1,requested_gpu=gpu)
 with (H/'RESOURCE_SUBMISSIONS.jsonl').open('a') as f:f.write(json.dumps(c)+'\n')
 return result
if __name__=='__main__':
 c=census();print(json.dumps(c,indent=2))
 if len(sys.argv)>1 and sys.argv[1]=='check':assert c['running_cpu']<=18 and c['running_gpu']<=2,'User-wide quota changed; stop before scientific computation'
