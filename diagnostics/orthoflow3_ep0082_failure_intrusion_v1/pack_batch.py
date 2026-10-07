from core import *
def main():
 stages=tuple(sys.argv[1:]) or ('refine2','escape2');name='_'.join(stages);assert not (H/f'plans/{name}').exists()
 unique={}
 for stage in stages:
  for p in sorted((H/f'plans/{stage}').glob('shard*.jsonl')):
   for line in open(p):
    r=json.loads(line);unique.setdefault((r['eta_key'],r['future_index']),r)
 eta_keys=sorted({k[0] for k in unique});(H/f'plans/{name}').mkdir(parents=True)
 for shard in range(2):
  with open(H/f'plans/{name}/shard{shard}.jsonl','w') as out:
   for j,k in enumerate(eta_keys):
    if j%2==shard:
     for fi in range(64):out.write(json.dumps(unique[k,fi])+'\n')
 cost=dict(question='Execute independently frozen stages together without changing coordinates or seed identities.',new_exact_Q64=len(eta_keys),cached_Q64=0,maximum_new_continuations=len(unique),GPU_shards=2,CPU_threads=4,estimated_wall_seconds=120+42*len(eta_keys)/2,source_stages=list(stages),protocol_changed=False)
 dump(f'rounds/{name}/cost_estimate.json',cost)
 print(json.dumps(cost))
if __name__=='__main__':main()
