"""Emit only the still absent screening tuples; never duplicate a rollout."""
import hashlib, json
from pathlib import Path
HERE=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_local_basin_continuity_v1')
def dig(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 p=json.loads((HERE/'screen_plan.json').read_text()); seen=set()
 for directory in (HERE/'raw').glob('screening*'):
  if not directory.is_dir(): continue
  for f in directory.glob('shard*.jsonl'):
   for line in f.read_text().splitlines():
    r=json.loads(line); seen.add((r['arm_id'],int(r['seed'])))
 arms=[]
 for a in p['arms']:
  seeds=[s for s in a['seeds'] if (a['arm_id'],int(s)) not in seen]
  if seeds:
   b=dict(a); b['seeds']=seeds; arms.append(b)
 plan={'schema':'orthoflow3_local_basin_screening_completion_v1','basis_family':'orthoflow3','parent_screen_plan_sha256':p['content_sha256'],
       'reason':'final exact coverage completion after cancelling original shards; seed tuples not present in immutable raw outputs only','arms':arms,'new_continuations':sum(map(lambda a:len(a['seeds']),arms)),'partition_count':1}
 plan['content_sha256']=dig(plan)
 (HERE/'screen_completion2_plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'arms':len(arms),'tuples':plan['new_continuations']},indent=2))
if __name__=='__main__':main()
