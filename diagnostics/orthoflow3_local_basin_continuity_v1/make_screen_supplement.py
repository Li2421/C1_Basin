"""Split only unfinished screening seeds after two Slurm shards were stopped.

The plan is seed-granular: existing valid rollout tuples are excluded rather
than re-run.  It is deliberately restricted to original modulo-4 shards 2/3,
whose jobs have been cancelled; shards 0/1 remain the sole writers for their
own tuples.
"""
import hashlib, json
from pathlib import Path

HERE=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_local_basin_continuity_v1')
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    old=json.loads((HERE/'screen_plan.json').read_text())
    seen=set()
    for f in (HERE/'raw'/'screening').glob('shard*.jsonl'):
        for line in f.read_text().splitlines():
            r=json.loads(line); seen.add((r['arm_id'],int(r['seed'])))
    arms=[]
    for i,arm in enumerate(old['arms']):
        if i%4 not in (2,3): continue
        missing=[s for s in arm['seeds'] if (arm['arm_id'],int(s)) not in seen]
        if missing:
            copy=dict(arm); copy['seeds']=missing; arms.append(copy)
    plan={'schema':'orthoflow3_local_basin_screening_supplement_v1','basis_family':'orthoflow3',
          'reason':'user-authorized idle-server increase 4 to 6 total shards; original shards 2/3 cancelled; excludes every completed arm/seed tuple',
          'parent_screen_plan_sha256':old['content_sha256'],'arms':arms,
          'new_continuations':sum(len(a['seeds']) for a in arms),'partition_count':4}
    plan['content_sha256']=digest(plan)
    (HERE/'screen_supplement_plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    (HERE/'screen_supplement_audit.json').write_text(json.dumps({'completed_tuples_excluded':len(seen),'unfinished_tuples':plan['new_continuations'],'original_indices': [2,3]},indent=2)+'\n')
    print(json.dumps({'arms':len(arms),'unfinished':plan['new_continuations']},indent=2))
if __name__=='__main__': main()
