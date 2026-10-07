"""Resource-only amendment after cancelled shards duplicated batches.

No screening result is read.  All +/-1 pairs survive; exactly six selected
/-4 pairs with the largest remaining horizon are removed to retain the hard
15,000-continuation limit after the unavoidable duplicate computation.
"""
import hashlib,json
from pathlib import Path
HERE=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_local_basin_continuity_v1')
def dig(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 old=json.loads((HERE/'cross_transfer_plan.json').read_text())
 p4=[a for a in old['arms'] if abs(int(a['offset_steps']))==4]
 drop={a['arm_id'] for a in sorted(p4,key=lambda a:(850-int(a['absolute_step']),str(a['arm_id'])),reverse=True)[:6]}
 arms=[a for a in old['arms'] if a['arm_id'] not in drop]
 new={'schema':'orthoflow3_local_basin_cross_transfer_plan_amended_v1','basis_family':'orthoflow3','parent_plan_sha256':old['content_sha256'],
      'amendment_reason':'Slurm cancellation race caused 1120 duplicate completed screening tuples; resource-only removal of six longest-horizon preselected +/-4 directional transfer pairs; no outcome file inspected',
      'selection_rule':'all +/-1 original pairs retained; remove six selected +/-4 pairs by greatest remaining horizon, tie arm_id descending','arms':arms,
      'dropped_arm_ids':sorted(drop),'new_continuations':sum(len(a['seeds']) for a in arms),'physical_step_upper_bound':sum((850-a['absolute_step'])*len(a['seeds']) for a in arms),'partition_count':6}
 new['content_sha256']=dig(new)
 (HERE/'cross_transfer_plan_amended.json').write_text(json.dumps(new,indent=2,sort_keys=True)+'\n')
 (HERE/'resource_amendment.json').write_text(json.dumps({'duplicate_screening_tuples':1120,'additional_completion_tuples_after_job392':1168,'cross_removed_pairs':len(drop),'cross_removed_tuples':old['new_continuations']-new['new_continuations'],'new_cross_tuples':new['new_continuations']},indent=2)+'\n')
 print(json.dumps({'kept':len(arms),'dropped':len(drop),'new_continuations':new['new_continuations'],'drops':sorted(drop)},indent=2))
if __name__=='__main__':main()
