from common import *
from shared_rollout_db.src.rollout_db import connect
counts=dict(recoverable_four_agents=0,known_other_agent_count=0,unrecoverable=0);sizes={}
with connect(True) as db:
 for row in db.execute('SELECT physical_state_json FROM state'):
  try:
   v=json.loads(row['physical_state_json']);arrays=[]
   for keys in [('positions','initial_positions'),('velocities','initial_velocities'),('goals',)]:arrays.append(np.asarray(next(v[k] for k in keys if k in v),float))
   shape=arrays[0].shape
   if len(shape)!=2 or shape[-1]!=2 or not all(a.shape==shape for a in arrays):raise ValueError('layout')
   if shape[0]==4:counts['recoverable_four_agents']+=1
   else:counts['known_other_agent_count']+=1;sizes[str(shape[0])]=sizes.get(str(shape[0]),0)+1
  except (TypeError,ValueError,KeyError,StopIteration):counts['unrecoverable']+=1
original=read(HERE/'STATE_DEDUP.json');assert counts['known_other_agent_count']+counts['unrecoverable']==original['unrecoverable_physical']
write(HERE/'STATE_DEDUP_VISIBILITY_REFINEMENT.json',dict(utc=utc(),counts=counts,known_other_agent_counts=sizes,original_overinclusive_unrecoverable_count=original['unrecoverable_physical'],only_new_four_agent_states_registered_since_initial_scan=True,interpretation='Known non-four-agent physical arrays cannot equal the new four-agent states; only unrecoverable records remain a physical-dedup visibility gap. Original preflight record preserved.'))
print(counts,sizes)
