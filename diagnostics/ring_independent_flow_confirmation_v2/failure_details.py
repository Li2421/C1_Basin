from core import *
from collections import Counter,defaultdict
prereg();labels_allowed();gate();assert (HERE/'ENDPOINT_OPENING.receipt.json').exists();rows=read(HERE/'FAILURE_RECORDS.json');summary={}
for controller,records in rows.items():
 groups=defaultdict(list)
 for r in records:groups[r['arm'],r['outcome'],r['error_type']].append(r['occurrence_step'])
 summary[controller]=[dict(arm=a,outcome=o,error_type=e,count=len(steps),min_recorded_step=min(steps),median_recorded_step=float(np.median(steps)),max_recorded_step=max(steps),time_seconds_range=[.05*min(steps),.05*max(steps)],interpretation='native termination step or recorded numerical error action index; timeout is horizon, no deadlock or causal attribution') for (a,o,e),steps in sorted(groups.items(),key=lambda x:str(x[0]))]
write(HERE/'FAILURE_TIMING_SUMMARY.json',dict(utc=utc(),controllers=summary,descriptive_only=True,numerical_unknown_never_relabelled_failure=True,dt=.05))
print('Descriptive failure times exported')
