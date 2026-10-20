"""Failure timing from immutable trace intervals; no new rollout."""
from core import *
from audit_utils import source_inventory
import collections,csv

def main():
    by=collections.defaultdict(list);rows=[]
    for f in sorted((HERE/'main/raw').glob('*.json')):
        r=read(f);c=r['case'];z=np.load(r['trace_path']);mask=z['slow_mask'];first=[]
        for i in range(4):
            delta=np.diff(np.r_[False,mask[:,i],False].astype(int));start=np.flatnonzero(delta==1);end=np.flatnonzero(delta==-1);qualified=start[(end-start)>=40]
            if len(qualified):first.append(float(qualified.min()*.05))
        onset=min(first) if first else None;q=dict(controller=c['base_controller_name'],arm=c['arm'],state=c['state_index'],seed=c['seed'],outcome=r['outcome'],failure_interval_end_seconds=r['physical_steps']*.05,first_qualifying_stall_onset_seconds=onset,stall_detection_seconds=onset+2 if onset is not None else None);rows.append(q);by[c['arm']].append(q)
    summary={}
    for arm,rs in by.items():
        fail=[r for r in rs if r['outcome']!='success'];on=[r['first_qualifying_stall_onset_seconds'] for r in fail if r['first_qualifying_stall_onset_seconds'] is not None];times=[r['failure_interval_end_seconds'] for r in fail];summary[arm]=dict(failure_types=dict(collections.Counter(r['outcome'] for r in fail)),failure_interval_end_median=float(np.median(times)) if times else None,failed_episodes_with_qualifying_low_speed=len(on),first_stall_onset_median=float(np.median(on)) if on else None,first_stall_onset_range=[float(min(on)),float(max(on))] if on else None)
    write(HERE/'FAILURE_TIMING.json',dict(utc=utc(),arms=summary,definition='low-speed diagnostic interval onset: velocity<.01 and not at goal continuously>=40steps; not a deadlock criterion; normal queues may qualify',collision_time='reported interval endpoint; contact occurred within previous .05s interval',new_rollouts=0))
    with (HERE/'failure_timing.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['| Arm | Failed termination types | Median terminal interval end (s) | Failures with qualifying low speed | Median first low-speed interval onset (s) |','|---|---|---:|---:|---:|']
    for a,r in summary.items():lines.append(f"| {a} | {r['failure_types']} | {r['failure_interval_end_median']} | {r['failed_episodes_with_qualifying_low_speed']} | {r['first_stall_onset_median']} |")
    (HERE/'failure_timing_table.md').write_text('\n'.join(lines)+'\n\nLow-speed runs are diagnostic, including possible normal queueing, not deadlocks.\n');source_inventory('imports_onsets.json')
if __name__=='__main__':main()
