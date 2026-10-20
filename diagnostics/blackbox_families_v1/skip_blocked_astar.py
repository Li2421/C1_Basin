"""Skip only unclaimed blocked-family tasks; never cancel or rerun a case."""
from common import *
import fcntl,time
rows=read(HERE/'main_MISSING_CASES.json');indices=[i for i,c in enumerate(rows) if c['family']=='prioritized_astar'];lo=min(indices);end=max(indices)+1;assert indices==list(range(lo,end));q=HERE/'main/queue/cursor'
write(HERE/'IMPLEMENTATION_GAPS.json',dict(utc=utc(),families={'prioritized_astar':dict(full_name='优先级时空 A* 规划',reason='Required empty-map straightness fails for oblique goals; initial unit test covered axis-aligned path only.',unit_test_path=str(HERE/'ASTAR_OBLIQUE_UNIT_TEST.json'),unit_test_sha256=sha(HERE/'ASTAR_OBLIQUE_UNIT_TEST.json'),capability_observed='PASS 6/6, upstream pilot only',confirmatory_status='GAP',no_parameter_or_selection_changes=True)},evidence_exclusion_based_on_implementation_test_not_evaluation_outcomes=True))
while True:
 with q.open('r+') as file:
  fcntl.flock(file,fcntl.LOCK_EX);i=int(file.read())
  if i>=lo:
   stop=min(i,end);skipped=rows[stop:end];file.seek(0);file.truncate();file.write(str(max(i,end)));file.flush();break
 time.sleep(.02)
write(HERE/'MAIN_BLOCKED_UNCLAIMED.json',dict(utc=utc(),blocked_family='prioritized_astar',range_start=lo,range_end=end,cursor_before=i,cursor_after=max(i,end),skipped_cases=skipped,skipped_rollout_uids=[c['rollout_uid'] for c in skipped],already_claimed_blocked_cases_retained=max(0,min(i,end)-lo),new_physical_steps_for_skipped=0,no_workers_cancelled=True,no_unknown_reruns=True))
print('skipped',len(skipped),'unclaimed A-star cases; retained',max(0,min(i,end)-lo),'in-flight cases',flush=True)
