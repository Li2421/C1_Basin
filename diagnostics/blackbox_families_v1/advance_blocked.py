"""Complete unaffected families after a preregistered implementation gap."""
from common import *
import subprocess,fcntl,time
from shared_rollout_db.src.rollout_db import connect
blocked=read(HERE/'MAIN_BLOCKED_UNCLAIMED.json');skipped=set(blocked['skipped_rollout_uids']);cases=read(HERE/'main_MISSING_CASES.json');expected=len(cases)-len(skipped)
while True:
 paths=list((HERE/'main/raw').glob('*.json'));n=len(paths);q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C|%b'],text=True)
 with (HERE/'RESOURCE_SAMPLES.jsonl').open('a') as file:file.write(json.dumps(dict(utc=utc(),queue=q,allocated_cpu=sum(int(x.split('|')[3]) for x in q.splitlines() if x.split('|')[2] in ['RUNNING','CONFIGURING','COMPLETING']),gpu_shard_jobs=sum('shard' in x.split('|')[4] for x in q.splitlines())))+'\n')
 write(HERE/'LIVE_STATUS.json',dict(utc=utc(),main_raw=n,main_expected_including_retained_engineering=expected,confirmatory_evaluation_episodes=30*32*2*8,blocked_unclaimed_not_executed=len(skipped),blocked_claimed_retained=blocked['already_claimed_blocked_cases_retained'],selections_frozen=True,main_labels_started=True,queue=q))
 print('main',n,'/',expected,'; blocked unclaimed',len(skipped),flush=True)
 if n==expected and 'r38_main' not in q:break
 time.sleep(20)
with (HERE/'OFFICIAL_MERGER.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with (HERE/'main_MERGE.log').open('w') as file:subprocess.run([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,stdout=file,stderr=subprocess.STDOUT,check=True)
post=planner.preflight(HERE/'main_PLANNED_ROLLOUTS.json');write(HERE/'main_POSTFLIGHT.json',post);absent=[];unknown=[]
with connect(True) as db:
 for c in read(HERE/'main_CASES.json'):
  r=db.execute('SELECT numerical_failure FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
  if r is None:absent.append(c['rollout_uid'])
  elif r['numerical_failure']:unknown.append(c['rollout_uid'])
assert set(absent)==skipped,(len(absent),len(skipped));write(HERE/'main_POSTFLIGHT_UNKNOWN.json',dict(utc=utc(),absent_rows=absent,intentionally_blocked_unclaimed=sorted(skipped),unknown_preserved=unknown,no_unknown_reruns=True,no_absent_valid_family_rows=True))
write(HERE/'READY_FOR_FINAL_ANALYSIS.json',dict(utc=utc(),all_required_available_families_complete=True,blocked_family='prioritized_astar',confirmatory_families=5,confirmatory_episodes=30*32*2*8,retained_engineering_episodes=blocked['already_claimed_blocked_cases_retained']))
print('AVAILABLE FAMILY EXECUTIONS COMPLETE; BLOCKED FAMILY GAP RETAINED',flush=True)
