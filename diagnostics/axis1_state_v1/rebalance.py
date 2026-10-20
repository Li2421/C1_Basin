from common import *
import signal,time,subprocess,fcntl
from shared_rollout_db.src.planner import preflight

def main():
 prereg();pids=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   args=(p/'cmdline').read_bytes().split(b'\0')
   if str(HERE/'run.py').encode() in args and b'main' in args:pids.append(int(p.name))
  except (PermissionError,FileNotFoundError,ProcessLookupError):continue
 assert len(pids)==6,pids
 rows=read(HERE/'main_MISSING_CASES.json');paused=[];reserved=[];excluded=[];preexisting=[];success=False
 try:
  for pid in pids:os.kill(pid,signal.SIGSTOP);paused.append(pid)
  for _ in range(100):
   if all('T' in (Path('/proc')/str(pid)/'status').read_text().split('State:')[1].splitlines()[0] for pid in paused):break
   time.sleep(.02)
  else:raise RuntimeError('Could not confirm all original workers stopped')
  for worker in range(6):
   original=rows[worker::6];firstmissing=next((i for i,c in enumerate(original) if not (HERE/'main/raw'/f"{c['rollout_uid']}.json").exists()),len(original))
   # Keep the possibly in-flight first missing case with its original worker.
   excluded.extend(original[:min(firstmissing+1,len(original))])
   for c in original[firstmissing+1:]:
    out=HERE/'main/raw'/f"{c['rollout_uid']}.json"
    if out.exists():preexisting.append(c['rollout_uid']);continue
    reserved.append(c)
  assert len({c['rollout_uid'] for c in reserved})==len(reserved)
  # Keep nearby controllers together for checkpoint/JIT reuse while sharing work globally.
  order={c['rollout_uid']:i for i,c in enumerate(rows)};reserved.sort(key=lambda c:order[c['rollout_uid']])
  manifest=HERE/'SPEEDUP_PLANNED_ROLLOUTS.json';write(manifest,{'requests':[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in reserved]});pf=preflight(manifest);write(HERE/'SPEEDUP_CACHE_PREFLIGHT.json',pf);assert pf['summary']['genuinely_missing']==len(reserved) and pf['summary']['ambiguous']==0
  write(HERE/'SPEEDUP_CASES.json',reserved);(HERE/'SPEEDUP_CURSOR').write_text('0')
  for c in reserved:
   out=HERE/'main/raw'/f"{c['rollout_uid']}.json"
   with out.open('x') as f:f.write(json.dumps(dict(schema='round32_work_reservation_v1',rollout_uid=c['rollout_uid'],not_an_outcome=True)))
  write(HERE/'SPEEDUP_OWNERSHIP.json',dict(utc=utc(),original_worker_pids=pids,original_workers=6,transferred=len(reserved),kept_original_including_possibly_inflight=len(excluded),preexisting_not_touched=preexisting,process_pause_only_no_episode_restarts=True,raw_reservation_files_are_not_results=True,quota_target_total_CPU=18,do_not_restart_current_or_unknown=True))
  seal('SPEEDUP_PREFLIGHT.receipt.json',[manifest,HERE/'SPEEDUP_CACHE_PREFLIGHT.json',HERE/'SPEEDUP_CASES.json',HERE/'SPEEDUP_OWNERSHIP.json',HERE/'run_speedup.py',HERE/'shared_queue.py'],before_submission=True,original_records_preserved=True)
  success=True
  print('transferred',len(reserved),'kept original',len(excluded),flush=True)
 finally:
  for pid in paused if success or not reserved else []:
   try:os.kill(pid,signal.SIGCONT)
   except ProcessLookupError:pass
if __name__=='__main__':main()
