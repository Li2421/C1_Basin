from pathlib import Path
import json,time,datetime,subprocess
h=Path(__file__).resolve().parent
while True:
 logs=list(h.glob('main_20580_*.log'));latest=[]
 for p in logs:
  for line in reversed(p.read_text().splitlines()):
   if line.startswith('{"phase"'):
    latest.append(json.loads(line));break
 count=len(list((h/'main/raw').glob('*.json')));elapsed=max((r['wall_seconds'] for r in latest),default=0);remaining=(21504-count)*elapsed/max(count,1)/60
 queue=subprocess.check_output(['squeue','-h','-o','%i %u %C %T %j']).decode().splitlines();status=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_file_count=count,planned=21504,elapsed_minutes=round(elapsed/60,2),linear_remaining_minutes=round(remaining,1),worker_reports=len(latest),queue=queue,no_success_labels_read=True)
 (h/'RUN_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
 if (h/'REPORT.md').exists():print('REPORT_READY',flush=True);break
 if count<21504 and not any('r14a_main' in r for r in queue):print('WORKERS_STOPPED_BEFORE_FULL_COUNT',flush=True);break
 if count==21504 and not any('r14a_finalize' in r for r in queue) and not (h/'REPORT.md').exists():print('FINALIZER_ENDED_WITHOUT_REPORT',flush=True);break
 time.sleep(50)
