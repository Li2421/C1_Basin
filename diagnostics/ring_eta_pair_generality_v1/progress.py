from pathlib import Path
import json,subprocess,re
HERE=Path(__file__).resolve().parent
counts={}
for f in list(HERE.glob('main_21175_*.log'))+list(HERE.glob('helper_21203_*.log'))+list(HERE.glob('helper2_21209.log')):
 for line in f.read_text(errors='replace').splitlines():
  if line.startswith('{'):
   try:r=json.loads(line)
   except ValueError:continue
   if r.get('phase')=='main' and 'new' in r:counts[int(r['worker'])]=r['new']
for f in (HERE/'main').glob('worker_*.json'):
 r=json.load(open(f));counts[int(f.stem.split('_')[1])]=r['new']
queue=subprocess.check_output(['squeue','-h','-o','%u|%C|%j|%T'],text=True);running=[x.split('|') for x in queue.splitlines() if x and x.split('|')[-1]=='RUNNING'];job=subprocess.check_output(['scontrol','show','job','21206'],text=True);state=re.search(r'JobState=(\w+)',job).group(1);reason=re.search(r'Reason=(\S+)',job).group(1);out={'completed_lower_bound':sum(counts.values()),'planned':23296,'allocated_CPU':sum(int(x[1]) for x in running),'other_users':[x for x in running if x[0]!='zhihan'],'finalizer_state':state,'reason':reason,'worker_progress':counts}
active=stopped=0
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  if 'diagnostics/ring_eta_pair_generality_v1/' not in cmd or not any(x in cmd for x in ['/run.py main ', '/helper.py ', '/helper2.py ', '/merge.py ', '/analyze.py ']):continue
  st=(p/'stat').read_text().split(') ')[1].split()[0]
  if st=='Z':continue
  if st in ['T','t']:stopped+=1
  else:active+=1
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
out.update(active_own_CPU=active,stopped_workers=stopped);print(json.dumps(out))
