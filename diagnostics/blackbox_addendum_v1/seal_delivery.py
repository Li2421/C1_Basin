from core import *
import subprocess,time
pid=int((HERE/'QUOTA_GUARD_PID').read_text())
for _ in range(12):
 r=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
 if not r.stdout.strip() or r.stdout.strip().startswith('Z'):break
 time.sleep(1)
else:raise RuntimeError('Governor still live; cannot seal resource monitor')
queue=subprocess.check_output(['squeue','-r','-h','-u','zhihan','-o','%i|%j|%T|%C|%b'],text=True)
owned={str(read(p)['job']) for p in HERE.glob('*_SUBMISSION.json') if 'job' in read(p)}
assert not [line for line in queue.splitlines() if line.split('|')[0].split('_')[0] in owned]
write(HERE/'FINAL_QUEUE_CHECK.json',dict(utc=utc(),queue=queue.splitlines(),current_experiment_jobs_remaining=0,governor_exited=True))
names=['PREREGISTRATION.md','results.json','REPORT.md','HANDOFF.md','AXIS_TABLE.md','AXIS_TABLE.csv','METHOD_METRICS.json','GAPS.md','GAPS.json','FINAL_VERIFICATION.json','INTEGRITY_BEGIN.json','INTEGRITY_END.json','INTEGRITY_COMPARISON.json','ASSET_FINAL_VERIFICATION.json','IMPORTED_SOURCE_FINAL.json','COPY_PROVENANCE.json','COPY_DIFF_REVIEW.json','DOCUMENT_REVIEW.json','RESOURCE_MONITOR.json','RESOURCE_EXCEPTION.json','REPAIRS_RESOLVED.json','RUN_STATUS.json','OPERATOR_STATE.json','FINAL_QUEUE_CHECK.json']
paths={HERE/n for n in names}
for pattern in ['*.receipt.json','*.py','*.sbatch','*_TABLE.md','*_TABLE.csv','A_CONTROLLER_RADIUS_CURVES.*','A_FAILURE_DENOMINATOR_CLARIFICATION.*']:
 paths.update(HERE.glob(pattern))
assert all(p.is_file() for p in paths)
seal('COMPLETION.receipt.json',sorted(paths),parts=['A','B','C','D','E','F'],all_authorized_branches_resolved=True,E_main_and_transfer='Not eligible; all fixed-instance capability gates failed',unknown_preserved=True,no_git_mutations=True,no_package_install=True,resource_exception_disclosed=True,no_video_required=True)
r=read(HERE/'COMPLETION.receipt.json');bad=[p for p,h in r['files'].items() if sha(p)!=h];assert not bad,bad
print(json.dumps(dict(sealed_files=len(r['files']),completion_hashes_passed=True,own_jobs_remaining=0,governor_exited=True,report_first_paragraph=(HERE/'REPORT.md').read_text().split('\n\n')[0]),ensure_ascii=False,indent=2))
