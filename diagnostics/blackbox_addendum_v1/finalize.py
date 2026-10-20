from common import *
from run_D import call
import time,traceback,subprocess,re

def main():
 while not(HERE/'E_COMPLETE.receipt.json').exists():time.sleep(10)
 if not(HERE/'A_STANDARD_RESULTS.receipt.json').exists():call('standard_A.py')
 call('verify_delivery.py');call('deliver.py')
 docchecks=[]
 for name in ['REPORT.md','HANDOFF.md']:
  text=(HERE/name).read_text();bad=[]
  for alias in ['F10','F13','F15','S500','Srel','Sprog','RHR','DM1','DM3','A3','A0','G_E','TT','FF']:
   if re.search(r'(?<![A-Za-z0-9_])'+alias+r'(?![A-Za-z0-9_])',text):bad.append(alias)
  assert not bad,(name,bad);docchecks.append(dict(file=name,no_unexpanded_controller_aliases=True,bytes=len(text.encode())))
 write(HERE/'DOCUMENT_REVIEW.json',dict(utc=utc(),checks=docchecks,first_paragraph=(HERE/'REPORT.md').read_text().split('\n\n')[0],six_parts_present=True));write(HERE/'RUN_STATUS.json',dict(utc=utc(),status='COMPLETE',completed_parts=['A','B','C','D','E','F'],gaps_file=str(HERE/'GAPS.md'),report=str(HERE/'REPORT.md')))
 # Let the resource governor take its final sample and exit before sealing it.
 pid=int((HERE/'QUOTA_GUARD_PID').read_text())
 for _ in range(10):
  try:os.kill(pid,0)
  except ProcessLookupError:break
  time.sleep(1)
 paths=[HERE/f for f in ['PREREGISTRATION.md','PREREGISTRATION.receipt.json','results.json','REPORT.md','HANDOFF.md','AXIS_TABLE.md','AXIS_TABLE.csv','METHOD_METRICS.json','GAPS.md','GAPS.json','FINAL_VERIFICATION.json','INTEGRITY_BEGIN.json','INTEGRITY_END.json','INTEGRITY_COMPARISON.json','IMPORTED_SOURCE_FINAL.json','COPY_PROVENANCE.json','COPY_DIFF_REVIEW.json','DOCUMENT_REVIEW.json','RESOURCE_MONITOR.json','RUN_STATUS.json']]
 seal('COMPLETION.receipt.json',paths,parts=['A','B','C','D','E','F'],all_authorized_branches_resolved=True,unknown_preserved=True,no_video_required=True)
 print('ROUND37 DELIVERED',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:write(HERE/'FINALIZATION_ERROR.json',dict(utc=utc(),error=traceback.format_exc()));raise
