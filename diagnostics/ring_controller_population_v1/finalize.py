from core import *
import subprocess
from collections import defaultdict
def main():
 labels_allowed();waves=0
 for k in range(1,5):
  f=HERE/f'wave{k}_VERIFY.json'
  if not f.exists():break
  assert read(f)['complete'];waves=k
 assert waves>=1
 py='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
 subprocess.run([py,'-B',str(HERE/'merge.py')],cwd=ROOT,check=True)
 groups=defaultdict(list)
 for k in range(1,waves+1):
  for c in read(HERE/f'wave{k}_MISSING_CASES.json'):groups[c['state_uid'],c['eta_uid'],c['controller_uid']].append(c['seed_key'])
 manifest=HERE/'completed_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=keys) for (s,e,c),keys in groups.items()]))
 subprocess.run([py,'-m','shared_rollout_db.plan','--manifest',str(manifest),'--output',str(HERE/'completed_POSTFLIGHT.json')],cwd=ROOT,check=True)
 subprocess.run([py,'-m','shared_rollout_db.plan','--manifest',str(HERE/'main_PLANNED_ROLLOUTS.json'),'--output',str(HERE/'full_POSTFLIGHT.json')],cwd=ROOT,check=True)
 post=read(HERE/'completed_POSTFLIGHT.json');assert post['summary']['ambiguous']==0,post['summary']
 subprocess.run([py,'-B',str(HERE/'postflight_unknown_audit.py')],cwd=ROOT,check=True)
 unknown=read(HERE/'POSTFLIGHT_UNKNOWN_AUDIT.json');assert unknown['status']=='PASS' and unknown['true_absent']==0
 write(HERE/'POSTFLIGHT_AUDIT.json',dict(utc=utc(),complete_waves=waves,completed_postflight_sha256=sha(HERE/'completed_POSTFLIGHT.json'),full_postflight_sha256=sha(HERE/'full_POSTFLIGHT.json'),unknown_audit_sha256=sha(HERE/'POSTFLIGHT_UNKNOWN_AUDIT.json'),summary=post['summary'],true_absent=0,only_complete_waves_analyzed=True))
 subprocess.run([py,'-B',str(HERE/'analyze.py')],cwd=ROOT,check=True)
 print('Round15 finalized',waves)
if __name__=='__main__':main()
