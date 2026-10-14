from core import *
from collections import Counter
def main():
 labels_allowed();base=read(HERE/'main_CACHE_PREFLIGHT.json');assert base['summary']['ambiguous']==0
 profiles=read(HERE/'FINAL_BASE_PROFILES.json')['profiles'];allrows=read(HERE/'main_MISSING_CASES.json');counts={}
 for k in range(1,5):
  rows=[c for c in allrows if profiles[c['base_controller_name']]['wave']==k]
  assert len(rows)<=14336
  write(HERE/f'wave{k}_MISSING_CASES.json',rows)
  write(HERE/f'wave{k}_CACHE_PREFLIGHT.json',dict(summary=dict(ambiguous=0,genuinely_missing=len(rows)),parent_manifest_sha256=sha(HERE/'main_PLANNED_ROLLOUTS.json'),parent_preflight_sha256=sha(HERE/'main_CACHE_PREFLIGHT.json'),wave=k))
  counts[k]=len(rows)
 write(HERE/'WAVE_PLAN.json',dict(utc=utc(),planned_per_wave=14336,genuinely_missing=counts,complete_wave_required=True,start_after_0730_forbidden=True,all_rollouts_before_1000=True))
 print(counts)
if __name__=='__main__':main()
