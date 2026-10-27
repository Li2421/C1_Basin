"""The user-stopped old batch is a closed logical scope, independent of later exact eta0 identities."""
from experiment import *
from registry import verify_receipt
OLD_STAGE='learning_labels_lit_circle_five'

def preserved_scope():
 verify_receipt('OLD_CIRCLE_LEARNING_STOPPED_20261021.receipt.json')
 rows=read(HERE/'OLD_CIRCLE_LEARNING_LANDED_20261021.json')['rows'];index={r['rollout_uid']:r for r in rows}
 assert len(rows)==len(index)==2044
 actual={p.stem for p in (HERE/OLD_STAGE/'raw').glob('*.json')};assert actual==set(index),'Old stopped raw scope changed'
 return index

def complete_original_landed_label_diagnoses():
 from correction_after_distributed_fixed_20261021_quota18_now_20261022 import analyze
 from result_inventory import find_raw
 from statistics_correction_revised_20261021_quota18_now_20261022 import save_or_assert
 originals=[];coverage=[];closed=preserved_scope()
 for scene in NAMES:
  original='learning_labels_'+scene;stage='landed_original_labels_20261021_'+scene;planned=read(HERE/f'{original}_CASES.json');landed=[]
  for c in planned:
   if original==OLD_STAGE and c['rollout_uid'] not in closed:continue
   path=Path(closed[c['rollout_uid']]['raw']) if original==OLD_STAGE else find_raw(c['rollout_uid'],original,scene)
   assert path.exists();landed.append(c)
  if original==OLD_STAGE:assert len(landed)==2044 and len(planned)-len(landed)==4100
  save_or_assert(stage+'_CASES.json',landed);originals.append(stage);coverage.append(dict(scene=scene,planned=len(planned),landed=len(landed),unwritten_not_resumed=len(planned)-len(landed),closed_at_user_stop=original==OLD_STAGE))
 analyze(originals)
 write(HERE/'ORIGINAL_LANDED_LABEL_DIAGNOSIS_COVERAGE_20261021.json',dict(utc=utc(),coverage=coverage,new_rollouts=0,unwritten_not_unknown=True,scope='Only original landed evidence; old circle strictly frozen2044 stop manifest. Later authorized revised eta0 identity completion does not expand stopped old logical scope.'))
