"""Assemble immutable historic and new labels without retraining or rerunning old cells."""
from common import *
from design import *
from stages import read_result
import numpy as np

def assemble():
 assert(HERE/'STAGE1_LABELS_COMPLETE.json').exists();out=read(HERE/'OLD_LABEL_TABLES.json');selected=read(HERE/'SELECTED_CONTROLLERS.json');lookup={}
 for c in read(HERE/'main_CASES.json'):lookup[(c['state']['scene'],c['controller_payload']['nominal']['controller'],c['state']['index'],c['requested_action'])]=c
 for scene in read(HERE/'SCENES.json'):
  n=scene['name']
  if not selected[n]['test_applicable']:out.pop(n,None);continue
  if n.startswith('lit_'):continue
  count=len(scene['states']);y=np.zeros((3,count,49));u=np.zeros_like(y);old=out.get(n);start=12 if old else 0
  if old:y[:,:12]=old['y'];u[:,:12]=old['unknown']
  ids=[]
  for ci,ctl in enumerate(CONTROL_IDS):
   for si in range(start,count):
    for ai,key in enumerate(ETA_KEYS):
     c=lookup[(n,ctl,si,key)];r=read_result('main',c);y[ci,si,ai]=r['success'];u[ci,si,ai]=r['numerical_failure'];ids.append(c['rollout_uid'])
  out[n]=dict(controllers=list(CONTROL_IDS),state_indices=list(range(count)),eta_keys=ETA_KEYS,y=y.tolist(),unknown=u.tolist(),new_rollout_uids=ids,historical_prefix_states=start)
 evals={n:list(range(8,24))if n in NEW_NAMES else list(range(12,28))if n.startswith('xsc_swap_')else list(out[n]['state_indices'])for n in out}
 write(HERE/'LABEL_TABLES.json',out);write(HERE/'EVALUATION_STATES.json',dict(scenes=evals,literature_states_previously_used_in_scene_selection=True))
 exploration={}
 for n in NEW_NAMES:
  if n not in out:continue
  y=np.asarray(out[n]['y'])[:,:8];q=y.mean((0,1));exploration[n]=dict(best_rate=float(q.max()),best_candidates=[menu()[i]for i in np.flatnonzero(q==q.max())],exploration_expected_report_only=True)
 write(HERE/'EXPLORATION_RECONSTRUCTION_CHECK.json',exploration);return out
if __name__=='__main__':assemble()
