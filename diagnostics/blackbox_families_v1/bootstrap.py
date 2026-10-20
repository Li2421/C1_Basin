import json,hashlib,difflib,datetime
from pathlib import Path
R=Path('/home/zhihan/research/Basin_C1'); H=R/'diagnostics/blackbox_families_v1'; D=R/'diagnostics'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
copies={'core.py':'axis2_controller_v1/core.py','legacy_controllers.py':'other_controller_classes_v1/controllers.py','historical_controllers.py':'axis2_controller_v1/controllers.py','gaussian.py':'axis2_controller_v1/gaussian.py','legacy_runtime.py':'axis2_controller_v1/runtime.py','trigger.py':'blackbox_addendum_v1/trigger.py','progress.py':'axis4a_geometry_horizon_v1/progress.py'}
for dest,src in copies.items():
 p=D/src;q=H/dest;assert not q.exists();q.write_bytes(p.read_bytes())
(H/'COPIED_SOURCES.json').write_text(json.dumps({k:{'source':str(D/v),'sha256':sha(D/v),'copy_sha256':sha(H/k),'diff':[]} for k,v in copies.items()},indent=2)+'\n')
# Parse every mandatory historical JSON completely; summarize without dumping millions of bytes.
files=['blackbox_addendum_v1/RIGHT_HAND_GRID_RESULTS.json','blackbox_addendum_v1/RIGHT_HAND_SELECTED.json','blackbox_addendum_v1/right_hand_tune_DB_ROWS.json','axis2_controller_v1/GAUSSIAN_CONFIGS.json','other_controller_classes_v1/FROZEN_CHOICES.json','ring_controller_critic_v1/MODEL_FREEZE.receipt.json']
report={}
for s in files:
 p=D/s;v=json.loads(p.read_text());report[s]={'sha256':sha(p),'bytes':p.stat().st_size,'count':len(v),'type':type(v).__name__}
 if isinstance(v,list):report[s]['all_row_key_sets']=sorted(set(tuple(sorted(x)) for x in v if isinstance(x,dict)))
(H/'MANDATORY_JSON_READING.json').write_text(json.dumps(report,indent=2)+'\n')
