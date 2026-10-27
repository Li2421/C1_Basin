"""Fixed capability gate, one prescribed mild fallback forB, missing-only main labels."""
from common import *
from design import *
from scenes import make_new
import numpy as np,subprocess

def read_result(stage,c):
 for name in [stage,'capability','capability_mild','main']:
  p=HERE/name/'raw'/f"{c['rollout_uid']}.json"
  if p.exists():return read(p)
 with dbm.connect(True)as db:rows=db.execute('SELECT sf.path,rs.source_line FROM rollout_source rs JOIN source_file sf USING(source_uid)WHERE rs.rollout_uid=?',(c['rollout_uid'],)).fetchall()
 for row in rows:
  p=Path(row['path'])
  if not p.exists():continue
  with p.open()as f:r=next((json.loads(l)for i,l in enumerate(f,1)if i==row['source_line']),None)
  if r and r.get('rollout_uid')==c['rollout_uid']:evidence(p);return r
 raise FileNotFoundError(c['rollout_uid'])

def capability_cases(mild=False):
 scenes=read(HERE/'SCENES_INITIAL.json')
 if mild:
  failed=read(HERE/'CAPABILITY_INITIAL.json')['mild_required'];scenes=[s for s in read(HERE/'SCENE_LIBRARY.json')if s['name']in failed]
 out=[]
 for scene in scenes:
  for ctl in controllers():
   for state in scene['states'][:4]:
    for i in range(state['N']):
     solo=dict(state,N=1,parent_N=state['N'],agent=i,parent_positions=state['positions'],positions=[state['positions'][i]],goals=[state['goals'][i]],velocities=[state['velocities'][i]],index=state['index']*100+i)
     for seed in [0,1]:out.append(make_new(scene,ctl,solo,menu()[0],seed))
 return out

def capability_table(stage):
 by={s['name']:s for s in read(HERE/'SCENE_LIBRARY.json')};table={}
 for c in read(HERE/f'{stage}_CASES.json'):
  n=c['state']['scene'];ctl=c['controller_payload']['nominal']['controller'];r=read_result(stage,c);v=table.setdefault(n,{}).setdefault(ctl,dict(total=0,success=0,unknown=0,collision=0,timeout=0));v['total']+=1
  for k in ['success','collision','timeout']:v[k]+=int(r[k])
  v['unknown']+=int(r['numerical_failure'])
 for n,rows in table.items():
  for v in rows.values():
   assert v['total']==8*by[n]['config']['num_agents'];v['rate']=v['success']/v['total'];v['upper_rate']=(v['success']+v['unknown'])/v['total'];v['passed']=v['rate']>=.95
 return table

def capability_select(final=False):
 table=capability_table('capability');by={s['name']:s for s in read(HERE/'SCENE_LIBRARY.json')};geometry=read(HERE/'GEOMETRY_BASE_REPORTS.json');mild_required=[]
 for n,rows in table.items():
  if by[n]['group']=='B'and not all(v['passed']for v in rows.values()):mild_required.append(n+'_mild')
 if not final:
  write(HERE/'CAPABILITY_INITIAL.json',dict(per_scene=table,mild_required=mild_required));return mild_required
 if(HERE/'capability_mild_CASES.json').exists():table.update(capability_table('capability_mild'))
 selected={};active=[]
 for original in read(HERE/'SCENES_INITIAL.json'):
  base=original['name'];n=base+'_mild'if base+'_mild'in mild_required else base;scene=by[n];rows=table[n];passed=all(v['passed']for v in rows.values());valid=all(v['eligible']for v in geometry[n]['states']);applicable=passed and valid
  selected[base]=dict(base_name=base,active_name=n,group=scene['group'],test_applicable=applicable,all_three_passed=passed,geometry_all_states_eligible=valid,status='可测试'if applicable else'控制器不适用'if not passed else'几何或路径缺口',mild_used=n!=base)
  if applicable:active.append(scene)
 write(HERE/'SCENES.json',active);write(HERE/'CAPABILITY_TABLE.json',dict(per_scene=table,scene_status=selected,threshold=.95,states=[0,1,2,3],seeds=[0,1],prespecified_B_mild_only=True));write(HERE/'SELECTED_CONTROLLERS.json',selected);return active

def main_cases():
 out=[]
 for scene in read(HERE/'SCENES.json'):
  for ctl in controllers():
   for state in scene['states']:
    for action in menu():out.append(make_new(scene,ctl,state,action,0))
 return out

def prepare(stage):
 assert read(HERE/'REPLAY_CHECK.json')['passed'];cases=capability_cases(stage=='capability_mild')if stage.startswith('capability')else main_cases()
 if stage=='main'and len(cases)>32000:
  write(HERE/'STOP.receipt.json',dict(reason='MAIN_ROLLOUT_BUDGET_EXCEEDED',count=len(cases),limit=32000));raise RuntimeError('ROLLOUT_BUDGET_EXCEEDED')
 write(HERE/f'{stage}_BUDGET.json',dict(count=len(cases),main_limit=32000,capability_excluded_from_main_limit=True));register(stage,cases)
 subprocess.run([sys.executable,'-B','-m','shared_rollout_db.plan','--manifest',str(HERE/f'{stage}_PLANNED_ROLLOUTS.json'),'--output',str(HERE/f'{stage}_OFFICIAL_PREFLIGHT.json')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
 from queueing import build
 build(stage)
