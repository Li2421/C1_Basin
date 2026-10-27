"""Reuse exact historical gates; label only prespecified missing physical states."""
from common import *
from design import *
from scenes import make_new
import numpy as np,subprocess

@lru_cache(None)
def old_cap_cases():
 from replay import iter_array
 path=OLD/'capability_CASES.json';evidence(path);out=[]
 for c in iter_array(path):
  if c['state']['scene']in('xsc_swap_n6','xsc_swap_n10')and c['controller_payload']['nominal']['controller']in CONTROL_IDS and c['state']['index']//100<4:out.append(c)
 assert len(out)==384;return out

def read_result(stage,c):
 for base in [HERE,OLD]:
  for name in [stage,'capability','main','pilot','pilot_tail']:
   p=base/name/'raw'/f"{c['rollout_uid']}.json"
   if p.exists():
    if base!=HERE:evidence(p)
    return read(p)
 if c.get('reuse_source'):
  p=Path(c['reuse_source']);evidence(p);return read(p)
 with dbm.connect(True)as db:rows=db.execute('SELECT sf.path,rs.source_line FROM rollout_source rs JOIN source_file sf USING(source_uid) WHERE rs.rollout_uid=?',(c['rollout_uid'],)).fetchall()
 for row in rows:
  p=Path(row['path'])
  if not p.exists():continue
  with p.open()as f:r=next((json.loads(l)for i,l in enumerate(f,1)if i==row['source_line']),None)
  if r and r.get('rollout_uid')==c['rollout_uid']:evidence(p);return r
 raise FileNotFoundError(c['rollout_uid'])

def capability_cases():
 out=copy.deepcopy(old_cap_cases())
 for scene in read(HERE/'SCENES.json'):
  if scene['name']not in NEW_NAMES:continue
  for ctl in controllers():
   for state in scene['states'][:4]:
    for i in range(state['N']):
     solo=dict(state,N=1,parent_N=state['N'],agent=i,parent_positions=state['positions'],positions=[state['positions'][i]],goals=[state['goals'][i]],velocities=[state['velocities'][i]],index=state['index']*100+i)
     for seed in(0,1):out.append(make_new(scene,ctl,solo,menu()[0],seed))
 assert len(out)==816;return out

def capability_select():
 table={n:{ctl:dict(total=0,success=0,unknown=0,collision=0,timeout=0)for ctl in CONTROL_IDS}for n in SCENE_NAMES if not n.startswith('lit_')}
 for c in read(HERE/'capability_CASES.json'):
  r=read_result('capability',c);row=table[c['state']['scene']][c['controller_payload']['nominal']['controller']];row['total']+=1
  for k in('success','collision','timeout'):row[k]+=int(r[k])
  row['unknown']+=int(r['numerical_failure'])
 selected={};geometry=read(HERE/'GEOMETRY_BASE_REPORTS.json');by={s['name']:s for s in read(HERE/'SCENES.json')}
 original_selected=read_evidence(OLD/'SELECTED_CONTROLLERS.json')
 for n in SCENE_NAMES:
  valid=all(s['eligible']for s in geometry[n]['states'])
  if n.startswith('lit_'):
   original=original_selected[n];assert all(any(v and v['controller']==ctl for v in original.values())for ctl in CONTROL_IDS)
   selected[n]=dict(test_applicable=valid,all_three_passed=True,geometry_all_states_eligible=valid,status='可测试'if valid else'几何或路径缺口',gate_source='Historical qualified controllers and exact existing labels; no new literature labels.')
  else:
   for row in table[n].values():
    assert row['total']==8*by[n]['config']['num_agents'];row['rate']=row['success']/row['total'];row['upper_rate']=(row['success']+row['unknown'])/row['total'];row['passed']=row['rate']>=.95
   passed=all(r['passed']for r in table[n].values());selected[n]=dict(test_applicable=passed and valid,all_three_passed=passed,geometry_all_states_eligible=valid,status='可测试'if passed and valid else'控制器不适用'if not passed else'几何或路径缺口',gate_source='Exact historical first4 solo cases reused'if n not in NEW_NAMES else'New first4 solo states, seeds0/1')
 write(HERE/'CAPABILITY_TABLE.json',dict(per_scene=table,scene_status=selected,threshold=.95,states=[0,1,2,3],seeds=[0,1],historical_reused=384,new_runs=432));write(HERE/'SELECTED_CONTROLLERS.json',selected);return selected

def main_cases():
 chosen=read(HERE/'SELECTED_CONTROLLERS.json');out=[]
 for scene in read(HERE/'SCENES.json'):
  n=scene['name']
  if n.startswith('lit_')or not chosen[n]['test_applicable']:continue
  states=scene['states']if n in NEW_NAMES else scene['states'][12:]
  for ctl in controllers():
   for state in states:
    for action in menu():out.append(make_new(scene,ctl,state,action,0))
 return out

def prepare(stage):
 assert read(HERE/'REPLAY_CHECK.json')['passed'];cases=capability_cases()if stage=='capability'else main_cases()
 budget=432+15288+384
 if budget>20000:raise RuntimeError('NEW_ROLLOUT_BUDGET_EXCEEDED')
 write(HERE/'ROLLOUT_BUDGET.json',dict(capability_new_max=432,main_new_max=15288,proposal_new_max=384,total_max=budget,limit=20000,passed=True,historical_capability384_not_rerun=True))
 register(stage,cases)
 subprocess.run([sys.executable,'-B','-m','shared_rollout_db.plan','--manifest',str(HERE/f'{stage}_PLANNED_ROLLOUTS.json'),'--output',str(HERE/f'{stage}_OFFICIAL_PREFLIGHT.json')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
 from queueing import build
 build(stage)
