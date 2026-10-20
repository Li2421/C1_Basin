from common import *
from runtime_new import make,fair
from student import Student
from data import teacher
from audit_utils import source_inventory
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
from pipeline import decentralized_action
from dataclasses import replace
import jax
G=guard()
ARMS=['safety baseline','F13','S500','RHR-A','RHR-B','RHR-C']
FAMILIES=['ST','TD','G','DM','AS']
FULL={'ST':'原离线蒸馏各自噪声分布式学生','TD':'共享观测联合MACFlow教师（分布式安全层）','G':'共享观测高斯行为克隆策略','DM':'分布式模型预测控制','AS':'A*规划controller'}
RULES={'RHR-A':dict(mode='progress',steps=20,speed=.35),'RHR-B':dict(mode='neighbor_right',steps=0,speed=.1,distance=.8),'RHR-C':dict(mode='progress',steps=40,speed=.25)}
def runtime(name,p,cache):
 if name not in cache:
  if name.startswith('ST'):cache[name]=Student(int(name[2:]))
  elif name.startswith('TD'):cache[name]=teacher(int(name[2:]))
 return make(name,p,cache)
def setup():
 e=read(D/'decentralized_blackbox_planners_v1/CONTROLLERS.json');t=read(HERE/'TEACHERS.json');students=read(HERE/'STUDENTS.json');base=read(D/'decentralized_pipeline_v1/main_protocol.json')['base_profiles'];p=dict(e)
 p['controller_order']=[f'ST{i:02}' for i in range(14)]+[f'TD{i:02}' for i in range(14)]+[f'{f}{i}' for f in ['G','DM','AS'] for i in range(6)]
 for f in ['ST','TD']:
  for i,r in enumerate(t):
   v=dict(base[r['name']]['payload']);v.update(controller_class=f,source_index=i)
   if f=='ST':v.update(student_checkpoint=students[str(i)],flow_checkpoint_sha256=sha(students[str(i)]['checkpoint']))
   p['profiles'][f'{f}{i:02}']=dict(payload=v)
 p.update(states=read(HERE/'EVAL_STATES.json'),experiment_uid=EXP+'_supplement',eta=read(D/'decentralized_pipeline_v1/main_protocol.json')['eta'])
 write(HERE/'CONTROLLERS.json',p)
 write(HERE/'PREREGISTRATION.receipt.json',dict(utc=utc(),sha256=sha(HERE/'PREREGISTRATION.md'),prompt_sha256=sha(ROOT/'docs/prompt_round30h_decentralized_fix_selection_20261017.md'),before_new_simulation=True,guard=guard()))
 source_inventory('imports_setup.json')
if __name__=='__main__':setup()
