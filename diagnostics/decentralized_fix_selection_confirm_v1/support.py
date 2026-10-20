from common import *
from runtime_new import make,fair
from student import Student
from data import teacher
from audit_utils import source_inventory
from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
from pipeline import decentralized_action,LocalSafety,local_basis
from dataclasses import replace
import jax
G=guard()
ARMS=['safety baseline','F13','S500','RHR-A','RHR-B','RHR-C']
FAMILIES=['ST','TD','G','DM','AS']
FULL={'ST':'闭环重标注及共享随机种子分布式学生','TD':'独立训练种子联合MACFlow教师（分布式安全层）','G':'新配方共享观测高斯行为克隆策略','DM':'新参数分布式模型预测控制','AS':'新设置A*规划controller'}
RULES={'RHR-A':dict(mode='progress',steps=20,speed=.35),'RHR-B':dict(mode='neighbor_right',steps=0,speed=.1,distance=.8),'RHR-C':dict(mode='progress',steps=40,speed=.25)}
FEATURES=['freeze_proportion','moving_no_progress_robot_seconds','progress100','progress200','progress300','intervention','mixed_proportion']
GRID_A=[0,.01,.025,.05,.1,.2,.35,.5,.75,1]
GRID_B=[-20,-10,-5,0,2,4,6,8,10,12,16,20]
def runtime(name,p,cache):
 if name not in cache:
  v=p['profiles'][name]['payload'];kind=v.get('controller_class')
  if kind=='ST':
   assert sha(v['flow_checkpoint_path'])==v['flow_checkpoint_sha256'];cache[name]=Student(v['source_index'],v['variant'],v['flow_checkpoint_path'])
  elif kind=='TD':
   assert sha(v['flow_checkpoint_path'])==v['flow_checkpoint_sha256'];cache[name]=RingFlowAdapter(v['flow_checkpoint_path'],read(v['adapter_manifest_path']),v['observation_frame'])
 return make(name,p,cache)
def source_score(choice,rate,families):return float(np.mean([rate[np.arange(len(choice)),choice][families==f].mean() for f in sorted(set(families))]))
def select(x,rule,secondary=False):
 rh=np.where(x[:,0]>rule['C'],3,4) if secondary else ARMS.index(rule['right_hand_arm'])
 return np.where(x[:,0]>rule['A'],rh,np.where(x[:,4]>rule['B'],0,2))
