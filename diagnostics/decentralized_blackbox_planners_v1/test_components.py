from runtime_new import *
from audit_utils import source_inventory
G=guard()
def main():
 p=read(HERE/'CONTROLLERS.json');checks={};s=read(HERE/'EVAL_STATES.json')[0]
 for i in range(6):
  rt=make(f'AS{i}',p,{});env=rt.make_env(s);pl=rt.planner;obstacles=np.empty((0,2));path=pl.plan(env.positions[0],env.goals[0],obstacles)
  checks[f'path_{i}']=len(path)>1 and all(pl.clear(a,b,obstacles) for a,b in zip(path[:-1],path[1:]));checks[f'deterministic_{i}']=np.array_equal(path,pl.plan(env.positions[0],env.goals[0],obstacles))
  from astar import track
  checks[f'speed_{i}']=all(np.linalg.norm(track(q,env.goals[0],path,.52))<=.52+1e-14 for q in path)
  a=rt.nominal(env,None)[1];checks[f'idempotent_{i}']=np.array_equal(a,rt.nominal(env,None)[1])
  # New independent runtime; change only other goals and neighbors outside sensing.
  rt2=make(f'AS{i}',p,{});env2=rt2.make_env(s);env2.goals[1:]+=100;far=np.linalg.norm(env.positions[1:]-env.positions[0],axis=1)>1.5;env2.positions[1:][far]+=100
  checks[f'private_information_{i}']=np.array_equal(a[0],rt2.nominal(env2,None)[1][0])
 # Neighbor-disc avoidance on otherwise clear annulus.
 rt=make('AS1',p,{});q=np.array([[1.7,0.]]);path=rt.planner.plan([1.7,-1.],[1.7,1.],q);checks['neighbor_disc_path']=len(path)>1 and all(rt.planner.clear(a,b,q) for a,b in zip(path[:-1],path[1:]))
 checks['RHR_24']=len(read(HERE/'RHR_GRID.json'))==24
 for i in range(6):
  v=p['profiles'][f'G{i}']['payload'];checks[f'checkpoint_G{i}']=sha(v['flow_checkpoint_path'])==v['flow_checkpoint_sha256']
 guard();write(HERE/'UNIT_TESTS.json',dict(passed=all(checks.values()),checks=checks,guard=guard()));source_inventory('imports_tests.json');assert all(checks.values()),checks
if __name__=='__main__':main()
