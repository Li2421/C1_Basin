from support import *
from metrics import features,signature_codes
G=guard()
def main():
 p=read(HERE/'diagnostic_protocol.json');checks=[]
 for path in sorted((HERE/'diagnostic/raw').glob('*.json')):
  r=read(path);z=np.load(r['trace_path']);m=features(z['positions'],z['actions'],z['goals'],z['velocities']);pos=z['positions'];d=np.linalg.norm(z['goals']-pos,axis=-1);v=z['velocities'];ang=np.unwrap(np.arctan2(pos[...,1],pos[...,0]),axis=0);_,mode=signature_codes(ang-ang[0]);frozen=((np.linalg.norm(v[1:],axis=-1)<.01)&(d[1:]>.09)).sum()/1200;cc=dict(complete=m['T']==300,no_labels=not r['terminal_labels_read'],initial_exact=np.array_equal(pos[0],np.asarray(p['states'][r['case']['state_index']]['physical']['positions'])),progress=all(abs(m['early'][str(t)]['progress']-r['features'][f'progress{t}'])<1e-12 for t in [100,200,300]),freeze=abs(frozen-r['features']['freeze_proportion'])<1e-12,mixed=abs(np.mean(mode[1:]==3)-r['features']['mixed_proportion'])<1e-12,moving_sensitivity=abs(m['moving_no_progress_robot_seconds']-r['unfinished_moving_no_progress_seconds'])<1e-10,intervention=abs(np.mean(np.linalg.norm(z['actions']-z['nominal'],axis=-1))-r['features']['intervention'])<1e-12,kinematics=m['kinematic_max_abs_error']<1e-10);checks.append(dict(probe=r['case']['rollout_uid'],checks=cc,passed=all(cc.values())))
 assert len(checks)==928;write(HERE/'DIAGNOSTIC_INDEPENDENT_AUDIT.json',dict(passed=all(q['passed'] for q in checks),probes=len(checks),results=checks,guard=guard()));source_inventory('imports_diagnostic_independent_audit.json');assert all(q['passed'] for q in checks);print('928 stored diagnostic traces independently verified',flush=True)
if __name__=='__main__':main()
