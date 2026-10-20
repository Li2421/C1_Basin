from support import *
from metrics import features,signature_codes
G=guard()
def main():
 prereg();pr=read(HERE/'diagnostic_protocol.json');checks=[]
 for path in sorted((HERE/'diagnostic/raw').glob('*.json')):
  r=read(path);z=np.load(r['trace_path']);m=features(z['positions'],z['actions'],z['goals'],z['velocities']);p=z['positions'];v=z['velocities'];g=z['goals'];dist=np.linalg.norm(g-p,axis=-1);angles=np.unwrap(np.arctan2(p[...,1],p[...,0]),axis=0);_,mode=signature_codes(angles-angles[0]);freeze=((np.linalg.norm(v[1:],axis=-1)<.01)&(dist[1:]>.09)).sum()/1200
  cc=dict(complete_300=m['T']==300,no_task_labels=not r['terminal_labels_read'],initial_state_exact=np.array_equal(p[0],np.asarray(pr['states'][r['case']['state_index']]['physical']['positions'])),progress_parity=all(abs(m['early'][str(t)]['progress']-r['features'][f'progress{t}'])<1e-12 for t in [100,200,300]),freeze_parity=abs(freeze-r['features']['freeze_proportion'])<1e-12,mixed_parity=abs(np.mean(mode[1:]==3)-r['features']['mixed_proportion'])<1e-12,round30g_moving_sensitivity_parity=abs(m['moving_no_progress_robot_seconds']-r['unfinished_moving_no_progress_seconds'])<1e-10,intervention_parity=abs(np.mean(np.linalg.norm(z['actions']-z['nominal'],axis=-1))-r['features']['intervention'])<1e-12,kinematic_identity=m['kinematic_max_abs_error']<1e-10)
  checks.append(dict(probe=r['case']['rollout_uid'],checks=cc,passed=all(cc.values())))
 assert len(checks)==736;write(HERE/'DIAGNOSTIC_INDEPENDENT_AUDIT.json',dict(passed=all(r['passed'] for r in checks),probes=len(checks),round30g_metric_source=sha(HERE/'metrics.py'),results=checks,guard=guard()));source_inventory('imports_diagnostic_audit.json');assert all(r['passed'] for r in checks);print('Independent stored-evidence audit passed',len(checks),flush=True)
if __name__=='__main__':main()
