from common import *
from progress import ProgressSwitch,offline_switch

def main():
 decision=read(HERE/'PSTAR_DECISION.receipt.json');assert decision['authorized_rollouts'];threshold=decision['pstar'];checks=[]
 for H in [500,700,1000]:
  g=np.ones((4,2))*10;p0=np.zeros((4,2));path=np.asarray([p0+(g-p0)*min(t/H,1) for t in range(H+1)]);online=ProgressSwitch(path[0],g,H,threshold)
  for t in range(H):online.eta_index(t,path[t],g)
  checks.append(dict(H=H,online=online.switched_at,offline=offline_switch(path,g,H,threshold)));assert online.switched_at==checks[-1]['offline']
 # Existing F13 trajectories: exact original action-index semantics, no invented terminal action.
 refs=read(HERE/'SOURCE_CORRECTIONS.json');names=read(HERE/'HISTORICAL_INVENTORY.json')['controller_names'];sid=read(HERE/'EVAL_STATES_ORIGINAL.json')[0]['state_uid']
 for name in names:
  ref=next(q for q in refs if q['case']['base_controller_name']==name and q['case']['state_uid']==sid and q['case']['arm']=='F13' and q['case']['seed']==0)
  with np.load(ref['trace_path']) as z:
   pos=z['positions'];g=z['goals']
   for H in [500,700,1000]:
    online=ProgressSwitch(pos[0],g,H,threshold)
    for t in range(min(H,len(pos)-1)):online.eta_index(t,pos[t],g)
    off=offline_switch(pos,g,H,threshold);assert online.switched_at==off;checks.append(dict(controller=name,H=H,online=online.switched_at,offline=off,source_trace=ref['trace_path']))
 write(HERE/'SELECTED_PROGRESS_TEST.json',dict(utc=utc(),passed=True,pstar=threshold,checks=checks,physical_steps=0));seal('SELECTED_PROGRESS_TEST.receipt.json',[HERE/'PSTAR_DECISION.receipt.json',HERE/'progress.py',HERE/'test_selected_progress.py',HERE/'SELECTED_PROGRESS_TEST.json'],before_B_submission=True);print('Selectedpstar online/offline tests PASS',flush=True)
if __name__=='__main__':main()
