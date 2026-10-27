"""Run only the selected, all-fold-frozen main-method generated candidate."""
from common import *
from requests_corrected import request,verify_freezes
import subprocess

def prepare():
 verify_freezes();known={etahex(a['eta_a']):i for i,a in enumerate(actions())};cases={};refs=[]
 for fold in read(HERE/'FOLDS.json'):
  for name,row in read(HERE/'folds'/fold['name']/'CHOICES.json').items():
   choice=row['methods']['v2'];a=choice['selected_index'];assert a is not None;eta=row['candidate_etas'][a]
   for ctl in row['controllers']:
    for state in row['state_indices']:
     ref=dict(fold=fold['name'],scene=name,controller=ctl,state_index=state,method='v2',chosen_index=a,eta=eta)
     if a<49 or etahex(eta)in known:ref.update(source='existing49_label',historical_action=known[etahex(eta)])
     else:
      c=request(name,ctl,state,eta);cases[c['rollout_uid']]=c;ref.update(source='generated_eta_rollout',rollout_uid=c['rollout_uid'])
     refs.append(ref)
 new=list(cases.values());prior=sum(len(read(HERE/f'{s}_MISSING_CASES.json'))for s in ['capability','main']);assert prior+len(new)<=20000
 write(HERE/'proposal_REFERENCES.json',refs);write(HERE/'FINAL_ROLLOUT_BUDGET.json',dict(stage1_new=prior,selected_proposals=len(new),new_total=prior+len(new),limit=20000,passed=True,all_folds_frozen_before_registration=True))
 register('proposal',new);subprocess.run([sys.executable,'-B','-m','shared_rollout_db.plan','--manifest',str(HERE/'proposal_PLANNED_ROLLOUTS.json'),'--output',str(HERE/'proposal_OFFICIAL_PREFLIGHT.json')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
 from queueing import build
 build('proposal');print('Selected proposal requests:',len(new),flush=True)
if __name__=='__main__':prepare()
