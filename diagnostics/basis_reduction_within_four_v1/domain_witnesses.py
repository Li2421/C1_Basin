"""Analyze an already executed matched-strength pair; no new rollout."""
import guard
from guard import HERE
import json,hashlib
from pathlib import Path
import numpy as np

def read(name):return json.loads((HERE/name).read_text())

def main():
 key=[16,0,'centralized_sampling',3]
 reference=next(g for g in read('SEARCH_reference_screen.json')['groups']if g['identity']==key)
 bound=next(g for g in read('SEARCH_four_2_screen.json')['groups']if g['identity']==key)
 a=reference['queries'][56]['result'];b=bound['solution']['result']
 assert reference['queries'][56]['index']==bound['solution']['index']==57
 assert a['spec']['eta']==b['spec']['eta']and a['spec']['coefficients']==b['spec']['coefficients']
 assert a['scenario_uid']==b['scenario_uid']and a['state_uid']==b['state_uid']and a['seed']==b['seed']
 for name in ['R_e','R_c','rho','K']:assert a['spec']['one_way'][name]==b['spec']['one_way'][name]
 with np.load(a['trace'])as old,np.load(b['trace'])as new:
  initial={name:old[name][0].tobytes()==new[name][0].tobytes()for name in ['positions','velocities']}
  initial['goals']=old['goals'].tobytes()==new['goals'].tobytes()
  n=min(len(old['actions']),len(new['actions']));different=[i for i in range(n)if old['actions'][i].tobytes()!=new['actions'][i].tobytes()];first=different[0]if different else None
  before={name:first is not None and old[name][:first+1].tobytes()==new[name][:first+1].tobytes()for name in ['positions','nominal','safe']}
  role_counts={}
  for label,trace in [('reference',old),('bound',new)]:
   stored=trace['admission_json'];frames=json.loads(str(stored.item()))if stored.ndim==0 else[json.loads(str(value))for value in stored]
   role_counts[label]=dict(goal_yield_active_steps=sum(any(any(g.get('goal_yield',[]))for g in frame)for frame in frames),recorded_steps=len(frames))
 assert all(initial.values())and all(before.values())
 out=dict(identity=key,candidate=57,same_original_coefficients=True,same_initial_state_and_seed=True,initial_bitwise_checks=initial,first_different_action_step=first,bitwise_checks_through_first_different_action_input=before,role_counts=role_counts,
  reference=dict(rollout_uid=a['rollout_uid'],q=a['spec']['one_way']['c1'],outcome=a['outcome'],completion_time=a['completion_time'],trace=a['trace']),
  bound=dict(rollout_uid=b['rollout_uid'],q=b['spec']['one_way']['c1'],outcome=b['outcome'],completion_time=b['completion_time'],trace=b['trace']),
  unchanged_coefficients=dict(eta=a['spec']['eta'],coefficients=a['spec']['coefficients']),
  interpretation='This executed pair changes only the common admission strength. The successful strength is below the reference lower bound0.5. It is evidence of parameter-range sensitivity, not an isolated proof of dimension reduction or a new geometric direction.',new_rollouts=0)
 (HERE/'DOMAIN_WITNESSES.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))

if __name__=='__main__':main()
