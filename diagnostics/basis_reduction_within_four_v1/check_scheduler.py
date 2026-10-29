"""Isolated state-machine fixtures: no simulation and no DB registration."""
from pathlib import Path
import tempfile,shutil
import screening as s
import analysis_results as a
import combinations as c

def main():
 root=s.HERE;temp=Path(tempfile.mkdtemp(prefix='scheduler_contract_',dir=root/'tmp'))
 for name in ['COMBINATIONS.json','FAMILIES_64.json']:shutil.copyfile(root/name,temp/name)
 c.HERE=temp;data=c.read(temp/'COMBINATIONS.json');parents=[x for x in data['combinations']if x['dimension']==4];children=c.lower(3,parents,{x['code']:200 for x in parents})
 assert len(children)<=8 and len({x['signature']for x in children})==len(children)
 assert all(x['dimension']==3 for x in children)
 assert c.lower(3,parents,{x['code']:0 for x in parents})==children
 for child in children:
  for sp in child['candidate_provenance']:
   assert('Q'in child['active'])or sp['values']['Q']==0.
 # Exercise pause/restart, first-success stopping and64 exhaustion on8 fake
 # cases in a private fixture tree, without touching scientific artifacts.
 s.HERE=temp;s.controllers=lambda:[dict(type_code='stub')];s.selected_entries=lambda all_scenes=False:[dict(type_id=1,level=0,scene=dict(name='fixture'))];a.write_report=lambda:None
 family=dict(code='fixture',candidate_pool=[dict(k=k)for k in range(1,65)])
 calls=[];crashed=[False]
 def make(scene,ctl,si,sp,**kwargs):return dict(rollout_uid=f'fixture_{si}_{sp["k"]}',si=si,k=sp['k'])
 def request(cases,stage,**kwargs):
  if cases[0]['k']==3 and not crashed[0]:crashed[0]=True;raise RuntimeError('fixture interruption')
  calls.extend((x['si'],x['k'])for x in cases)
  return{x['rollout_uid']:dict(success=int(x['si']==0 and x['k']==2),numerical_failure=0)for x in cases}
 s.run_requests=request
 try:s.search(family,16,make)
 except RuntimeError as exc:assert str(exc)=='fixture interruption'
 result=s.search(family,64,make)
 assert result['complete64']and [g['searched']for g in result['groups']]==[2]+[64]*7
 assert len(calls)==450 and len(set(calls))==450
 before=len(calls);s.search(family,64,make);assert len(calls)==before
 s.dump(root/'SCHEDULER_CHECK.json',dict(passed=True,environment_steps=0,db_writes=0,fixture_directory=str(temp),checks=['deterministic bounded child dedup','binding removal projection','pending resume without duplicate query','first success stop2','64-only exhaustion','completed stage idempotence']))
 print('Scheduler and lower-level generation contract checks passed')

if __name__=='__main__':main()
