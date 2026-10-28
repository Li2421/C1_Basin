"""Isolated fake-result contract check; never registers or runs a rollout."""
import followup as f
import followup_reporting as report
from pathlib import Path
import tempfile

def main():
    root=f.HERE;temp=Path(tempfile.mkdtemp(prefix='scheduler_contract_',dir=root/'tmp'))
    original_report=report.write_report;report.write_report=lambda:None
    f.HERE=temp;f.controllers=lambda:[dict(type_code='stub')]
    f.dump(temp/'SCENES.json',dict(entries=[dict(type_id=1,level=0,scene=dict(name='fixture'))]))
    f.dump(temp/'HARD_CASES.json',dict(cases=[dict(type_id=1,level=0,controller='stub',state_index=i)for i in(0,1)],count=2))
    family=dict(scheme='B',candidate_pool=[dict(index=k)for k in range(1,65)])
    calls=[];crashed=[False]
    def make(scene,controller,si,sp,**kwargs):return dict(rollout_uid=f'fixture_{si}_{sp["index"]}',si=si,index=sp['index'])
    def execute(cases,stage):
        if cases[0]['index']==3 and not crashed[0]:crashed[0]=True;raise RuntimeError('simulated infrastructure interruption')
        calls.extend((c['si'],c['index'])for c in cases)
        return {c['rollout_uid']:dict(success=int(c['si']==0 and c['index']==2),numerical_failure=0)for c in cases}
    f.run_requests=execute
    try:f.search(family,'hard',make)
    except RuntimeError as exc:assert str(exc)=='simulated infrastructure interruption'
    result=f.search(family,'hard',make)
    assert result['complete']and len(calls)==66 and len(set(calls))==66
    assert [g['searched']for g in result['groups']]==[2,64]
    previous=len(calls);assert f.search(family,'hard',make)['complete']and len(calls)==previous
    assert f.qualifies(dict(groups=[dict(success=i<37)for i in range(41)]))
    assert not f.qualifies(dict(groups=[dict(success=i<36)for i in range(41)]))
    f.HERE=root;report.write_report=original_report
    f.dump(root/'FOLLOWUP_SCHEDULER_CHECK.json',dict(passed=True,environment_steps=0,db_writes=0,fixture_directory=str(temp),checks=['first success stops at2','uncovered stops at64','pending frontier survives interruption','no duplicate candidate after resume','completed state idempotent','exact90percent integer threshold37of41']))

if __name__=='__main__':main()
