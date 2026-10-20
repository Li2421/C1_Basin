"""Exact preflight of genuine new 700-step unlabeled behavior probes."""
from common import *
from actions_c6 import action,case
from registry import register
from requery_client import FrozenMean

def main():
    prereg();assert read(HERE/'EXECUTION_EQUIVALENCE.json')['passed'];assert (HERE/'OPENING_RECORD.receipt.json').exists()
    base=read(HERE/'BASE_PROTOCOL.json');source=read(D/'crosstype2_mixed_zoo_selector_v1/PROBE_MANIFEST.json');states=source['states'];assert len(states)==16
    ref=FrozenMean();means=[]
    for s in states:means.append(ref.query(s['state_uid'],s['physical']))
    ref.close();write(HERE/'PROBE_FROZEN_MEANS.json',means);rows=[];profiles={}
    for name in base['controller_order']:
        for si,s in enumerate(states):
            for arm,eta in [('safety',[0.,0.,0.]),('frozen_mean',means[si]['eta_b'])]:
                a=action(eta,eta,.968,name='probe_'+arm);c,cp=case(base,name,s,a,0,si,probe=True);c['probe_arm']=arm;rows.append(c);profiles[cp['controller_uid']]=cp
    assert len(rows)==3904
    register('probe',dict(base,states=states,phase='probe'),rows,profiles)
    missing=read(HERE/'probe_MISSING_CASES.json');write(HERE/'probe_EXECUTION_CASES.json',missing)
    write(HERE/'probe_BUDGET.json',dict(utc=utc(),requests=len(rows),exact_DB_reuse=len(rows)-len(missing),actual_execution_cap=len(missing),legacy300_not_extended=True,unlabeled_records_not_rollout_outcomes=True,probe_journal_policy='append_journal official default; merger skips unlabeled records, no fake success or outcome'))
    seal('probe_EXECUTION_FROZEN.receipt.json',[HERE/n for n in ('probe_protocol.json','probe_CASES.json','probe_PREFLIGHT.receipt.json','probe_EXECUTION_CASES.json','probe_BUDGET.json','PROBE_FROZEN_MEANS.json','PROBE_FEATURE_DEFINITION.receipt.json','OPENING_RECORD.receipt.json','design_probe_c6.py','worker_c6.py','actions_c6.py','execution_c6.py','schedule_c6.py')],before_rollout=True,no_task_outcomes=True)
    print('Probe preflight sealed:',len(missing),'genuinely missing',flush=True)

if __name__=='__main__':main()
