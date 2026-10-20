"""Post-confirmation accounting of executed/reused requests, without reruns."""
from common import *


def main():
    assert (HERE/'CONFIRMATORY_RESULTS.receipt.json').exists(),'Do not inspect locked execution metadata before confirmation'
    phases=set()
    for path in HERE.glob('*_SUBMISSION.json'):phases.update(read(path).get('phases',[]))
    records=[];seen=set();sources=[]
    for phase in sorted(phases):
        receipt=HERE/f'{phase}_COMPLETE.receipt.json';assert receipt.exists(),phase
        sources.append(receipt);cases=read(HERE/f'{phase}_CASES.json');missing=read(HERE/f'{phase}_MISSING_CASES.json')
        row=dict(phase=phase,planned_requests=len(cases),preflight_missing=len(missing),raw_requests=0,
            explicit_zero_physics_reuse=0,new_physical_trajectory_attempts=0,new_physical_steps=0,reused_probe_prefix_steps=0)
        for path in (HERE/phase/'raw').glob('*.json'):
            record=read(path);uid=record['rollout_uid'];assert uid not in seen,uid;seen.add(uid)
            row['raw_requests']+=1
            terminal_reuse=bool(record.get('physical_reuse',False))
            prefix_reuse=record.get('reused_prefix_source') is not None and record.get('new_physical_steps')==0
            row['explicit_zero_physics_reuse']+=int(terminal_reuse or prefix_reuse)
            row['new_physical_trajectory_attempts']+=int(not (terminal_reuse or prefix_reuse))
            row['new_physical_steps']+=int(record.get('new_physical_steps',0 if terminal_reuse else record['physical_steps']))
            row['reused_probe_prefix_steps']+=int(record.get('reused_prefix_steps',0))
        records.append(row)
    old=[r for r in records if r['phase'] in ('old_probe','old_evaluation','old_tuning_v2')]
    old_attempts=sum(r['new_physical_trajectory_attempts'] for r in old)
    assert old_attempts<=18846,(old_attempts,18846)
    write(HERE/'EXECUTION_BUDGET_FINAL.json',dict(utc=utc(),phases=records,
        old_type_new_physical_trajectory_attempts=old_attempts,old_type_authorized_cap=18846,
        development_additional_label_requests=0,
        notes=['Same-stage extra workers share the queue; submissions are not counted as additional requests.',
               'A fresh numerical failure still counts as an attempted trajectory even when no action was completed.',
               'Explicit terminal/preserved-prefix zero-physics reuse is counted separately.',
               'PPO checkpoint selection is separately itemized on training-held states.',
               'Mandatory reproduction and training-environment equivalence are separately retained unit-gate evidence.']))
    seal('EXECUTION_BUDGET_FINAL.receipt.json',[HERE/'EXECUTION_BUDGET_FINAL.json',HERE/'OLD_TYPE_FINAL_PREFLIGHT_BUDGET.json',
        HERE/'collect_execution_budget.py',*sources])
    print('Execution budget accounted; old-type physical attempts:',old_attempts)


if __name__=='__main__':main()
