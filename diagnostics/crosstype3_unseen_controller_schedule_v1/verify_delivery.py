"""Final receipt graph, timing, scope and unchanged-confirmation checks."""
from common import *
from datetime import datetime
def main():
    assert (HERE/'DELIVERY.receipt.json').exists();failures=[];count=0
    for p in HERE.rglob('*.receipt.json'):
        for path,h in read(p).get('files',{}).items():
            count+=1
            if not Path(path).exists() or sha(path)!=h:failures.append(dict(receipt=str(p),file=path))
    assert not failures,failures[:10]
    choices=read(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json');assert len(choices['known_controllers'])==42 and len(set(choices['known_controllers']))==42
    assert set(choices['known_controllers'])==set(read(HERE/'PROMPT_REVISION_ACTIVE.json')['known_evaluation_order'])
    assert len(read(HERE/'FROZEN_PIPELINE.json')['outer'])==16
    frozen_time=datetime.fromisoformat(read(HERE/'FROZEN_PIPELINE.receipt.json')['utc']);chosen_time=datetime.fromisoformat(read(HERE/'NEW_EVAL_CHOICES.receipt.json')['utc']);submitted=datetime.fromisoformat(read(HERE/'new_evaluation_labels_SUBMISSION.json')['utc']);assert frozen_time<=chosen_time<submitted
    confirmation_time=datetime.fromisoformat(read(HERE/'CONFIRMATORY_RESULTS.receipt.json')['utc']);diagnostic_time=datetime.fromisoformat(read(HERE/'diagnostic_labels_SUBMISSION.json')['utc']);assert confirmation_time<diagnostic_time
    assert sha(HERE/'CONFIRMATORY_RESULTS.json')==read(HERE/'FAILURE_ATTRIBUTION.json')['confirmation_sha256']
    assert read(HERE/'NEW_EVAL_BUDGET.json')['conservative_additional_missing_cap']<=3744
    with np.load(HERE/'TRAINING_BUNDLE.npz',allow_pickle=False) as z:
        assert z['labels'].shape==(104,48,26,1) and not np.any(z['labels']==-2);names=list(z['names']);pure={k:z[k] for k in ('labels','uids','action','candidate_indices')}
    with np.load(D/'crosstype1c_two_eta_schedule_v1/TRAINING_BUNDLE.npz',allow_pickle=False) as z:
        for i,n in enumerate(z['names']):
            j=names.index(n)
            for k in pure:assert np.array_equal(pure[k][j],z[k][i]),(n,k)
    cases=read(HERE/'new_evaluation_CASES.json');seen_known=False
    for c in cases:
        assert c.get('rhr_rule') is None
        if c['base_controller_name'] in choices['known_controllers']:seen_known=True
        else:assert not seen_known,'Unseen evaluation must be scheduled first'
    for line in (HERE/'RESOURCE_CHECKS.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['admitted']:assert r['cpu_shards']+r['requested_cpu']<=18 and r['gpu_shards']+r['requested_gpu']<=2
    for phase in ('training','capability','probe','new_evaluation','diagnostic'):assert (HERE/f'{phase}_COMPLETE.receipt.json').exists()
    assert read(HERE/'INTEGRITY_COMPARISON.json')['passed']
    write(HERE/'FINAL_VERIFICATION.json',dict(utc=utc(),passed=True,receipt_file_links_verified=count,old62_training_labels_uids_actions_draws_exact=True,known_instances=42,frozen_outer_type_models=16,freeze_then_choices_then_new_submission=True,confirmation_before_diagnostic_submission=True,confirmation_unchanged_after_diagnostics=True,unseen_requests_before_known=True,conservative_extra_under3744=True,all_admitted_resource_checks_within18cpu2gpu=True))
    seal('FINAL_VERIFICATION.receipt.json',[HERE/'FINAL_VERIFICATION.json',HERE/'verify_delivery.py',HERE/'DELIVERY.receipt.json'],fully_reviewable=True)
    print('Final receipt/timing/scope verification passed;',count,'file links',flush=True)
if __name__=='__main__':main()
