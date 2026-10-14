"""Seal all preregistered rules and model artifacts before new outcome labels."""
from core import *


def main():
    prereg()
    assert not (HERE/'NEW_LABELS_STARTED.receipt.json').exists()
    assert not (HERE/'FINAL_FREEZE.receipt.json').exists()
    for path in ('HISTORICAL_EVIDENCE_AUDIT.json','TRAINING_AUDIT.json',
                 'CHECKPOINT_NOVELTY_AUDIT.json','COMPATIBILITY_AUDIT.json',
                 'STATE_DEDUP_AUDIT.json'):
        assert read(HERE/path)['status']=='PASS', path
    assert read(HERE/'A3_ALL/TRAINING_AUDIT.json')['status']=='FITTED_BEFORE_NEW_SUCCESS_LABELS'
    assert read(HERE/'R5_FROZEN.receipt.json')['final_success_labels_read_or_generated'] is False
    assert read(HERE/'CHOICES_FROZEN.receipt.json')['status']=='FROZEN_BEFORE_NEW_SUCCESS_LABELS'
    assert read(HERE/'R5_PROBE_AUDIT.json')['success_rows_created']==0
    base=read(HERE/'FINAL_BASE_PROFILES.json')
    assert len(base['eligible_controller_order'])==17
    choices=read(HERE/'FROZEN_CHOICES.json')
    assert choices['controller_order']==base['eligible_controller_order']
    assert len(choices['state_uids'])==48 and len(choices['controllers'])==17
    files=[
        'PREREGISTRATION.md','PREREGISTRATION.receipt.json',
        'PREREGISTRATION_AMENDMENT.md','PREREGISTRATION_AMENDMENT.receipt.json',
        'HISTORICAL_EVIDENCE_AUDIT.json','N1_DATASET_FILE_HASHES.json',
        'TRAINING_PLAN.json','TRAINING_AUDIT.json',
        'CHECKPOINT_NOVELTY_AUDIT.json','COMPATIBILITY_AUDIT.json',
        'FINAL_BASE_PROFILES.json','FRESH_STATES.json','STATE_DEDUP_AUDIT.json',
        'R5_FEATURES.npz','R5_PROBE_AUDIT.json','R5_FROZEN.receipt.json',
        'A3_ALL/transform.npz','A3_ALL/TRAINING_AUDIT.json',
        'FROZEN_CHOICES.json','CHOICES_FROZEN.receipt.json',
        'core.py','train_one.py','prepare_targets.py','states.py','fit_a3_all.py',
        'probe_r5.py','freeze_choices.py','freeze_final.py','design.py','run.py',
        'gate.py','merge.py','postflight_unknown_audit.py','analyze.py',
    ]
    for name in base['eligible_controller_order']:
        checkpoint=Path(base['profiles'][name]['payload']['flow_checkpoint_path'])
        assert checkpoint.is_relative_to(HERE) and checkpoint.exists()
        assert sha(checkpoint)==base['profiles'][name]['payload']['flow_checkpoint_sha256']
        files.append(str(checkpoint.relative_to(HERE)))
        result=checkpoint.parent/'ROUND17_TRAINING_RESULT.json'
        assert result.exists();files.append(str(result.relative_to(HERE)))
    for seed in (17,23,41):
        files.append(f'A3_ALL/A3_all_seed{seed}_step1000.msgpack')
        for model in ('A0','A3','A3_all'):
            files.append(f'predictions/{model}_seed{seed}.npz')
    files=sorted(set(files))
    for name in files:assert (HERE/name).is_file(),name
    receipt=dict(status='FROZEN_BEFORE_NEW_SUCCESS_LABELS',utc=utc(),
                 new_success_labels_generated=False,controller_count=17,
                 fresh_state_count=48,probe_count=544,choice_count=17,
                 A3_all_training_controller_count=92,
                 files={name:sha(HERE/name) for name in files},
                 external_model_freeze_sha256=sha(R16/'MODEL_FREEZE.receipt.json'),
                 external_GE_freeze_sha256=sha(R16/'predictions/G_E_FROZEN_CHOICES.json'))
    write(HERE/'FINAL_FREEZE.receipt.json',receipt)
    labels_allowed()
    print(json.dumps(dict(status=receipt['status'],files=len(files),
                          sha256=sha(HERE/'FINAL_FREEZE.receipt.json'))))


if __name__=='__main__':main()
