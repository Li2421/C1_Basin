"""Register the completed source-only experiment in the global rollout DB."""
import json
from .pipeline import OUT,EXP,read
from shared_rollout_db.src.rollout_db import connect,transaction,canonical


def main():
    audit=read(OUT/'alignment_audit.json')
    decision=read(OUT/'final_decision.json')
    assert audit['journal_DB_aligned']==10628
    assert audit['exact_reusable']==15341
    assert audit.get('not_attempted',0)==audit.get('invalid',0)==0
    with connect() as db,transaction(db):
        row=db.execute('SELECT metadata_json FROM experiment WHERE experiment_uid=?',(EXP,)).fetchone()
        assert row is not None
        metadata=json.loads(row['metadata_json'])
        metadata.update(source_gate=decision['source_gate'],target_labels_opened=False,
                        final_decision_path=str(OUT/'final_decision.json'),
                        valid_exact_seed_records=15341,numerical_attempts_not_imputed=19,
                        input_quality_excluded_state_count=3)
        db.execute('''UPDATE experiment SET end_time=CURRENT_TIMESTAMP,
            new_rollout_count=?, reused_rollout_count=?,metadata_json=? WHERE experiment_uid=?''',
            (10628,4727,canonical(metadata),EXP))
    print({'experiment_uid':EXP,'new_rollouts':10628,'reused_valid_seed_records':4727,'complete':True})


if __name__=='__main__':main()
