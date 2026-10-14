from core import *
from shared_rollout_db.src.rollout_db import connect,canonical

def main():
    prereg();base=read(HERE/'FINAL_BASE_PROFILES.json');assert len(base['eligible_controller_order'])==8
    novelty=read(PARENT/'SEMANTIC_CONTROLLER_NOVELTY_AUDIT.json');excluded=novelty['already_evaluated_identical_best_policies'];assert not any(n+'__final' in base['profiles'] for n in excluded)
    eligible=set(n[:-7] for n in base['eligible_controller_order']);checks=[r for r in novelty['checks'] if r['run'] in eligible];assert len(checks)==8 and all(not r['policy_parameter_bytes_identical'] for r in checks)
    besthashes={r['best_policy_parameter_sha256'] for r in novelty['checks']};assert all(r['final_policy_parameter_sha256'] not in besthashes for r in checks)
    assert sha(HERE/'FRESH_STATES.json')==sha(PARENT/'FRESH_STATES.json');assert sha(HERE/'STATE_DEDUP_AUDIT.json')==sha(PARENT/'STATE_DEDUP_AUDIT.json')
    statechecks=[];counts=[]
    with connect(True) as db:
        for s in read(HERE/'FRESH_STATES.json'):
            r=db.execute('SELECT physical_state_json FROM state WHERE state_uid=?',(s['state_uid'],)).fetchone();assert r and r[0]==canonical(s['physical']);statechecks.append(s['state_uid'])
        for n in base['eligible_controller_order']:
            h=base['profiles'][n]['payload']['flow_checkpoint_sha256'];k=db.execute('SELECT COUNT(*) FROM rollout r JOIN controller_config c ON c.controller_uid=r.controller_uid WHERE c.flow_checkpoint_sha256=?',(h,)).fetchone()[0];assert k==0;counts.append(dict(controller=n,task_success_evidence_rows=k))
    write(HERE/'RESUMED_STATE_NOVELTY_AUDIT.json',dict(utc=utc(),passed=True,own_preregistered_states_reused=64,state_uids=statechecks,original_dedup_sha256=sha(PARENT/'STATE_DEDUP_AUDIT.json'),source_states_sha256=sha(PARENT/'FRESH_STATES.json'),final_evidence_counts=counts,all_distinct_from_all14_source_parameter_hashes=True,excluded_identical=excluded,no_success_labels_read=True))
    invalid_names=['probe_protocol.json','R5_FINAL_FEATURES.npz','R5_RAW_INDEX.json','R5_PROBE_AUDIT.json','R5_FROZEN.receipt.json','SELECTOR_MODELS.json','SELECTOR_FROZEN.receipt.json','FROZEN_CHOICES.csv','protocol.json','main_CASES.json','gate_CASES.json']
    invalid_files=[dict(path=str(PARENT/n),sha256=sha(PARENT/n)) for n in invalid_names]
    invalid_files += [dict(path=str(q),sha256=sha(q)) for q in sorted((PARENT/'probe/raw').glob('*'))]
    write(HERE/'INVALID_PRECISION_EVIDENCE.json',dict(utc=utc(),valid_for_confirmation=False,reason='Import overwrote FP32 with x64=True; all previous target descriptors/predictions/choices invalid, preserved. Historical source fits remain fixed.',files=invalid_files,precision_audit_sha256=sha(PARENT/'PRECISION_IMPORT_AUDIT.json'),final_success_labels_generated=0))
    print('PASS: 8 novel policies, original 64 planned states, zero final evidence')
if __name__=='__main__':main()
