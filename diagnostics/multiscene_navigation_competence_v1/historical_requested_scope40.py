"""Restrict the inherited audit to precisely the controller classes requested."""
from common import *
from collections import Counter

def main():
    assert not (HERE/'fits40').exists(),'Source scope must be corrected before any selector fitting'
    old=read(HERE/'HISTORICAL_COMPATIBLE_LABELS40.json');rows=[];excluded=[]
    for r in old:
        kind=r['profile'].get('controller_class');checkpoint=r['profile'].get('flow_checkpoint_sha256')
        permitted=(kind in [None,'G','deterministic','mixture','diffusion'] and checkpoint is not None) or kind=='DM'
        if permitted:
            assert r['profile']['orthoflow3_sha256']==sha(ROOT/'shared_control/basis_families.py')
            assert r['profile'].get('jax_enable_x64') is False
            rows.append(r)
        else:excluded.append(dict(rollout_uid=r['rollout_uid'],class_code=kind,reason='Controller class absent from explicit historical training-source list'))
    keys={r['nominal_key'] for r in rows};z=np.load(HERE/'selector_labels40/ring_exchange.npz');mask=np.isin(z['controller_names'],list(keys));N=len(mask)
    values={k:z[k][mask] if z[k].ndim and len(z[k])==N else z[k] for k in z.files}
    assert int(values['q'].sum())==len(rows) and int((values['q']-values['n']).sum())==sum(r['numerical_failure'] for r in rows)
    output=HERE/'selector_labels40/ring_exchange_requested.npz';assert not output.exists();np.savez_compressed(output,**values)
    p=HERE/'HISTORICAL_REQUESTED_COHORT40_FROZEN.json';features=read(HERE/'HISTORICAL_PROBE_FEATURES40.json')
    write(p,dict(utc=utc(),rows=rows,excluded=excluded,accepted_rows=len(rows),unknown_preserved=sum(r['numerical_failure'] for r in rows),nominal_instances=len(keys),complete_descriptors=len(keys-set(features['missing_full_descriptor_keys'])),missing_descriptor_keys=sorted(keys&set(features['missing_full_descriptor_keys'])),by_class=dict(Counter(r['profile'].get('controller_class') or 'Flow' for r in rows)),by_arm=dict(Counter(r['arm'] for r in rows)),new_rollouts=0,new_training=0,reason='Inherited round39 compatibility audit was broader than round39 explicit source-class list; correct the training-source scope before any selector fitting',old_audit_and_dataset_preserved=True,filtered_dataset=str(output),filtered_dataset_sha256=sha(output),basis_byte_identical_to_original_PoC=sha(ROOT/'shared_control/basis_families.py')==sha(FIELD/'source/shared_control/basis_families.py')))
    seal('HISTORICAL_REQUESTED_COHORT40_FROZEN.receipt.json',[p,output,HERE/'HISTORICAL_EVIDENCE40_FROZEN.receipt.json',HERE/'ring_exchange_SELECTOR_LABELS40_PACKAGED.receipt.json',HERE/'historical_requested_scope40.py'],before_any_selector_training=True,before_any_generated_corrective_evaluation_labels=True)
    write(HERE/'HISTORICAL_SELECTOR_SOURCE40_CURRENT.json',dict(receipt='HISTORICAL_REQUESTED_COHORT40_FROZEN.receipt.json',receipt_sha256=sha(HERE/'HISTORICAL_REQUESTED_COHORT40_FROZEN.receipt.json'),dataset='selector_labels40/ring_exchange_requested.npz',dataset_sha256=sha(output)))
    print(dict(accepted=len(rows),excluded=len(excluded),unknown=sum(r['numerical_failure'] for r in rows),instances=len(keys)),flush=True)

if __name__=='__main__':main()
