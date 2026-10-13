"""Integrity and numerical cross-checks for the independent research handoff."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
H=Path(__file__).resolve().parent
def load(n):return json.loads((H/n).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=load('preregistration_receipt.json');a=load('addendum_e4_receipt.json')
    assert sha(H/'PREREGISTRATION.md')==p['sha256']
    assert sha(H/'ADDENDUM_E4.md')==a['sha256']
    results=load('results.json');idx=load('PANEL_INDEX.json');panels=load('E0_PANEL_RESULTS.json')
    ready=[x for x in idx if x['status']=='ready']
    assert len(ready)==len(panels)==97
    n=0
    for x in ready:
        file=Path(x['file'])
        assert file.is_file() and sha(file)==x['sha256']
        z=np.load(file,allow_pickle=False);y=z['Y']
        assert y.shape==(x['n_states'],x['n_eta'],16)
        assert np.isin(y,[0,1]).all()
        assert len(set(z['rollout_uids'].reshape(-1).astype(str)))==x['selected_rollouts']
        n+=int(y.size)
    assert results['E0']['seed_aligned_panels']==97
    assert results['E0']['primary_gate_pass']==33
    assert results['reuse']['unique_reused_overall']==187888
    assert results['reuse']['new_rollouts_this_task']==0
    assert results['E1']['status']=='NOT_TRIGGERED'
    assert results['E3']['status'].startswith('PARTIAL_REUSE_ONLY')
    source=json.loads((H.parent/'state_nuisance_rootcause_v1/heldout_adaptive_selection_v1/audit_results.json').read_text())
    comp=source['cohorts']['all']['comparisons']['qstate512_minus_qeta']
    assert abs(results['E2']['gain_pp']-100*comp['mean'])<1e-12
    assert all(abs(a-100*b)<1e-12 for a,b in zip(results['E2']['gain_state_bootstrap_ci95_pp'],comp['paired_physical_state_bootstrap_95']))
    for name in ['REPORT.md','HANDOFF.md','PREREGISTRATION.md','E0_PANEL_RESULTS.json',
                 'E4_REGIME_RESULTS.json','E5_DENSE_ETA_AUDIT.json','results.json']:
        assert (H/name).stat().st_size>100
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'valid':True,'aligned_panels':len(ready),'panel_label_positions_across_nested_panels':n,
             'distinct_reused_physical_rollouts':results['reuse']['unique_reused_overall'],
             'new_rollouts':0,'panel_hashes_verified':len(ready),
             'preregistration_hashes_verified':True,'historical_E2_numeric_crosscheck':True,
             'results_sha256':sha(H/'results.json'),
             'report_sha256':sha(H/'REPORT.md'),
             'handoff_sha256':sha(H/'HANDOFF.md')}
    (H/'VALIDATION.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print(receipt,flush=True)
if __name__=='__main__':main()
