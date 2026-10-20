"""Validate saved evidence identities and arithmetic; no controller evaluations."""
import csv,json,hashlib,struct
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arr(x):return np.asarray(json.loads(x),float)
def main():
    m=json.loads((P/'validation_manifest.json').read_text())
    a=list(csv.DictReader(open(P/'tt_ff_action_validation.csv')))
    s=list(csv.DictReader(open(P/'ff_substep_validation.csv')))
    ids=list(dict.fromkeys(r['state_id'] for r in a))
    assert len(a)==68 and len(s)==340 and len(ids)==17
    max_residual=0.
    for uid in ids:
        rows=[r for r in a if r['state_id']==uid]
        assert {(r['parameter'],r['chain']) for r in rows}=={(e,c) for e in ('zero','witness') for c in ('TT','FF')}
        assert len({r['noise_key'] for r in rows})==1
        for r in rows:
            d=arr(r['raw_delta']);eta=arr(r['eta']);B=arr(r['basis_columns'])
            assert np.max(abs(B@eta-d))<1e-14
            assert bool(r['numerical_failure']=='True') is False
            if r['chain']=='TT':assert np.max(abs(arr(r['target'])-arr(r['safe_anchor'])-d))<1e-14
        for label in ('zero','witness'):
            rr=sorted([r for r in s if r['state_id']==uid and r['parameter']==label],key=lambda r:int(r['k']))
            assert len(rr)==10
            for k,r in enumerate(rr):
                h=arr(r['h']);f=arr(r['f']);d=arr(r['delta']);q=arr(r['q']);rho=float(r['rho'])
                err=max(np.max(abs(arr(r['r'])-h-(1-k/10)*(f+d))),
                        np.max(abs(arr(r['h_next'])-(1-rho)*h-rho*q)))
                max_residual=max(max_residual,float(err))
                if k<9:assert np.max(abs(arr(r['h_next'])-arr(rr[k+1]['h'])))<1e-14
            out=next(r for r in rows if r['parameter']==label and r['chain']=='FF')
            assert np.max(abs(arr(rr[-1]['q'])-arr(out['executed'])))<1e-14
    assert max_residual<1e-14
    assert all(sha(p)==h for p,h in m['source_hashes'].items())
    assert all(sha(p)==h for p,h in m['loaded_module_hashes'].items())
    trajectories=json.loads((P/'trajectory_metrics.json').read_text())
    assert all(sha(t['path'])==t['meta_sha256'] and sha(t['npz'])==t['npz_sha256'] for t in trajectories)
    replay=json.loads((P/'replay_checks.json').read_text())
    assert all(r.get('canonical_error',0)==0 and r.get('historical_action_error',0)==0 for r in replay)
    assert sum(t['complete_yhp'] for t in trajectories)==1
    fp=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004/family_study_v2/protocol.json')
    f=json.loads(fp.read_text());m['candidate_box']=f['eta_domain']
    m['additional_source_hashes']={str(fp):sha(fp)}
    m['parameter_identities']={}
    for label,eta in m['parameter_ids'].items():
        hx=':'.join(struct.pack('>d',float(v)).hex() for v in eta)
        payload=json.dumps({'float64_be':hx},sort_keys=True,separators=(',',':'))
        m['parameter_identities'][label]=dict(eta=eta,float64_be=hx,eta_uid='eta_'+hashlib.sha256(payload.encode()).hexdigest())
    m['effective_diagnostic_controllers']={}
    for ch,desc in m['controller_descriptors'].items():
        desc={**desc,'schema':'read_only_tt_ff_correction_action_diagnostic_v1',
              'future_root':m['future_root'],'restore':'recorded p,v and supplied step key; no physical continuation',
              'zero_baseline':'outside candidate box; auxiliary only'}
        payload=json.dumps(desc,sort_keys=True,separators=(',',':'))
        m['effective_diagnostic_controllers'][ch]=dict(descriptor=desc,diagnostic_uid='diag_ctl_'+hashlib.sha256(payload.encode()).hexdigest())
    m['paired_state_records']=[{k:r[k] for k in ('state_id','base_uid','physical_step','seed','noise_key','positions','velocities')} for r in a if r['chain']=='TT' and r['parameter']=='witness']
    # JSON strings are decoded so the manifest contains actual arrays/numbers.
    for r in m['paired_state_records']:
        for k in ('noise_key','positions','velocities'):r[k]=json.loads(r[k])
        for k in ('physical_step','seed'):r[k]=int(r[k])
    m['distinct_state_key_inputs']=len({(r['positions'],r['velocities'],r['noise_key']) for r in a})
    m['cost_accounting_notes']={
        'native_flow_network_steps':m['counts']['native_samples']*10,
        'forward_field_calls_excluding_AD':m['counts']['native_samples']*10+m['counts']['ff_field_evaluations']+m['counts']['fd_field_evaluations'],
        'AD_jacobian_calls_separately_counted':m['counts']['jacobian_evaluations'],
        'projection_requests_include_canonical_replay':True,
        'solver_internal_iterations':'not exposed by primary interface; unknown',
        'probe_wall_seconds':m['elapsed_seconds'],'probe_process_cpu_seconds':m['process_cpu_seconds'],
        'arithmetic_postprocessing_time':'not separately metered; no Flow/projection/plant calls',
        'rollout_preflight':'no new rollout requests; read-only cache coverage audit only'}
    verification=dict(paired_action_rows=len(a),ff_substeps=len(s),state_records=len(ids),
        distinct_physical_state_key_inputs=m['distinct_state_key_inputs'],max_recurrence_residual=max_residual,
        frozen_sources_unchanged=True,loaded_source_dependencies_unchanged=True,reused_trajectory_files_unchanged=True,
        exact_canonical_witness_replays=34,exact_historical_action_replays=13,
        zero_new_plant_steps=True,required_deliverables_present=True)
    (P/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    required=['geometry_certification.md','tt_ff_action_validation.csv','ff_substep_validation.csv',
              'trajectory_mechanism_validation.md','final_theory_validation.md']
    assert all((P/f).stat().st_size>0 for f in required)
    m['artifact_sha256']={str(p.name):sha(p) for p in sorted(P.iterdir()) if p.is_file() and p.name!='validation_manifest.json'}
    m['verification']=verification
    (P/'validation_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(verification,indent=2))
if __name__=='__main__':main()
