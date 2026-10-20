"""Cross-agent evidence and arithmetic checks. Does not execute controller or plant."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import json,csv,hashlib,datetime,platform
import numpy as np
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(name):return json.loads((P/name).read_text())
def main():
    audit=read('input_audit.json');A=read('agent_a/geometry_contract.json');B=read('agent_b/results.json')
    C=read('agent_c/results.json');D=read('agent_d/results.json');arith=read('arithmetic_certificate.json')
    checked={**audit['source_and_prior_evidence_hashes'],**D['read_hashes']}
    for p,h in checked.items():assert sha(p)==h,p
    b_uses_prior_snapshot=B['geometry_contract_sha256']!=sha(P/'agent_a/geometry_contract.json')
    consumed=read('geometry_contract_consumed_by_B.json') if b_uses_prior_snapshot else A
    if b_uses_prior_snapshot:
        assert B['geometry_contract_sha256']==sha(P/'geometry_contract_consumed_by_B.json')
    excluded={'free_space_cross_section','free_space_boundary_note'}
    assert {k:v for k,v in consumed.items() if k not in excluded}=={k:v for k,v in A.items() if k not in excluded}
    assert C['inputs']['sha256']==sha(P/'agent_a/geometry_contract.json')
    rows=list(csv.DictReader((P/'agent_b/zero_offset_actions.csv').open()))
    assert len(rows)==24 and not B['numerical_failures']
    walls=np.asarray([r['endpoints'] for r in A['walls']]);seg=walls[:,1]-walls[:,0]
    R=A['constants_m']['R_w'];Ds=A['constants_m']['D_s'];max_cert_error=0.
    for r in rows:
        p=np.asarray(json.loads(r['positions']));u=np.asarray(json.loads(r['executed']))
        rel=p[0]-p[1]
        t=np.clip(np.sum((p[:,None]-walls[:,0])*seg,axis=-1)/np.sum(seg*seg,axis=-1),0,1)
        delta=p[:,None]-(walls[:,0]+t[:,:,None]*seg);dist=np.linalg.norm(delta,axis=-1)
        normal=delta/dist[:,:,None]
        wallslack=np.sum(normal*u[:,None],axis=-1)+(dist-R)
        pairslack=2*rel@(u[0]-u[1])+(rel@rel-Ds*Ds)
        assert wallslack.min()>=-1e-9 and pairslack>=-1e-9
        assert np.linalg.norm(u,axis=1).max()<=.5+1e-10
        max_cert_error=max(max_cert_error,abs(min(float(wallslack.min()),float(pairslack))-float(r['min_cbf_slack'])))
        assert float(r['u2y'])>0
        if r['parameter']=='witness':assert np.asarray(json.loads(r['raw_delta']))[1,1]==0
    for uid in {r['state_id'] for r in rows}:
        for seed in ('0','1'):
            rr=[r for r in rows if r['state_id']==uid and r['seed']==seed]
            assert len({r['key'] for r in rr})==1
            tt=[r for r in rr if r['chain']=='TT']
            assert abs(float(tt[0]['u2y'])-float(tt[1]['u2y']))<1e-14
    assert max_cert_error<1e-12
    max_recurrence=0.
    for tr in read('agent_b/probe_traces.json'):
        path=np.asarray(tr['path']);q=np.asarray(tr['targets'])
        assert path.shape==(11,2,2) and q.shape==(10,2,2)
        for k in range(10):
            rho=min(1.,.1/(1-k/10))
            max_recurrence=max(max_recurrence,float(np.max(abs(path[k+1]-(1-rho)*path[k]-rho*q[k]))))
        r=next(r for r in rows if r['state_id']==tr['state_id'] and int(r['seed'])==tr['seed'] and r['parameter']==tr['parameter'] and r['chain']=='FF')
        assert np.max(abs(path[-1]-np.asarray(json.loads(r['executed']))))<1e-14
    assert max_recurrence<1e-14
    assert sum(int(r['projector_requests']) for r in rows)+B['native_samples']==B['projector_requests']==138
    assert B['ff_integrations']==12 and B['ff_fields']==120 and B['native_samples']==6
    covered=list(csv.DictReader((P/'agent_d/coverage.csv').open()))
    assert len(covered)==768 and D['available_records']==768
    assert D['paired_noise_mismatches']==D['unpaired_keys']==0
    observed={}
    for ch in ('TT','FF'):
        rr=[r for r in covered if r['chain']==ch]
        assert len(rr)==384
        assert len({r['state_uid'] for r in rr})==24
        assert all(r['available']=='True' and r['seed']=='0' for r in rr)
        out={k:sum(r[k]=='True' for r in rr) for k in ('entry_visited','pass_visited','pass_then_held_exit_no_collision','entry_then_pass_held_exit','collision','numerical_failure','success')}
        out['outcomes']=dict(Counter(r['termination'] for r in rr));observed[ch]=out
        for ix in range(16):
            panel=[r for r in rr if int(r['eta_index'])==ix]
            assert len(panel)==24
    assert observed['TT']['entry_visited']==83 and observed['FF']['entry_visited']==95
    assert observed['TT']['pass_visited']==0 and observed['FF']['pass_visited']==27
    assert observed['FF']['pass_then_held_exit_no_collision']==27
    assert observed['FF']['entry_then_pass_held_exit']==13
    assert sum(r['reference_certified'] for r in D['initials'])==24
    assert D['reference_initial_domain']['reference_total_steps_upper']==798<850
    for v in arith['exact_strict_margins'].values():assert Fraction(v['rational'])>0
    for k in ('conflict_domain_volume_m4','single_agent_area_m2','entry_fraction_of_conflict'):
        lo=Fraction(arith[k]['lower_rational']);hi=Fraction(arith[k]['upper_rational']);assert 0<lo<hi
    assert C['bounds']['4D_volume_m4']==float(Fraction(arith['exact_volumes_m4']['K_pass']))
    assert abs(D['volumes']['entry_m4']-float(Fraction(arith['exact_volumes_m4']['K_entry'])))<1e-18
    assert C['new_projection_calls']==C['new_flow_calls']==C['new_rollouts']==0
    assert D['new_rollouts']==D['new_flow']==D['new_projection']==D['new_steps']==0
    assert B['new_rollouts']==B['new_steps']==B['new_training']==0
    verification=dict(frozen_and_prior_files_checked=len(checked),all_input_hashes_unchanged=True,
        zero_offset_action_rows=24,all_reconstructed_actions_safe_with_frozen_tolerances=True,
        max_saved_certificate_error=max_cert_error,max_saved_FF_recurrence_error=max_recurrence,
        cached_trajectory_records=768,distinct_initial_states=24,fixed_eta_candidates=16,seed_indices=[0],
        paired_noise_prefix_mismatches=0,exact_arithmetic_strict_inequalities_pass=True,
        exact_volume_and_raw_margin_enclosures_pass=True,new_rollouts=0,new_physical_steps=0,
        observed_coverage=observed,full_episode_eta_star_evidence='UNKNOWN; not present in historical K16')
    (P/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    search=dict(requested_filename='C1_Narrow_Passage_Part1_to_Part6_v0_2.md',
        searched_roots=[audit['source_root'],audit['historical_root'],str(P.parents[1])],
        method='rg --files --hidden -g *C1_Narrow_Passage_Part1_to_Part6_v0_2* -g *Narrow_Passage* -g !**/.git/**',
        found=[],user_supplied_proposition_title='Lateral Reconfiguration Necessity',
        user_supplied_eq15='-h<=y1<=0, y_e<=y2<=H, b_-<=x2<=b_+, -S<=x1-x2<=D',
        full_manuscript_audit='NOT PERFORMED; structural paper recommendations explicitly CONDITIONAL')
    (P/'manuscript_search.json').write_text(json.dumps(search,indent=2)+'\n')
    revision=dict(B_consumed_sha256=B['geometry_contract_sha256'],
        final_A_sha256=sha(P/'agent_a/geometry_contract.json'),
        consumed_snapshot='geometry_contract_consumed_by_B.json' if b_uses_prior_snapshot else 'agent_a/geometry_contract.json',
        changed_fields=['free_space_cross_section','free_space_boundary_note'] if b_uses_prior_snapshot else [],
        reason='Root review corrected ownership at y=h; final middle branch is h<y<w. Original B input is retained byte-identically for provenance. A later full reproduction can consume the final contract directly.',
        invariants='Runtime config, source hashes, checkpoint, all walls, numeric constants, and all other fields identical. B proof directly checks finite-segment corner constraints and is unaffected by display-branch revision.')
    (P/'geometry_contract_revision.json').write_text(json.dumps(revision,indent=2)+'\n')
    manifest=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checkpoint=A['checkpoint'],checkpoint_sha256=A['checkpoint_sha256'],
        configuration=A['runtime_config'],cbf=A['cbf_config'],frozen_source_and_evidence_hashes=checked,
        agents=[dict(name=n,path=str(P/n)) for n in ('agent_a','agent_b','agent_c','agent_d')],
        dependency_order='A completed geometry -> independent B/C -> D after completed A/B/C -> root cross-check',
        new_probe=dict(states=3,seeds=[0,1],etas={'auxiliary_zero':[0,0,0],'witness':[.5,0,.5]},
            placements=['TT','FF'],previous_velocity='zero',future_root=2026100601,
            uid_rule=B['noise_uid_rule'],action_rows=24,native_samples=6,native_network_steps=60,
            ff_integrations=12,ff_field_calls=120,projector_requests=138,
            numerical_failures=B['numerical_failures'],measured_wall_seconds=B['wall_seconds'],
            measured_cpu_seconds=B['cpu_seconds'],purpose='sample actual instantaneous zero-offset startup; no trajectory or uniform guarantee'),
        evidence_reuse=dict(cached_trajectories=768,position_records=D['cached_position_records_read'],
            states=24,eta_candidates=16,seed=0,placements=['TT','FF'],unique_files_read=D['read_file_count'],
            measured_wall_seconds=D['wall_seconds'],measured_cpu_seconds=D['cpu_seconds']),
        extra_costs=dict(A_runtime_load_measured_wall=A['wall_seconds'],C_static_seconds=C['wall_seconds'],
            root='exact arithmetic and saved-evidence checks; no new controller evaluation',
            measured_scopes='per-script timed segments, not end-to-end conversation time'),
        totals=dict(new_rollouts=0,new_physical_steps=0,new_training=0,new_fits=0,gpu_seconds=0,
            new_action_rows=24,new_native_samples=6,new_ff_fields=120,new_projector_requests=138),
        limits=['configuration-space volume only; not full velocity/noise basin measure',
            'single-seed historical finite panel; no B15/population coverage',
            'eta* full episodes UNKNOWN','original paper not available; paper recommendations CONDITIONAL'],
        packages=dict(python=platform.python_version(),numpy=np.__version__),verification=verification)
    manifest['geometry_contract_revision']=revision
    manifest['artifacts_sha256']={str(p.relative_to(P)):sha(p) for p in sorted(P.rglob('*')) if p.is_file() and p.name!='validation_manifest.json'}
    (P/'validation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(verification,indent=2))
if __name__=='__main__':main()
