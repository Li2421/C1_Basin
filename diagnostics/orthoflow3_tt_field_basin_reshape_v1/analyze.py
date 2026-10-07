"""Read-only provenance audit and paired basin analysis; never executes a rollout.

Use the project's .venv-c1 Python. Export once from archived databases; subsequent
analyses use the portable seed-level snapshot in this directory.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import itertools
import json
import math
import sqlite3
import struct
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, qmc

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[1]
MATCHED = MAIN / 'diagnostics/orthoflow3_tt_ff_matched_learnability_v1'
FIELD = MAIN.parent / 'Basin_C1_flow_field_poc_20261004'
FAMILY = FIELD / 'family_study_v2'
SCENES = ('toy_give_way', 'ring_exchange')
CHAINS = ('TT', 'FT', 'TF', 'FF')


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ident(prefix, obj):
    return prefix + '_' + hashlib.sha256(canonical(obj).encode()).hexdigest()


def eta_id(eta):
    hx = ':'.join(struct.pack('>d', float(x)).hex() for x in eta)
    return ident('eta', {'float64_be': hx})


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def ro(path):
    con = sqlite3.connect('file:' + str(Path(path).resolve()) + '?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA query_only=ON')
    return con


def seed_key(state, seed):
    ns = int(hashlib.sha256(state['uid'].encode()).hexdigest()[:8], 16)
    return canonical(dict(future_index=seed, future_root=2026100403, rng_namespace=ns))


def export():
    """Verify native keys/code/records, export all primary and factorial evidence."""
    p, f = read(MATCHED/'protocol.json'), read(FAMILY/'protocol.json')
    assert p['eta'] == f['eta'][:16]
    expected = qmc.scale(qmc.Sobol(3, scramble=True, seed=202610042).random_base2(4),
                         [.5, -.5, 0], [1.25, .5, .75])
    np.testing.assert_array_equal(expected, p['eta'])
    files = {}

    def check(path, expected_hash=None):
        path = Path(path)
        if str(path) not in files:
            files[str(path)] = sha(path)
        if expected_hash is not None:
            assert files[str(path)] == expected_hash, ('Source hash mismatch', path)
        return files[str(path)]

    check(MATCHED/'execution.py', p['execution_sha256'])
    freeze = read(MATCHED/'test_truth_frozen.json')
    check(MATCHED/'test_truth.npz', freeze['truth_sha256'])
    check(MATCHED/'models_frozen.json', freeze['model_freeze_sha256'])
    check(MATCHED/'test_predictions_frozen.json', freeze['prediction_freeze_sha256'])
    dep = read(MATCHED/'runtime_dependency_audit.json')
    check(FIELD/'source_snapshot.json', dep['snapshot_sha256'])
    for r in dep['files']:
        check(r['path'], r['sha256'])
    for r in dep['main_vs_copied_import_equivalence']:
        check(r['main_path'], r['sha256'])
        check(r['copied_path'], r['sha256'])
    for rel, hx in f['controller_code_hashes'].items():
        check(FIELD/rel, hx)
    for sc in SCENES:
        pp = p['controllers'][sc]
        a, b = (pp[ch]['payload'] for ch in ('TT', 'FF'))
        for k in ('scenario_uid', 'flow_checkpoint_sha256', 'orthoflow3_sha256',
                  'safety_projection_sha256', 'cbf', 'environment_sha256', 'horizon',
                  'dt', 'success_semantics', 'conditioning', 'rng', 'jax_enable_x64',
                  'pseudo_steps', 'action_code_hashes', 'agent_slot_semantics'):
            assert a[k] == b[k], ('Unpaired controller configuration', sc, k)
        assert pp['TT']['controller_uid'] != pp['FF']['controller_uid']
        check(a['flow_checkpoint_path'], a['flow_checkpoint_sha256'])
        check(b['execution_callable'], b['execution_callable_sha256'])
    smoke = read(MATCHED/'execution_smoke.json')
    assert smoke['success'] and smoke['rng_key_parity'] and smoke['TT_FF_UID_disjoint']
    assert smoke['protocol_sha256'] == check(MATCHED/'protocol.json')
    assert read(FAMILY/'postflight.json')['protocol_sha256'] == check(FAMILY/'protocol.json')

    indices = [i for i, s in enumerate(p['states']) if s['metadata']['split'] == 'test']
    states = [p['states'][i] for i in indices]
    assert len(states) == 32 and len({s['family'] for s in states}) == 32
    assert all(s['metadata']['outcome_blind'] and s['physical']['timestep'] == 0 for s in states)
    assert all(sum(s['scenario'] == sc for s in states) == 16 for sc in SCENES)
    physical = lambda s: (s['scenario'], canonical(s['physical']))
    assert not ({physical(s) for s in states} & {physical(s) for s in f['states']})
    source_family = {s['family'] for s in p['states'] if s['metadata']['split'] != 'test'}
    assert not (source_family & {s['family'] for s in states})
    allrows, journals, ledger, requests = [], {}, [], []
    primary = np.zeros((2, 32, 16, 16, 3), np.int8)
    with ro(MAIN/'shared_rollout_db/rollout.sqlite') as db:
        for si, s in enumerate(states):
            sr = db.execute('SELECT * FROM state WHERE state_uid=?', (s['state_uid'],)).fetchone()
            assert sr and sr['content_hash'] == s['content_hash'] and sr['identity_quality'] == 'CONTENT_EXACT'
            assert json.loads(sr['physical_state_json']) == s['physical']
            for ci, ch in enumerate(('TT', 'FF')):
                ctl = p['controllers'][s['scenario']][ch]
                cr = db.execute('SELECT * FROM controller_config WHERE controller_uid=?', (ctl['controller_uid'],)).fetchone()
                assert cr and cr['compatibility_quality'] == 'EXACT_PROFILE'
                assert json.loads(cr['config_json']) == ctl['payload']
                for ei, eta in enumerate(p['eta']):
                    eid = eta_id(eta)
                    rr = {r['seed_key']: dict(r) for r in db.execute(
                        'SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',
                        (s['state_uid'], eid, ctl['controller_uid']))}
                    requests.append(dict(state_uid=s['state_uid'], eta_uid=eid,
                                         controller_uid=ctl['controller_uid'], seed_keys=[seed_key(s,k) for k in range(16)]))
                    for k in range(16):
                        key = seed_key(s, k)
                        assert key in rr, ('Missing exact paired record', s['uid'], ei, ch, k)
                        r = rr[key]
                        assert not r['conflict_quarantined'] and r['compatibility_quality'] == 'EXACT_REUSE'
                        src = r['original_source_file']
                        if src not in journals:
                            rows = [json.loads(line) for line in Path(src).read_text().splitlines()]
                            journals[src] = {ident('record', v): v for v in rows}
                            check(src)
                            known = db.execute('SELECT sha256 FROM source_file WHERE path=?', (src,)).fetchall()
                            assert files[src] in {v['sha256'] for v in known}
                        jr = journals[src][r['raw_record_hash']]
                        raw = jr['raw']
                        assert jr['state_uid'] == s['state_uid'] and jr['physical_content_hash'] == s['content_hash']
                        assert jr['eta_uid'] == eid and jr['controller_uid'] == ctl['controller_uid'] and jr['seed_key'] == key
                        assert jr['protocol_sha256'] == files[str(MATCHED/'protocol.json')]
                        assert raw['state_uid'] == s['uid'] and raw['seed'] == k and raw['eta'] == eta
                        for name in ('success', 'collision', 'numerical_failure'):
                            assert jr[name] == r[name]
                        assert r['success'] == int(raw['success']) and r['numerical_failure'] == int(raw['error'] is not None)
                        assert bool(r['success']) == (raw['termination'] == 'success' and raw['error'] is None and not raw['collision'])
                        primary[ci, si, ei, k] = [r['success'], r['numerical_failure'], r['collision']]
                        allrows.append(dict(cohort='primary', state_index=si, state_uid=s['uid'], scenario=s['scenario'],
                            eta_index=ei, eta1=eta[0], eta2=eta[1], eta3=eta[2], chain=ch, seed=k,
                            seed_key=key, success=r['success'], numerical=r['numerical_failure'], collision=r['collision'],
                            termination=r['outcome'], controller_uid=ctl['controller_uid'], record_id=r['rollout_uid'],
                            source=src, record_hash=r['raw_record_hash']))
        check(MATCHED/'canonical_test_keys.json')
        key_map = {(r['state_index'], r['eta_index'], r['chain']): r['rollout_uids']
                   for r in read(MATCHED/'canonical_test_keys.json')}
        for si, i in enumerate(indices):
            for ei in range(16):
                for ch in ('TT', 'FF'):
                    actual = [r['record_id'] for r in allrows if r['state_index']==si and r['eta_index']==ei and r['chain']==ch]
                    assert actual == key_map[i, ei, ch]
    archived = np.load(MATCHED/'test_truth.npz')
    np.testing.assert_array_equal(archived['indices'], indices)
    np.testing.assert_array_equal(archived['outcomes'], primary)

    factorial = np.zeros((4, 16, 16, 16, 3), np.int8)
    zero = np.zeros((4, 16, 16, 3), np.int8)
    state_index = {s['uid']: i for i, s in enumerate(f['states'])}
    seen = set()
    with ro(FAMILY/'rollout.sqlite') as db:
        for row in db.execute("SELECT * FROM rollout WHERE phase='independent' AND pseudo_steps=10 ORDER BY case_id"):
            r = json.loads(row['result_json'])
            ei = r['eta_index']
            if ei == 16:
                continue  # pre-existing fixed center is not in either estimand
            assert ei in list(range(16)) + [17]
            si, ci, k = state_index[r['state_uid']], CHAINS.index(r['chain']), r['seed']
            s, eta = f['states'][si], f['eta'][ei]
            task = dict(phase='independent', state=s, eta_index=ei, eta=eta,
                        seed=k, chain=r['chain'], pseudo_steps=10)
            expected_id = hashlib.sha256(canonical(task).encode()).hexdigest()[:24]
            assert row['case_id'] == r['id'] == expected_id
            assert row['controller_uid'] == r['controller_uid'] and json.loads(row['eta_json']) == r['eta'] == eta
            path = FAMILY/'results'/(r['id']+'.json')
            check(path, row['result_sha256'])
            assert read(path) == r and r['protocol_sha256'] == files[str(FAMILY/'protocol.json')]
            key = ci, si, ei, k
            assert key not in seen
            seen.add(key)
            value = [int(r['success']), int(r['error'] is not None), int(r['collision'])]
            assert not (value[0] and (value[1] or value[2]))
            assert bool(value[0]) == (r['termination'] == 'success' and r['error'] is None and not r['collision'])
            if ei == 17:
                zero[ci, si, k] = value
            else:
                factorial[ci, si, ei, k] = value
            allrows.append(dict(cohort='factorial_zero' if ei==17 else 'factorial', state_index=si,
                state_uid=s['uid'], scenario=s['scenario'], eta_index=ei, eta1=eta[0], eta2=eta[1], eta3=eta[2],
                chain=r['chain'], seed=k, seed_key=seed_key(s,k), success=value[0], numerical=value[1], collision=value[2],
                termination=r['termination'], controller_uid=r['controller_uid'], record_id=r['id'],
                source=str(path), record_hash=row['result_sha256']))
    assert len(seen) == 4*16*17*16
    for sc in SCENES:
        for ch in CHAINS:
            assert len({r['controller_uid'] for r in allrows if r['cohort']=='factorial' and r['scenario']==sc and r['chain']==ch})==1

    # Verify stored physical/noise traces in a fixed, outcome-blind audit subset.
    # Existing postflight validated all traces. Here independently compare all
    # four chains' stored keys on eta index0/seed0 for every factorial state.
    trace_pairs = 0
    for si, s in enumerate(f['states']):
        traces = []
        for ch in CHAINS:
            r = next(v for v in allrows if v['cohort']=='factorial' and v['state_index']==si
                     and v['chain']==ch and v['eta_index']==0 and v['seed']==0)
            path = Path(r['source']).with_suffix('.npz')
            with ro(FAMILY/'rollout.sqlite') as db:
                hx = db.execute('SELECT npz_sha256 FROM rollout WHERE case_id=?',(r['record_id'],)).fetchone()[0]
            check(path, hx)
            z = np.load(path)
            np.testing.assert_allclose(z['positions'][0], s['physical']['positions'], atol=1e-12, rtol=0)
            traces.append(z['noise_keys'])
        for a,b in itertools.combinations(traces,2):
            n=min(len(a),len(b))
            np.testing.assert_array_equal(a[:n],b[:n])
            trace_pairs += 1

    for path in [MATCHED/'plan.json', MATCHED/'design.py', MATCHED/'worker.py',
                 MATCHED/'execution_smoke.json', MATCHED/'runtime_dependency_audit.json',
                 FAMILY/'design.py', FAMILY/'runner.py', FAMILY/'postflight.json',
                 FAMILY/'analysis.json', FAMILY/'decision.json', FAMILY/'supplement.json',
                 FIELD/'validation_stage2/REPORT.md', FIELD/'family_study_v1/REPORT.md',
                 FIELD/'family_study_v1/preregistered_protocol.json',
                 FIELD/'family_study_v1/runner.py', FIELD/'validation_stage2/factorial_controller.py',
                 MAIN/'shared_rollout_db/seed_policy_v1.json', MAIN/'shared_rollout_db/README.md',
                 MAIN/'diagnostics/orthoflow3_core_hypothesis_audit_v1/final_report.md',
                 MAIN/'diagnostics/orthoflow3_controller_conditioning_probe_v1/final_report.md',
                 MAIN/'diagnostics/orthoflow3_t0_basin_shape_v1/final_report.md',
                 MAIN/'diagnostics/orthoflow3_closed_loop_contract_v1/CONTINUATION_CONTRACT.md']:
        check(path)
    assert read(FIELD/'family_study_v1/preregistered_protocol.json')['future_root'] == f['future_root'] == 2026100403
    assert not primary[...,2].any() and not factorial[...,2].any()
    (HERE/'inputs').mkdir(exist_ok=True)
    write(HERE/'inputs/primary_protocol.json', p)
    write(HERE/'inputs/factorial_protocol.json', f)
    write(HERE/'inputs/factorial_prior_analysis.json', read(FAMILY/'analysis.json'))
    write(HERE/'inputs/factorial_prior_decision.json', read(FAMILY/'decision.json'))
    write(HERE/'inputs/primary_states.json', states)
    np.savez_compressed(HERE/'paired_snapshot.npz', primary=primary, factorial=factorial,
                        zero=zero, eta=expected)
    allrows.sort(key=lambda r:(r['cohort'],r['state_index'],r['eta_index'],r['chain'],r['seed']))
    with gzip.open(HERE/'seed_outcomes.csv.gz','wt',newline='') as out:
        w=csv.DictWriter(out, list(allrows[0]));w.writeheader();w.writerows(allrows)
    write(HERE/'planned_reuse.json',dict(requests=requests,new_rollouts=0,source='original exact TEST requests'))
    summary=dict(primary_requested=16384, primary_scientific_records=int(16384-primary[...,1].sum()),
        primary_numerical_attempts=int(primary[...,1].sum()), factorial_primary_requested=16384,
        factorial_numerical_attempts=int(factorial[...,1].sum()), zero_control_requested=1024,
        total_archived_records=len(allrows), missing_attempted_records=0, ambiguous_identities=0,
        conflicts=0, new_rollouts=0, numerical_imputed=0, controllers_not_relabeled=True,
        primary_tensor_matches_archive=True, original_test_keys_match=True,
        regenerated_Sobol_bit_exact=True, independent_trace_key_comparisons=trace_pairs,
        source_files_verified=len(files), database_access='mode=ro; query_only=ON; no DB writes',
        pairing='exact physical state, float64 eta, state+seed+physical-step RNG, controller-specific immutable records')
    write(HERE/'cache_audit.json',summary)
    write(HERE/'source_hashes.json',files)
    write(HERE/'snapshot_integrity.json', {str(x.relative_to(HERE)):sha(x) for x in [
        HERE/'paired_snapshot.npz',HERE/'seed_outcomes.csv.gz',*sorted((HERE/'inputs').glob('*.json'))]})
    print(json.dumps(summary),flush=True)


def membership(outcomes, threshold=15):
    successes = outcomes[...,0].sum(-1)
    unknown = outcomes[...,1].sum(-1)
    return np.where(successes>=threshold,1,np.where(successes+unknown<threshold,0,-1)), successes, unknown


def wilson(s,n):
    z=1.959963984540054;p=s/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    delta=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-delta),min(1.,center+delta)]


def signflip_exact(integer_differences):
    """Whole-state sign permutations via integer dynamic programming."""
    d=[abs(int(x)) for x in integer_differences if x]
    counts={0:1}
    for v in d:
        new=collections.defaultdict(int)
        for total,n in counts.items():new[total-v]+=n;new[total+v]+=n
        counts=dict(new)
    observed=abs(int(sum(integer_differences)))
    return sum(n for t,n in counts.items() if abs(t)>=observed)/(2**len(d))


def bootstrap_indices(scenes):
    rng=np.random.default_rng(2026100601)
    groups=[np.flatnonzero(np.asarray(scenes)==s) for s in sorted(set(scenes))]
    return np.concatenate([rng.choice(g,size=(50000,len(g)),replace=True) for g in groups],axis=1)


def summarize_complete(a,b,states,draw):
    assert np.all(a>=0) and np.all(b>=0)
    rescue=(a==0)&(b==1);brk=(a==1)&(b==0);both=(a==1)&(b==1);neither=(a==0)&(b==0)
    changed=rescue|brk;n,m=a.shape
    d=b.sum(1)-a.sum(1);wins=int((d>0).sum());losses=int((d<0).sum());ties=n-wins-losses
    def ci(v):return np.quantile(np.asarray(v)[draw].mean(1),[.025,.975]).tolist()
    s=dict(states=n,eta=m,cells=n*m,TT_count=int(a.sum()),Field_count=int(b.sum()),
        TT_fraction=float(a.mean()),Field_fraction=float(b.mean()),
        change_count=int(changed.sum()),change_rate=float(changed.mean()),
        rescue=int(rescue.sum()),break_count=int(brk.sum()),both_success=int(both.sum()),both_failure=int(neither.sum()),
        volume_delta=float(d.mean()/m),volume_delta_state_bootstrap95=ci(d/m),
        change_rate_state_bootstrap95=ci(changed.mean(1)),TT_fraction_state_bootstrap95=ci(a.mean(1)),
        Field_fraction_state_bootstrap95=ci(b.mean(1)),states_changed=int(changed.any(1).sum()),
        states_changed_fraction=float(changed.any(1).mean()),states_changed_wilson95=wilson(int(changed.any(1).sum()),n),
        volume_increase_states=wins,volume_decrease_states=losses,volume_tie_states=ties,
        state_sign_p=float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.,
        state_signflip_p=signflip_exact(d),volume_delta_quantiles=np.quantile(d/m,[0,.25,.5,.75,1]).tolist())
    assert s['Field_count']-s['TT_count']==s['rescue']-s['break_count']
    assert sum(s[k] for k in ('rescue','break_count','both_success','both_failure'))==n*m
    return s


def summarize(a,b,states):
    scenes=[s['scenario'] for s in states];draw=bootstrap_indices(scenes)
    locations=[(ch,int(i),int(j)) for ch,x in enumerate((a,b)) for i,j in np.argwhere(x<0)]
    if not locations:
        result=summarize_complete(a,b,states,draw)
        return {**result,'unresolved_memberships':0}
    assert len(locations)<=8,'Use conservative interval analysis rather than exponential enumeration'
    candidates=[]
    for values in itertools.product((0,1),repeat=len(locations)):
        xx=[a.copy(),b.copy()]
        for (ch,i,j),v in zip(locations,values):xx[ch][i,j]=v
        candidates.append(summarize_complete(*xx,states,draw))
    result={k:np.stack([c[k] for c in candidates]).min(axis=0).tolist() for k in candidates[0]}
    bounds={k:[np.stack([c[k] for c in candidates]).min(axis=0).tolist(),
               np.stack([c[k] for c in candidates]).max(axis=0).tolist()] for k in candidates[0]}
    return dict(unresolved_memberships=len(locations),assignment_bounds=bounds,
                description='All mathematically possible unresolved membership assignments; no failure imputation')


def holm(p):
    p=np.asarray(p);order=np.argsort(p);adj=np.empty(len(p))
    adj[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return adj


def analyze():
    for path,hx in read(HERE/'snapshot_integrity.json').items():assert sha(HERE/path)==hx,path
    data=np.load(HERE/'paired_snapshot.npz');primary=data['primary'];factorial=data['factorial'];eta=data['eta']
    states=read(HERE/'inputs/primary_states.json');f=read(HERE/'inputs/factorial_protocol.json')
    perstate=[];cells=[];summaries=[]
    for cohort,arr,ss,chains in [('primary',primary,states,('TT','FF')),
                               ('factorial',factorial,f['states'],CHAINS)]:
        b,s,u=membership(arr)
        comparisons=[('TT','FF')] if cohort=='primary' else [('TT','FF'),('TT','TF'),('FT','FF'),('TT','FT'),('TF','FF')]
        for ca,cb in comparisons:
            a,bb=b[chains.index(ca)],b[chains.index(cb)]
            for scene in ('all',*SCENES):
                ix=[i for i,st in enumerate(ss) if scene=='all' or st['scenario']==scene]
                summary=summarize(a[ix],bb[ix],[ss[i] for i in ix])
                summaries.append(dict(cohort=cohort,scene=scene,contrast=ca+'->'+cb,**summary))
            for i,st in enumerate(ss):
                lower_a=(a[i]==1);upper_a=a[i]!=0;lower_b=bb[i]==1;upper_b=bb[i]!=0
                known=(a[i]>=0)&(bb[i]>=0);rescue=(a[i]==0)&(bb[i]==1);brk=(a[i]==1)&(bb[i]==0)
                perstate.append(dict(cohort=cohort,contrast=ca+'->'+cb,state_uid=st['uid'],scenario=st['scenario'],
                    TT_count_lower=int(lower_a.sum()),TT_count_upper=int(upper_a.sum()),
                    Field_count_lower=int(lower_b.sum()),Field_count_upper=int(upper_b.sum()),
                    TT_fraction_lower=float(lower_a.mean()),TT_fraction_upper=float(upper_a.mean()),
                    Field_fraction_lower=float(lower_b.mean()),Field_fraction_upper=float(upper_b.mean()),
                    rescue=int(rescue.sum()),break_count=int(brk.sum()),both_success=int(((a[i]==1)&(bb[i]==1)).sum()),
                    both_failure=int(((a[i]==0)&(bb[i]==0)).sum()),unresolved_pairs=int((~known).sum()),
                    change_lower=float((rescue|brk).mean()),change_upper=float(((rescue|brk)|~known).mean()),
                    delta_lower=float(lower_b.mean()-upper_a.mean()),delta_upper=float(upper_b.mean()-lower_a.mean())))
        for ci,ch in enumerate(chains):
            for i,st in enumerate(ss):
                for j,e in enumerate(eta):
                    cells.append(dict(cohort=cohort,state_uid=st['uid'],scenario=st['scenario'],chain=ch,eta_index=j,
                        eta1=e[0],eta2=e[1],eta3=e[2],successes=int(s[ci,i,j]),scientific_failures=int(16-s[ci,i,j]-u[ci,i,j]),
                        numerical_unknown=int(u[ci,i,j]),B15=int(b[ci,i,j]),Q16_lower=float(s[ci,i,j]/16),
                        Q16_upper=float((s[ci,i,j]+u[ci,i,j])/16)))
    csv_write(HERE/'per_state.csv',perstate);csv_write(HERE/'cell_membership.csv',cells)
    write(HERE/'summary.json',summaries)
    tab=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in summaries if r.get('unresolved_memberships')==0]
    csv_write(HERE/'main_quantitative_table.csv',tab)

    # Independent verification of archival factorial count summaries.
    old=read(HERE/'inputs/factorial_prior_analysis.json');fb,fs,fu=membership(factorial)
    old_by={x['state_uid']:x for x in old['states']}
    for i,st in enumerate(f['states']):
        for ci,ch in enumerate(CHAINS):
            expected=old_by[st['uid']]['cells'][ch]
            assert [int((fb[ci,i]==1).sum()),int((fb[ci,i]!=0).sum())]==expected['B15_count']
            np.testing.assert_array_equal(np.stack([fs[ci,i]/16,(fs[ci,i]+fu[ci,i])/16],axis=1),expected['Q16_distribution'])

    threshold=[]
    for t in (14,15,16):
        b,_,_=membership(primary,t)
        for scene in ('all',*SCENES):
            ix=[i for i,st in enumerate(states) if scene=='all' or st['scenario']==scene]
            threshold.append(dict(threshold=t,scene=scene,**summarize(b[0,ix],b[1,ix],[states[i] for i in ix])))
    write(HERE/'threshold_sensitivity.json',threshold)

    # These tests assess paired seed outcomes, not 512 independent state trials.
    qtests=[]
    for i,st in enumerate(states):
        for j in range(16):
            a,bb=primary[:,i,j,:,0];valid=~primary[:,i,j,:,1].any(axis=0)
            nr=int(((a==0)&(bb==1)&valid).sum());nb=int(((a==1)&(bb==0)&valid).sum())
            qtests.append(dict(state_uid=st['uid'],scenario=st['scenario'],eta_index=j,valid_pairs=int(valid.sum()),
                seed_rescue=nr,seed_break=nb,p=float(binomtest(nr,nr+nb).pvalue) if nr+nb else 1.))
    adj=holm([r['p'] for r in qtests])
    for r,v in zip(qtests,adj):r['p_Holm_512']=float(v)
    csv_write(HERE/'paired_seed_tests.csv',qtests)
    significant=[r for r in qtests if r['p_Holm_512']<.05]
    write(HERE/'paired_seed_test_summary.json',dict(tests=len(qtests),familywise_significant_cells=len(significant),
        significant_states=len({r['state_uid'] for r in significant}),
        positive_cells=sum(r['seed_rescue']>r['seed_break'] for r in significant),
        negative_cells=sum(r['seed_rescue']<r['seed_break'] for r in significant),
        inference='Exact matched-seed outcome tests; Holm controls multiplicity even across dependent cells. Common valid slots only; secondary solver-conditioned evidence.'))

    rng=np.random.default_rng(2026100602);stability=[]
    for repetition in range(2000):
        sampled=np.zeros_like(primary)
        for i in range(len(states)):
            take=rng.integers(0,16,16)
            sampled[:,i]=primary[:,i][:,:,take,:]
        b,_,_=membership(sampled)
        a,bb=b
        delta_lo=(bb==1).mean()-(a!=0).mean();delta_hi=(bb!=0).mean()-(a==1).mean()
        changed=(a>=0)&(bb>=0)&(a!=bb)
        possible=changed|((a<0)|(bb<0))
        stability.append([delta_lo,delta_hi,changed.mean(),possible.mean(),changed.any(1).mean(),possible.any(1).mean()])
    v=np.array(stability)
    write(HERE/'paired_seed_stability.json',dict(replicates=len(v),
        volume_delta_percentile95_conservative=[float(np.quantile(v[:,0],.025)),float(np.quantile(v[:,1],.975))],
        membership_change_percentile95_conservative=[float(np.quantile(v[:,2],.025)),float(np.quantile(v[:,3],.975))],
        changed_state_fraction_percentile95_conservative=[float(np.quantile(v[:,4],.025)),float(np.quantile(v[:,5],.975))],
        positive_delta_fraction_conservative=float((v[:,0]>0).mean()),
        qualification='Conditional paired-seed resampling stability only; all eta and chains share each state resample. Not an independent replication or a confidence bound on the population success basin.'))

    zb,zs,zu=membership(data['zero'])
    zero_controls=[]
    for ca,cb in [('TT','TF'),('FT','FF')]:
        ai,bi=CHAINS.index(ca),CHAINS.index(cb)
        zero_controls.append(dict(contrast=ca+'->'+cb,success_counts_equal=bool(np.array_equal(zs[ai],zs[bi])),
            same_memberships=bool(np.array_equal(zb[ai],zb[bi])),numerical=int(zu[[ai,bi]].sum()),
            seed_outcome_disagreements=int((data['zero'][ai,...,0]!=data['zero'][bi,...,0]).sum()),
            counts_before=zs[ai].tolist(),counts_after=zs[bi].tolist()))
    write(HERE/'zero_correction_control.json',zero_controls)
    representatives={sc:min([s for s in states if s['scenario']==sc],
        key=lambda s:hashlib.sha256(('basin-map-v1|'+s['uid']).encode()).hexdigest())['uid'] for sc in SCENES}
    write(HERE/'figure_selection.json',dict(rule=read(HERE/'analysis_protocol.json')['figures'],states=representatives))
    primary_rows=[r for r in summaries if r['cohort']=='primary']
    allrow=next(r for r in primary_rows if r['scene']=='all')
    reshape=allrow['states_changed_wilson95'][0]>.5 and all(r['states_changed_fraction']>.5 for r in primary_rows)
    expand=reshape and all(r['volume_delta_state_bootstrap95'][0]>0 and r['state_sign_p']<.05
                           and r['volume_increase_states']>r['volume_decrease_states'] for r in primary_rows)
    claim=('Field systematically reshapes and expands the robust success basin' if expand else
           'Field systematically reshapes the robust success basin' if reshape else
           'Field changes some intervention outcomes' if allrow['change_count'] else 'insufficient evidence')
    write(HERE/'decision.json',dict(claim=claim,
        mandatory_scope='Net empirical B15 expansion on the prespecified 16-point uniform Sobol measure in the tested Toy/Ring true-t0 cohorts. Not set inclusion or continuous/population-volume certification.',
        new_rollouts=0,analysis_is_retrospective=True,main=allrow,
        counterevidence='Earlier generator-proposal pilot Toy B15 count40->26; effects depend on state and eta measure. Do not claim universal enlargement.',
        primary_mechanism='TT->FF is the full formulation change; TT->TF factorial control isolates correction relocation with terminal safety. FT includes terminal safety closure, limiting literal orthogonality.',
        uncertainty='State bootstrap conditions on one eta bank and fixed seed panel; no independent Sobol scrambles. B15 is empirical, not a lower confidence bound on true p.'))
    print(json.dumps(dict(claim=claim,primary=primary_rows,paired_seed_significant=len(significant))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('export','analyze'))
    globals()[parser.parse_args().command]()
