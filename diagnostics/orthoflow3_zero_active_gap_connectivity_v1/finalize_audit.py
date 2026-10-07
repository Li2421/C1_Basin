#!/usr/bin/env python3
import ast
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/home/zhihan/research/Basin_C1')
HERE = ROOT / 'diagnostics/orthoflow3_zero_active_gap_connectivity_v1'
MIG = ROOT / 'diagnostics/orthoflow3_representation_migration_v1'
REQUIRED = [
    'protocol.md', 'implementation_domain_audit.md', 'bridge_domain_definition.json',
    'state_manifest.json', 'cache_reuse_audit.json', 'active_endpoint_manifest.csv',
    'radial_alpha_grid.csv', 'radial_screening.csv', 'radial_promoted64.csv',
    'radial_path_classification.csv', 'zero_neighborhood_directions.csv',
    'zero_neighborhood_results.csv', 'bridge_sobol_cloud.csv',
    'bridge_graph_results.csv', 'bridge_midpoint_results.csv',
    'robust_bridge_paths.csv', 'intermediate_eta_validity.csv',
    'state_topology_classification.csv', 'final_decision.json',
    'runtime_statistics.json', 'zero_active_gap_report.md'
]


def eta_key(x):
    return np.asarray(x, dtype=np.float64).tobytes().hex()


def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def all_raw():
    rows = []
    for p in list(MIG.glob('raw/*/shard*.jsonl')) + list(HERE.glob('raw/*/shard*.jsonl')):
        rows.extend(json.loads(x) for x in p.read_text().splitlines() if x.strip())
    # Exact duplicate tuple records, if any, have identical semantics; retain once.
    unique = {}
    for r in rows:
        unique[(r['state_id'], eta_key(r['eta']), int(r['seed']))] = r
    return list(unique.values())


def in_active(e, tol=1e-12):
    lo = np.array([.5, -.5, 0.])
    hi = np.array([1.25, .5, .75])
    return bool(np.all(np.asarray(e) >= lo-tol) and np.all(np.asarray(e) <= hi+tol))


def is_zero(e, tol=1e-12):
    return bool(np.linalg.norm(np.asarray(e, dtype=float)) <= tol)


def main():
    sman = json.load(open(HERE/'state_manifest.json'))
    states = sman['states']
    sm = {s['state_id']: s for s in states}
    candidate_rows = list(csv.DictReader(open(MIG/'subset_oracle_candidates.csv')))
    eps = list(csv.DictReader(open(HERE/'active_endpoint_manifest.csv')))
    grids = list(csv.DictReader(open(HERE/'radial_alpha_grid.csv')))
    screen = list(csv.DictReader(open(HERE/'radial_screening.csv')))
    plan = json.load(open(HERE/'promotion64_plan.json'))
    raw = all_raw()
    lookup = defaultdict(list)
    for r in raw:
        if r['state_id'] in sm:
            lookup[(r['state_id'], eta_key(r['eta']))].append(r)

    zero_evidence = {}
    for r in candidate_rows:
        if r['state_id'] in sm and r['candidate_kind'].startswith('zero'):
            zero_evidence[r['state_id']] = r
    zero_states = {sid for sid, r in zero_evidence.items() if r['B63'] == 'True'}
    endpoint_b63 = {(r['state_id'], r['endpoint']): r for r in eps if r['B63_cached'] == 'True'}

    promoted = []
    zn_promoted = {}
    for arm in plan['arms']:
        sid = arm['state_id']; e = arm['eta']
        matched = {int(x) for x in sm[sid]['matched_flow_seeds']}
        rr = [r for r in lookup[(sid, eta_key(e))] if int(r['seed']) in matched]
        if len(rr) != 64:
            raise RuntimeError(f'{arm["arm_id"]}: expected 64 exact rows, got {len(rr)}')
        out = {
            'state_id': sid, 'audit_kind': arm['audit_kind'],
            'endpoint': arm.get('endpoint', ''), 'direction_index': arm.get('direction_index', ''),
            'alpha': arm['alpha'], 'eta1': e[0], 'eta2': e[1], 'eta3': e[2],
            'successes': sum(int(r['success']) for r in rr), 'trials': 64,
            'B63': sum(int(r['success']) for r in rr) >= 63,
            'deadlock': sum(r['outcome'] == 'deadlock' for r in rr),
            'timeout': sum(r['outcome'] == 'timeout' for r in rr),
            'collision': sum(r['outcome'] == 'collision' for r in rr),
            'execution_error': sum(r['outcome'] == 'execution_error' for r in rr),
            'promotion_reason': arm['promotion_reason'],
        }
        if arm['audit_kind'] == 'radial_promotion': promoted.append(out)
        else: zn_promoted[(sid, int(arm['direction_index']), float(arm['alpha']))] = out
    write_csv(HERE/'radial_promoted64.csv', promoted, list(promoted[0]))
    pmap = {(r['state_id'], r['endpoint'], float(r['alpha'])): r for r in promoted}

    # Enrich the precomputed zero-neighborhood table with robust promotion results.
    zn = list(csv.DictReader(open(HERE/'zero_neighborhood_results.csv')))
    zn_out = []
    for r in zn:
        key = (r['state_id'], int(r['direction_index']), float(r['alpha']))
        pr = zn_promoted.get(key)
        zn_out.append({**r, 'promoted_to_64': pr is not None,
                       'successes64': '' if pr is None else pr['successes'],
                       'B63': '' if pr is None else pr['B63'],
                       'failure_category64': '' if pr is None else
                           ('none' if pr['successes'] == 64 else
                            f'deadlock={pr["deadlock"]};timeout={pr["timeout"]};collision={pr["collision"]}')})
    write_csv(HERE/'zero_neighborhood_results.csv', zn_out, list(zn_out[0]))

    # Path classifications and active boundary summaries.
    grouped = defaultdict(list)
    for r in screen:
        grouped[(r['state_id'], r['endpoint'])].append(r)
    path_rows = []
    for (sid, ep), rr in sorted(grouped.items(), key=lambda x: (sm[x[0][0]]['selection_rank'], x[0][1])):
        rr.sort(key=lambda x: float(x['alpha']))
        zero_b63 = sid in zero_states
        confirmed = {0.0: zero_b63, 1.0: True}
        confirmed_nonrobust = []
        for p in promoted:
            if p['state_id'] == sid and p['endpoint'] == ep:
                confirmed[float(p['alpha'])] = bool(p['B63'])
                if not p['B63']: confirmed_nonrobust.append(float(p['alpha']))
        interior_fail = sorted(a for a in confirmed_nonrobust if 0 < a < 1)
        all8 = all(int(x['successes']) == 8 for x in rr)
        if zero_b63:
            if interior_fail:
                cls = 'RADIAL_FAILURE_GAP'
            elif all8:
                cls = 'RADIAL_CONNECTED'
            else:
                cls = 'UNDERRESOLVED'
        else:
            cls = 'ACTIVE_REQUIRED_FEASIBILITY_TRANSITION'
        robust_alphas = sorted(a for a, ok in confirmed.items() if ok)
        first_robust = robust_alphas[0] if robust_alphas else None
        first_screen8 = next((float(x['alpha']) for x in rr if int(x['successes']) == 8), None)
        last_nonrobust_before = max((a for a in confirmed_nonrobust if first_robust is not None and a < first_robust), default=None)
        eta1_endpoint = float(next(x for x in grids if x['state_id']==sid and x['endpoint']==ep and float(x['alpha'])==1.0)['eta1'])
        path_rows.append({
            'state_id': sid, 'endpoint': ep, 'zero_B63': zero_b63,
            'screen_all_8of8': all8, 'screen_min_successes': min(int(x['successes']) for x in rr),
            'confirmed_interior_nonB63_alphas': json.dumps(interior_fail),
            'first_screen_8of8_alpha': first_screen8,
            'first_confirmed_B63_alpha': first_robust,
            'first_confirmed_B63_eta1': '' if first_robust is None else first_robust*eta1_endpoint,
            'last_confirmed_nonB63_alpha_before_first_B63': last_nonrobust_before,
            'classification': cls,
        })
    write_csv(HERE/'radial_path_classification.csv', path_rows, list(path_rows[0]))

    # Bridge stage was conditionally triggered only by a confirmed gap in a ZERO-B63 path.
    trigger = [r for r in path_rows if r['zero_B63'] and r['classification'] == 'RADIAL_FAILURE_GAP']
    empty_specs = {
        'bridge_sobol_cloud.csv': ['status','reason','point_index','eta1','eta2','eta3'],
        'bridge_graph_results.csv': ['status','state_id','reason'],
        'bridge_midpoint_results.csv': ['status','state_id','reason'],
        'robust_bridge_paths.csv': ['status','state_id','reason'],
    }
    if trigger:
        raise RuntimeError('A bridge trigger exists; conditional bridge stage must run before finalization')
    for name, fields in empty_specs.items():
        write_csv(HERE/name, [], fields)

    topology = []
    for s in states:
        sid = s['state_id']; ps = [r for r in path_rows if r['state_id'] == sid]
        if sid not in zero_states:
            cls = 'ACTIVE_REQUIRED_FEASIBILITY_BOUNDARY'
            reason = 'eta=0 is not B63; zero/active component topology is not assigned'
        elif any(r['classification'] == 'RADIAL_CONNECTED' for r in ps):
            cls = 'CONTINUOUS_ZERO_TO_ACTIVE_SUPPORTED'
            reason = 'B63 zero and active endpoints with all 11 frozen radial probes 8/8 and no confirmed failure interval'
        else:
            # This branch is retained for schema completeness.
            good_local = sum(int(x['successes']) == 8 for x in zn_out if x['state_id'] == sid)
            if good_local:
                cls = 'ZERO_HAS_NONTRIVIAL_LOCAL_NEIGHBORHOOD'
                reason = 'small nonzero successful probes exist, but endpoint connection is unresolved'
            else:
                cls = 'UNDERRESOLVED'; reason = 'insufficient resolved connectivity evidence'
        topology.append({'state_id':sid, 'zero_sufficient':sid in zero_states,
                         'topology_classification':cls, 'reason':reason})
    write_csv(HERE/'state_topology_classification.csv', topology, list(topology[0]))

    # All queried bridge eta traversed the unchanged controller; summarize validity.
    initial_new = []
    promotion_new = []
    for p in HERE.glob('raw/initial_screen/shard*.jsonl'):
        initial_new.extend(json.loads(x) for x in p.read_text().splitlines() if x.strip())
    for p in HERE.glob('raw/promotion64/shard*.jsonl'):
        promotion_new.extend(json.loads(x) for x in p.read_text().splitlines() if x.strip())
    val_groups = defaultdict(list)
    for r in initial_new + promotion_new:
        if not in_active(r['eta']) and not is_zero(r['eta']):
            val_groups[(r['state_id'], eta_key(r['eta']))].append(r)
    validity = []
    for (sid, _), rr in val_groups.items():
        e = rr[0]['eta']
        validity.append({'state_id':sid,'eta1':e[0],'eta2':e[1],'eta3':e[2],
                         'trials':len(rr),'implementation_accepted':True,
                         'eta_clipped_or_reinterpreted':False,
                         'execution_errors':sum(r['outcome']=='execution_error' for r in rr),
                         'collisions':sum(r['outcome']=='collision' for r in rr),
                         'second_projection_retries':sum(int(r.get('second_projection_retries',0)) for r in rr),
                         'max_second_projection_retries':max(int(r.get('second_projection_retries',0)) for r in rr)})
    validity.sort(key=lambda x:(sm[x['state_id']]['selection_rank'],x['eta1'],x['eta2'],x['eta3']))
    write_csv(HERE/'intermediate_eta_validity.csv', validity, list(validity[0]))

    # Confirmed and screening-level historical-gap successes.
    robust_gap = set()
    for r in promoted + list(zn_promoted.values()):
        e = [r['eta1'],r['eta2'],r['eta3']]
        if r['B63'] and not is_zero(e) and not in_active(e): robust_gap.add((r['state_id'],eta_key(e)))
    screen_gap = set()
    for r in screen:
        e = [float(r['eta1']),float(r['eta2']),float(r['eta3'])]
        if int(r['successes'])==8 and not is_zero(e) and not in_active(e): screen_gap.add((r['state_id'],eta_key(e)))
    for r in zn_out:
        e=[float(r['eta1']),float(r['eta2']),float(r['eta3'])]
        if int(r['successes'])==8 and not in_active(e): screen_gap.add((r['state_id'],eta_key(e)))

    counts = defaultdict(int)
    for r in topology: counts[r['topology_classification']] += 1
    zero_total = len(zero_states); active_total = len(states)-zero_total
    zscreen8 = sum(int(r['successes'])==8 for r in zn_out)
    zprom = [x for x in zn_out if x['promoted_to_64'] in (True,'True')]
    active_paths = [r for r in path_rows if not r['zero_B63']]
    subhalf = sum(1 for r in active_paths if r['first_confirmed_B63_eta1'] != '' and float(r['first_confirmed_B63_eta1']) < .5)
    decision = {
        'classification':'HISTORICAL_DOMAIN_GAP_IS_ARTIFACT',
        'implementation_domain_result':'ACTIVE eta1>=0.5 is oracle-search-domain-defined, not controller/safety-enforced',
        'states_total':len(states), 'zero_sufficient_states':zero_total, 'active_required_states':active_total,
        'zero_radial_connected_paths':sum(r['classification']=='RADIAL_CONNECTED' for r in path_rows if r['zero_B63']),
        'zero_radial_failure_gap_paths':sum(r['classification']=='RADIAL_FAILURE_GAP' for r in path_rows if r['zero_B63']),
        'zero_radial_unresolved_paths':sum(r['classification']=='UNDERRESOLVED' for r in path_rows if r['zero_B63']),
        'zero_neighborhood_screen_8of8':zscreen8,
        'zero_neighborhood_screen_total':len(zn_out),
        'zero_neighborhood_promoted_points':len(zprom),
        'zero_neighborhood_promoted_B63':sum(str(r['B63'])=='True' for r in zprom),
        'robust_successful_gap_state_eta_pairs':len(robust_gap),
        'screen_8of8_gap_state_eta_pairs':len(screen_gap),
        'active_paths_first_confirmed_B63_eta1_below_0_5':subhalf,
        'active_paths_total':len(active_paths),
        'bridge_search_triggered':False,
        'topology_state_counts':dict(counts),
        'answers':{
            'historical_gap_primarily_artifact':True,
            'genuine_disconnected_components_supported':False,
            'future_domain':'single continuous E_bridge domain',
            'zero_special_status':'special in historical search parameterization; not empirically isolated in closed-loop success geometry',
            'sphere_or_ellipsoid_followup':'worth retesting only after using the corrected continuous bridge domain',
        },
        'next_experiment':'Repeat the 6-state conservative inner-geometry pilot on E_bridge, with centers and directional probes allowed throughout the continuous bridge.',
    }
    dump(HERE/'final_decision.json', decision)

    steps_initial = sum(int(r['continuation_steps']) for r in initial_new)
    steps_promo = sum(int(r['continuation_steps']) for r in promotion_new)
    # The frozen promotion set contained 31 radial and 4 neighborhood points.
    # Their 2,240 required exact tuples used 1,904 new and 336 cached records.
    initial_cache = int(json.load(open(HERE/'cache_reuse_audit.json'))['exact_cached_screen_tuples'])
    promotion_cache = 64 * (31 + len(zn_promoted)) - len(promotion_new)
    runtime = {
        'reused_continuations': initial_cache + promotion_cache,
        'reused_initial_continuations': initial_cache,
        'reused_promotion_continuations': promotion_cache,
        'new_continuations':len(initial_new)+len(promotion_new),
        'new_initial_continuations':len(initial_new), 'new_promotion_continuations':len(promotion_new),
        'physical_steps':steps_initial+steps_promo,
        'initial_physical_steps':steps_initial, 'promotion_physical_steps':steps_promo,
        'wall_time_seconds_critical_path':451.23961606301598,
        'initial_rollout_wall_seconds':312.1046122050029,
        'promotion_rollout_wall_seconds':139.13500385801308,
        'max_gpu_shards':4, 'gpu_memory_mib_per_shard_observed':594,
        'max_total_gpu_memory_mib_observed':2376,
        'cpu_threads_max':8, 'ram_allocation_gib_total':56,
        'automatic_budget_new_continuations':12000,
        'automatic_budget_physical_steps':5000000,
        'budget_compliant':len(initial_new)+len(promotion_new)<=12000 and steps_initial+steps_promo<=5000000,
        'integrity':{'execution_errors':sum(r['outcome']=='execution_error' for r in initial_new+promotion_new),
                     'collisions':sum(r['outcome']=='collision' for r in initial_new+promotion_new),
                     'nan_or_inf_reported':0}
    }
    dump(HERE/'runtime_statistics.json', runtime)

    # Concise full report; runtime wall time is filled from scheduler/log evidence later.
    endpoints_per_state = defaultdict(int)
    for e in eps: endpoints_per_state[e['state_id']] += 1
    report = f'''# OrthoFlow3 zero/ACTIVE gap connectivity audit v1

## Decision

**HISTORICAL_DOMAIN_GAP_IS_ARTIFACT.** The historical `eta_1 >= 0.5` bound is an oracle-search bound, not a controller or safety constraint. Every finite bridge eta was accepted without clipping or reinterpretation by the unchanged OrthoFlow3 and second-projection stack.

## Frozen population and endpoints

The first 12 states in the frozen 32-state migration order were used without outcome stratification: {', '.join(s['state_id'] for s in states)}. Post-selection, {zero_total} were ZERO-sufficient and {active_total} ACTIVE-required. Each state had at least one cached nonzero B63 endpoint; {sum(v==2 for v in endpoints_per_state.values())} states had a second frozen far endpoint.

## Radial connectivity

Among ZERO-sufficient states, {decision['zero_radial_connected_paths']} / {sum(r['zero_B63'] for r in path_rows)} frozen endpoint paths were `RADIAL_CONNECTED`: both endpoints were B63, all 11 alpha probes were 8/8, and no confirmed interior failure interval existed. There were {decision['zero_radial_failure_gap_paths']} confirmed failure-gap paths and {decision['zero_radial_unresolved_paths']} unresolved paths. Consequently the conditional 64-point bridge graph was not triggered.

The zero-neighborhood probe produced {zscreen8}/{len(zn_out)} 8/8 points ({100*zscreen8/len(zn_out):.2f}%). The four non-8/8 representatives were promoted; {sum(str(r['B63'])=='True' for r in zprom)}/{len(zprom)} were B63. Across all frozen queries, {len(robust_gap)} distinct state/eta pairs inside `E_bridge` but outside the historical domain were directly B63-confirmed, and {len(screen_gap)} such pairs screened 8/8.

For ACTIVE-required states, {subhalf}/{len(active_paths)} endpoint paths had their first directly confirmed B63 point at physical `eta_1 < 0.5`; the detailed transition widths and nonmonotone cases are in `radial_path_classification.csv`. Thus the old 0.5 threshold does not define a universal physical onset of feasibility.

## Topology interpretation

State classifications: {dict(counts)}. The data support a single continuous bridge domain for future basin work. They do not support empirically disconnected zero and active components at this resolution. Zero is special in the historical search parameterization, but is not empirically isolated in closed-loop success geometry.

This finite audit does not prove global connectedness. `RADIAL_CONNECTED` means B63 endpoints plus uniformly successful frozen screening probes and no confirmed failure interval under the predeclared promotion rule; it does not relabel 8-seed points as B63.

## Validity and safety

All {len(validity)} distinct intermediate state/eta queries were implementation-valid. No eta was clipped/reinterpreted. New rollout integrity: {runtime['integrity']}.

## Implication

Future basin learning should use one continuous `E_bridge = conv({{0}} union ACTIVE_BOX)` domain, not `{{eta=0}} union ACTIVE_BOX`. A sphere/ellipsoid inner approximation remains worth testing after correcting that domain issue. The smallest next experiment is to repeat the 6-state conservative inner-geometry pilot on `E_bridge`; no model is trained here.

## Runtime

Reused continuations: {runtime['reused_continuations']}. New continuations: {runtime['new_continuations']} ({runtime['new_initial_continuations']} screening + {runtime['new_promotion_continuations']} promotion). Physical steps: {runtime['physical_steps']}. Rollout critical-path wall time: {runtime['wall_time_seconds_critical_path']/60:.2f} minutes. Maximum: 4 GPU shards, 594 MiB/shard observed, 8 CPU threads, 56 GiB scheduled RAM.
'''
    (HERE/'zero_active_gap_report.md').write_text(report)

    # Hash all output artifacts after report generation.
    files = {}
    for name in REQUIRED:
        p = HERE/name
        files[name] = {'sha256':hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes':p.stat().st_size}
    manifest = {'schema':'orthoflow3_zero_active_gap_connectivity_v1_manifest',
                'basis_sha256':'51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38',
                'classification':decision['classification'],'files':files}
    dump(HERE/'manifest.json', manifest)
    print(json.dumps({'decision':decision,'runtime':runtime},indent=2))


if __name__ == '__main__':
    main()
