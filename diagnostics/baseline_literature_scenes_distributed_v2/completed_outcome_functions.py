"""Outcome-only derivatives of the immutable main collector and summary.

No missing diagnosis is imputed. Hierarchy, paired seeds and bootstrap stay exact.
"""
from experiment import *
from result_inventory import find_raw, qualification
import numpy as np
import collections
import importlib.util
_spec = importlib.util.spec_from_file_location('sealed_v2_outcome_statistics', HERE / 'statistics.py')
_stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stats)
bootstrap = _stats.bootstrap
contrast = _stats.contrast
FIXED = _stats.FIXED
LAYERS = _stats.LAYERS

def collect_completed_outcomes(stages):
    records = []
    sources = []
    for stage in stages:
        for c in read(HERE / f'{stage}_CASES.json'):
            path = find_raw(c['rollout_uid'], stage, c['state']['scene'])
            raw = read(path)
            ctl = c['controller_payload']['nominal']
            s = c['state']
            records.append(dict(layer=c.get('safety_layer', 'central'), scene=s['scene'], N=s['N'], state=s['index'], controller=ctl['controller'], type=ctl['type_code'], method=c['method'], seed=c['seed'], success=raw['success'], upper=min(1, raw['success'] + raw['numerical_failure']), collision=raw['collision'], timeout=raw['timeout'], unknown=raw['numerical_failure'], rollout_uid=c['rollout_uid']))
            sources.append(dict(rollout_uid=c['rollout_uid'], logical_stage=stage, raw=str(path), raw_sha256=sha(path)))
    return (records, sources)

def summarize_outcomes(records, arms, scope, remove_failures=False, census=()):
    q = {s: {r['controller'] for r in qualification(s) if r['qualified']} for s in NAMES}
    failures = {(r['scene'], r['layer'], r['controller']) for r in census if r['controller_failure_definite']}
    lookup = {}
    typ = {}
    for r in records:
        if r['scene'] not in scope or r['controller'] not in q[r['scene']] or (r['layer'], r['method']) not in arms:
            continue
        if remove_failures and (r['scene'], r['layer'], r['controller']) in failures:
            continue
        typ[r['controller']] = r['type']
        lookup[r['controller'], r['scene'], r['N'], r['state'], r['seed'], (r['layer'], r['method'])] = r
    ctls = sorted(typ)
    states = sorted({(key[1], key[2], key[3]) for key in lookup})
    M = np.full((len(ctls), len(states), 2 * len(arms)), np.nan)
    metrics = {}
    counts = {str(a): collections.Counter() for a in arms}
    harm = {}
    for ci, ctl in enumerate(ctls):
        for si, state in enumerate(states):
            rows = [[lookup.get((ctl, *state, seed, arm)) for seed in (0, 1)] for arm in arms]
            if any((r is None for rr in rows for r in rr)):
                continue
            for ai, (arm, rr) in enumerate(zip(arms, rows)):
                M[ci, si, 2 * ai] = np.mean([r['success'] for r in rr])
                M[ci, si, 2 * ai + 1] = np.mean([r['upper'] for r in rr])
                for r in rr:
                    counts[str(arm)].update({k: r[k] for k in ('success', 'collision', 'timeout', 'unknown')})
                    counts[str(arm)]['episodes'] += 1
                    counts[str(arm)]['agents'] += r['N']
            for ai, arm in enumerate(arms):
                if arm[1] == 'safety' or (arm[0], 'safety') not in arms:
                    continue
                base = rows[arms.index((arm[0], 'safety'))]
                h = harm.setdefault(str(arm), dict(known=0, upper_including_unknown=0))
                h['known'] += sum((a['success'] and (not b['success']) and (not b['unknown']) for a, b in zip(base, rows[ai])))
                h['upper_including_unknown'] += sum((a['success'] and (not b['success']) for a, b in zip(base, rows[ai])))
    supported = np.isfinite(M).all(-1).any(-1)
    M = M[supported]
    ctls = [c for c, ok in zip(ctls, supported) if ok]
    if not ctls:
        return dict(status='N/A', reason='No complete qualified paired state clusters')
    types = np.asarray([typ[c] for c in ctls])
    tables = []
    comparisons = []
    for label in [*sorted(set(types)), 'type_equal']:
        mask = np.ones(len(types), bool) if label == 'type_equal' else types == label
        point, draws = bootstrap(M[mask], types[mask])
        for ai, arm in enumerate(arms):
            row = dict(type=label, layer=arm[0], method=arm[1], success_rate=float(point[2 * ai]), ci95=np.quantile(draws[:, 2 * ai], [0.025, 0.975]).tolist(), success_upper=float(point[2 * ai + 1]), upper_ci95=np.quantile(draws[:, 2 * ai + 1], [0.025, 0.975]).tolist(), controllers=int(mask.sum()))
            base = (arm[0], 'safety')
            if arm != base and base in arms:
                row['delta_vs_safety'] = contrast(point, draws, arms, {arm: 1, base: -1})
            if arm[1] == 'learned_scene':
                fixes = [(arm[0], m) for m in FIXED[1:] if (arm[0], m) in arms]
                row['delta_vs_fixed'] = {a[1]: contrast(point, draws, arms, {arm: 1, a: -1}) for a in fixes}
                if fixes:
                    ix = [2 * arms.index(a) for a in fixes]
                    i = 2 * ai
                    delta = draws[:, i] - draws[:, ix].max(1)
                    row['delta_vs_best_aggregate_fixed'] = dict(delta=float(point[i] - point[ix].max()), ci95=np.quantile(delta, [0.025, 0.975]).tolist(), bootstrap_reselects_best_fixed=True, identified_bounds=[float(point[i] - point[np.asarray(ix) + 1].max()), float(point[i + 1] - point[ix].max())])
            tables.append(row)
        for m in sorted({a[1] for a in arms}):
            C = ('central', m)
            D = ('distributed', m)
            if C in arms and D in arms:
                item = dict(type=label, method=m, distributed_minus_central=contrast(point, draws, arms, {D: 1, C: -1}))
                if m != 'safety' and ('central', 'safety') in arms and (('distributed', 'safety') in arms):
                    item['correction_gain_difference'] = contrast(point, draws, arms, {D: 1, ('distributed', 'safety'): -1, C: -1, ('central', 'safety'): 1})
                comparisons.append(item)
    drops = {}
    for arm in arms:
        base = (arm[0], 'safety')
        if arm == base or base not in arms:
            continue
        i = 2 * arms.index(arm)
        j = 2 * arms.index(base)
        lows = np.nanmean(M[..., i], 1)
        ups = np.nanmean(M[..., i + 1], 1)
        blo = np.nanmean(M[..., j], 1)
        bup = np.nanmean(M[..., j + 1], 1)
        drops[str(arm)] = dict(definite_10pp_drop=[c for c, l, u, b in zip(ctls, lows, ups, blo) if u - b <= -0.1 + 1e-12], possible_10pp_drop=[c for c, l, b in zip(ctls, lows, bup) if l - b <= -0.1 + 1e-12])
    return dict(status='complete', statistics=tables, layer_comparisons=comparisons, pooled_counts={k: dict(v) for k, v in counts.items()}, harms_vs_safety=harm, controller_10pp_drops=drops, controllers=ctls, state_clusters=states, removed_definite_controller_failures=remove_failures, scope=scope, bootstrap_iterations=5000, weighting='Equal type, equal controller within type, equal available (scene,N,index) state within controller; seeds and all arms paired')
