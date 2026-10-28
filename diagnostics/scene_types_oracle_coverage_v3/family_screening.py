"""User-authorized heuristic futility gates; never an impossibility certificate.

Only orchestration reads this module. No controller or rollout semantics change.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY = {
    'version': 1, 'reference_family': 4, 'protected_families': [0, 4],
    'checkpoints': [16, 32], 'maximum_novel_successes_to_prune': 2,
    'coverage_deficit_fraction': {'16': .10, '32': .08},
    'maximum_16_to_32_gain_fraction': .03,
    'require_lower_coverage_on_both_controllers': True,
    'interpretation': 'Adaptive development screening, not independent validation or mathematical exclusion. Unrun candidates remain unknown.',
}

def key(g):
    return g['type_id'], g['level'], g['controller'], g['state_index']

def covered(g, k):
    return bool(g['success'] and g['first_success'] <= k)

def decision(members, reference, k):
    """Pure, matched-case audit at a fully observed candidate prefix."""
    refs = {key(g): g for g in reference}
    if not members or any(key(g) not in refs for g in members):
        return None
    if any(g.get('gap') or g.get('skipped') or
           not (g['success'] or g['searched'] >= k) for g in members):
        return None
    if any(not (refs[key(g)]['success'] or refs[key(g)]['searched'] == 64) for g in members):
        return None
    n = len(members)
    found = sum(covered(g, k) for g in members)
    ref_found = sum(covered(refs[key(g)], k) for g in members)
    novel = sum(covered(g, k) and not refs[key(g)]['success'] for g in members)
    gain = found - sum(covered(g, 16) for g in members)
    per_controller = {}
    for ctl in sorted({g['controller'] for g in members}):
        rows = [g for g in members if g['controller'] == ctl]
        per_controller[ctl] = dict(candidate=sum(covered(g, k) for g in rows),
                                  reference=sum(covered(refs[key(g)], k) for g in rows))
    prune = ((ref_found-found)/n >= POLICY['coverage_deficit_fraction'][str(k)]
             and novel <= POLICY['maximum_novel_successes_to_prune']
             and all(v['candidate'] < v['reference'] for v in per_controller.values())
             and (k == 16 or gain/n <= POLICY['maximum_16_to_32_gain_fraction']))
    return dict(checkpoint=k, cases=n, found=found, reference_found=ref_found,
                deficit_fraction=(ref_found-found)/n, novel_vs_reference64=novel,
                gain_since16=gain, per_controller=per_controller, prune=prune,
                evidence='Matched finite prefixes; heuristic resource allocation only.')

def review(fi, members, groups):
    if fi in POLICY['protected_families']:
        return False
    if any(g.get('screened_out') for g in members):
        return True
    path = HERE / f'family_{fi}_screening.json'
    history = json.loads(path.read_text()) if path.exists() else dict(family=fi, policy=POLICY, reviews=[])
    reference = [g for g in groups if g['family'] == 4]
    for k in POLICY['checkpoints']:
        if any(r['checkpoint'] == k for r in history['reviews']):
            continue
        result = decision(members, reference, k)
        if result is None:
            continue
        history['reviews'].append(result)
        if result['prune']:
            for g in members:
                if not g['success'] and g['searched'] < 64:
                    g['screened_out'] = dict(checkpoint=k, reason='heuristic_family_screening', audit=str(path))
            history['stopped'] = True
            history['unrun_candidates'] = sum(64-g['searched'] for g in members if g.get('screened_out'))
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(history, indent=2)+'\n'); tmp.replace(path)
        if result['prune']:
            return True
    return False
