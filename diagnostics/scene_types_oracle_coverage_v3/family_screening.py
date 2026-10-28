"""User-authorized heuristic futility gates; never an impossibility certificate.

Only orchestration reads this module. No controller or rollout semantics change.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY = {
    'version': 3, 'reference_selection': 'best fully observed matched prefix', 'protected_families': [0, 4],
    'checkpoints': [16, 32],
    'coverage_deficit_fraction': {'16': .10, '32': .08},
    'require_lower_coverage_on_both_controllers': True,
    'specialist_other_max_coverage': .25,
    'specialist_min_coverage': .50,
    'specialist_scope': 'each scene pooled and separately per nominal controller',
    'interpretation': 'Adaptive development screening, not independent validation or mathematical exclusion. Unrun candidates remain unknown.',
}

def key(g):
    return g['type_id'], g['level'], g['controller'], g['state_index']

def covered(g, k):
    return bool(g['success'] and g['first_success'] <= k)

def specialists(fi, members, groups, k):
    """Keep a strong hard-scene specialist, even with poor aggregate coverage.

    Unknown prefixes are not failures. Known successes from incomplete rivals
    can disprove that all others are poor, but cannot establish that claim.
    """
    hits = []
    for scene in sorted({(g['type_id'], g['level']) for g in members}):
        scene_rows = [g for g in members if (g['type_id'], g['level']) == scene]
        for ctl in [None] + sorted({g['controller'] for g in scene_rows}):
            rows = [g for g in scene_rows if ctl is None or g['controller'] == ctl]
            if not rows or any(g.get('gap') or g.get('skipped') or
                               not (g['success'] or g['searched'] >= k) for g in rows):
                continue
            identities = {key(g) for g in rows}
            rival_counts = {}; complete = []
            for rival in sorted({g['family'] for g in groups} - {fi}):
                rr = [g for g in groups if g['family'] == rival and key(g) in identities]
                if len(rr) != len(rows):
                    continue
                rival_counts[str(rival)] = sum(covered(g, k) for g in rr)
                if all(not g.get('gap') and not g.get('skipped') and
                       (g['success'] or g['searched'] >= k) for g in rr):
                    complete.append(rival)
            best = max(rival_counts.values(), default=0)
            found = sum(covered(g, k) for g in rows)
            if 4 in complete and best/len(rows) <= POLICY['specialist_other_max_coverage'] and found/len(rows) >= POLICY['specialist_min_coverage']:
                hits.append(dict(type_id=scene[0], level=scene[1], controller=ctl,
                                 cases=len(rows), found=found, other_best=best,
                                 rival_counts=rival_counts, completed_rivals=complete,
                                 successful_cases=[key(g) for g in rows if covered(g, k)],
                                 note='Only observed evidence; untested rivals remain unknown.'))
    return hits

def decision(members, reference, k):
    """Pure, matched-case audit at a fully observed candidate prefix."""
    refs = {key(g): g for g in reference}
    if not members or any(key(g) not in refs for g in members):
        return None
    if any(g.get('gap') or g.get('skipped') or
           not (g['success'] or g['searched'] >= k) for g in members):
        return None
    if any(refs[key(g)].get('gap') or refs[key(g)].get('skipped') or
           not (refs[key(g)]['success'] or refs[key(g)]['searched'] >= k) for g in members):
        return None
    n = len(members)
    found = sum(covered(g, k) for g in members)
    ref_found = sum(covered(refs[key(g)], k) for g in members)
    novel = sum(covered(g, k) and not covered(refs[key(g)], k) for g in members)
    gain = found - sum(covered(g, 16) for g in members)
    per_controller = {}
    for ctl in sorted({g['controller'] for g in members}):
        rows = [g for g in members if g['controller'] == ctl]
        per_controller[ctl] = dict(candidate=sum(covered(g, k) for g in rows),
                                  reference=sum(covered(refs[key(g)], k) for g in rows))
    prune = ((ref_found-found)/n >= POLICY['coverage_deficit_fraction'][str(k)]
             and all(v['candidate'] < v['reference'] for v in per_controller.values()))
    return dict(checkpoint=k, cases=n, found=found, reference_found=ref_found,
                deficit_fraction=(ref_found-found)/n, novel_vs_reference_same_budget=novel,
                gain_since16=gain, per_controller=per_controller, prune=prune,
                evidence='Matched finite prefixes; heuristic resource allocation only.')

def review(fi, members, groups):
    if fi in POLICY['protected_families']:
        return False
    if any(g.get('screened_out') for g in members):
        return True
    path = HERE / f'family_{fi}_screening.json'
    history = json.loads(path.read_text()) if path.exists() else dict(family=fi, policy=POLICY, reviews=[])
    if history.get('specialist_protected'):
        return False
    for k in POLICY['checkpoints']:
        if any(r['checkpoint'] == k for r in history['reviews']):
            continue
        alternatives = []
        for candidate in sorted({g['family'] for g in groups} - {fi}):
            result = decision(members, [g for g in groups if g['family'] == candidate], k)
            if result is not None:
                result['reference_family'] = candidate
                alternatives.append(result)
        if not alternatives:
            continue
        dimensions = [3, 1, 2, 3, 4, 5, 2]
        result = max(alternatives, key=lambda r: (r['reference_found'], -dimensions[r['reference_family']]))
        result['specialist_scenes'] = specialists(fi, members, groups, k)
        if result['specialist_scenes']:
            result['aggregate_prune_before_specialist_exception'] = result['prune']
            result['prune'] = False
            history['specialist_protected'] = True
            history['protection_budget'] = 64
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
        if history.get('specialist_protected'):
            return False
    return False
