#!/usr/bin/env python3
"""Offline PACT construction, cached selection, validation design, and t0 inventory."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, ConvexHull, distance_matrix

ROOT = Path('/home/zhihan/research/Basin_C1')
DIAG = ROOT / 'diagnostics'
HERE = DIAG / 'orthoflow3_t0_pact_training_readiness_v1'
SHAPE = DIAG / 'orthoflow3_t0_basin_shape_v1'
COMP = DIAG / 'orthoflow3_t0_basin_completion_v1'
MAN = DIAG / 'orthoflow3_b63_manifold_representation_v1'
MULTI = DIAG / 'orthoflow3_t0_multiball_basin_learning_v1'
MARGIN = DIAG / 'orthoflow3_basin_margin_learning_v1'
QV2 = DIAG / 'orthoflow3_q_learnability_v2'
DIRECTG = DIAG / 'direct_eta_basin_geometry_audit_v1'
BASIS = DIAG / 'double_bottleneck_eta_basis_redesign/tools/bases.py'

AFF = np.array([0.875, 0.0, 0.375])
SCALE = np.array([0.75, 1.0, 0.75])
BRIDGE_SCALES = (1.5, 2.0, 2.5, 3.0)
DELTAS = (0.0, 0.025, 0.05, 0.075, 0.10)
TOL = 2e-9


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(dict.fromkeys(k for r in rows for k in r)) if rows else ['status'])
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def nt(raw):
    return (np.asarray(raw, float) - AFF) / SCALE


def rt(norm):
    return AFF + SCALE * np.asarray(norm, float)


def eta_key(raw):
    return np.asarray(raw, dtype=np.float64).tobytes().hex()


def barycentric(point, tri):
    a, b, c = tri
    mat = np.column_stack((a - c, b - c))
    try:
        uv = np.linalg.solve(mat, point - c)
    except np.linalg.LinAlgError:
        return np.array([math.nan] * 3)
    return np.array([uv[0], uv[1], 1.0 - uv.sum()])


def point_in_triangle(point, tri, tol=TOL):
    lam = barycentric(point, tri)
    return bool(np.all(np.isfinite(lam)) and np.min(lam) >= -tol and np.max(lam) <= 1 + tol)


def cell_contains(s, nval, cell, tol=TOL, retained=False):
    tri = cell['S']
    lam = barycentric(s, tri)
    if not np.all(np.isfinite(lam)):
        return False
    if retained:
        # Shrinking the triangle toward its centroid by gamma maps the
        # barycentric lower bound to (1-gamma)/3.
        if np.min(lam) < (1.0 - 0.85) / 3.0 - tol:
            return False
        dm = 0.75 * cell['delta_minus']
        dp = 0.75 * cell['delta_plus']
    else:
        if np.min(lam) < -tol:
            return False
        dm, dp = cell['delta_minus'], cell['delta_plus']
    surface = float(lam @ cell['N'])
    off = nval - surface
    return -dm - tol <= off <= dp + tol


def pact_contains(s, nval, cells, retained=False):
    return any(cell_contains(s, nval, cell, retained=retained) for cell in cells)


def components(cells):
    if not cells:
        return 0
    edge_to_cells = defaultdict(list)
    for i, cell in enumerate(cells):
        v = cell['vertex_indices']
        for edge in ((v[0], v[1]), (v[1], v[2]), (v[0], v[2])):
            edge_to_cells[tuple(sorted(edge))].append(i)
    adj = defaultdict(set)
    for ids in edge_to_cells.values():
        for i in ids:
            adj[i].update(j for j in ids if j != i)
    unseen = set(range(len(cells)))
    count = 0
    while unseen:
        count += 1
        q = deque([unseen.pop()])
        while q:
            i = q.popleft()
            for j in adj[i]:
                if j in unseen:
                    unseen.remove(j)
                    q.append(j)
    return count


def build_scale(state_id, train_rows, neg_rows, indep_rows, frame, scale):
    center = np.array([float(frame[f'center_t{i}']) for i in (1, 2, 3)])
    U = np.array([[float(frame[f'u1_t{i}']) for i in (1, 2, 3)],
                  [float(frame[f'u2_t{i}']) for i in (1, 2, 3)]]).T
    normal = np.array([float(frame[f'normal_t{i}']) for i in (1, 2, 3)])

    # Exact tangential duplicate removal is deterministic.  It prevents a
    # Delaunay implementation detail from choosing between stacked vertices.
    unique = {}
    for row in sorted(train_rows, key=lambda r: r['eta_key_float64']):
        x = nt([row['eta1'], row['eta2'], row['eta3']])
        s = (x - center) @ U
        nval = float((x - center) @ normal)
        skey = tuple(np.round(s, 12))
        unique.setdefault(skey, (row, x, s, nval))
    verts = list(unique.values())
    S = np.array([v[2] for v in verts])
    if len(S) < 6:
        return [], {'state_id': state_id, 'bridge_scale': scale, 'eligible': False,
                    'reason': 'too_few_construction_vertices'}
    D = distance_matrix(S, S)
    np.fill_diagonal(D, np.inf)
    fifth = np.sort(D, axis=1)[:, min(4, len(S) - 2)]
    d5 = float(np.median(fifth))
    triang = Delaunay(S)
    neg = []
    for row in neg_rows:
        x = nt([row['eta1'], row['eta2'], row['eta3']])
        neg.append((row, (x - center) @ U, float((x - center) @ normal)))

    cells, candidate_rows = [], []
    for tid, simplex in enumerate(triang.simplices):
        simplex = tuple(int(i) for i in simplex)
        tri = S[list(simplex)]
        edge_lengths = [float(np.linalg.norm(tri[a] - tri[b])) for a, b in ((0, 1), (1, 2), (0, 2))]
        area = float(abs(np.cross(tri[1] - tri[0], tri[2] - tri[0])) / 2.0)
        N = np.array([verts[i][3] for i in simplex])
        reason = ''
        if max(edge_lengths) > scale * d5 + TOL:
            reason = 'edge_bridge_limit'
        offsets = []
        if not reason:
            for row, sf, nf in neg:
                if point_in_triangle(sf, tri):
                    lam = barycentric(sf, tri)
                    offsets.append((float(nf - lam @ N), row['eta_key_float64']))
            if any(abs(off) <= TOL for off, _ in offsets):
                reason = 'nonB63_on_surface'
        if reason:
            candidate_rows.append({'state_id': state_id, 'bridge_scale': scale,
                                   'triangle_id': tid, 'accepted': False,
                                   'rejection_reason': reason, 'max_edge': max(edge_lengths),
                                   'd5': d5, 'area': area,
                                   'vertex_keys': ';'.join(verts[i][0]['eta_key_float64'] for i in simplex)})
            continue
        plus = max(d for d in DELTAS if not any(-TOL <= off <= d + TOL for off, _ in offsets))
        minus = max(d for d in DELTAS if not any(-d - TOL <= off <= TOL for off, _ in offsets))
        cell = {'state_id': state_id, 'bridge_scale': scale, 'triangle_id': tid,
                'vertex_indices': simplex, 'vertex_keys': [verts[i][0]['eta_key_float64'] for i in simplex],
                'S': tri, 'N': N, 'delta_minus': float(minus), 'delta_plus': float(plus),
                'area': area, 'max_edge': max(edge_lengths), 'd5': d5,
                'center': center, 'U': U, 'normal': normal}
        cells.append(cell)
        candidate_rows.append({'state_id': state_id, 'bridge_scale': scale,
                               'triangle_id': tid, 'accepted': True, 'rejection_reason': '',
                               'max_edge': max(edge_lengths), 'd5': d5, 'area': area,
                               'delta_minus': minus, 'delta_plus': plus,
                               'vertex_keys': ';'.join(cell['vertex_keys'])})

    def recall(rows):
        if not rows:
            return math.nan, 0, 0
        hit = 0
        for row in rows:
            x = nt([row['eta1'], row['eta2'], row['eta3']])
            hit += pact_contains((x - center) @ U, float((x - center) @ normal), cells)
        return hit / len(rows), hit, len(rows)

    false_inclusions = 0
    for row, sf, nf in neg:
        false_inclusions += pact_contains(sf, nf, cells)
    indep_recall, indep_hit, indep_n = recall(indep_rows)
    all_success = train_rows + indep_rows
    all_recall, all_hit, all_n = recall(all_success)
    support = np.unique(np.concatenate([c['S'] for c in cells], axis=0), axis=0) if cells else np.empty((0, 2))
    eig_ratio, hull_area, diameter = 0.0, 0.0, 0.0
    if len(support) >= 3:
        eig = np.sort(np.linalg.eigvalsh(np.cov(support.T)))[::-1]
        eig_ratio = float(eig[1] / max(eig[0], 1e-12))
        try:
            hull_area = float(ConvexHull(support).volume)
        except Exception:
            hull_area = 0.0
        diameter = float(distance_matrix(support, support).max())
    total_area = float(sum(c['area'] for c in cells))
    nonzero_fraction = float(np.mean([(c['delta_minus'] > 0) or (c['delta_plus'] > 0) for c in cells])) if cells else 0.0
    metric = {'state_id': state_id, 'bridge_scale': scale, 'eligible': false_inclusions == 0,
              'cached_false_inclusions': false_inclusions, 'triangle_count': len(cells),
              'candidate_triangle_count': len(triang.simplices), 'components': components(cells),
              'd5': d5, 'L_max': scale * d5, 'independent_B63_recall': indep_recall,
              'independent_B63_hit': indep_hit, 'independent_B63_total': indep_n,
              'all_B63_recall': all_recall, 'all_B63_hit': all_hit, 'all_B63_total': all_n,
              'total_triangle_area': total_area, 'support_hull_area': hull_area,
              'tangential_diameter': diameter, 'tangential_eigen_ratio': eig_ratio,
              'nonzero_thickness_cell_fraction': nonzero_fraction,
              'median_delta_minus': float(np.median([c['delta_minus'] for c in cells])) if cells else 0.0,
              'median_delta_plus': float(np.median([c['delta_plus'] for c in cells])) if cells else 0.0,
              'max_delta_minus': max([c['delta_minus'] for c in cells], default=0.0),
              'max_delta_plus': max([c['delta_plus'] for c in cells], default=0.0)}
    return cells, metric, candidate_rows


def selected_validation_points(state_id, cells, rejected, eq, existing_keys):
    if not cells:
        return []
    center = cells[0]['center']; U = cells[0]['U']; normal = cells[0]['normal']

    def raw_point(cell, lam, off):
        s = np.asarray(lam) @ cell['S']
        nval = float(np.asarray(lam) @ cell['N'] + off)
        x = center + U @ s + normal * nval
        return rt(x), s, nval

    def in_domain(raw):
        x = nt(raw)
        return bool(np.all(eq[:, :3] @ x + eq[:, 3] <= 2e-9))

    rows, seen = [], set(existing_keys)
    mean_s = np.mean(np.concatenate([c['S'] for c in cells], axis=0), axis=0)
    central = sorted(cells, key=lambda c: (float(np.linalg.norm(c['S'].mean(axis=0) - mean_s)), c['triangle_id']))

    def add(category, cell, lam, off, relation):
        raw, s, nval = raw_point(cell, lam, off)
        key = eta_key(raw)
        if key in seen or not in_domain(raw):
            return False
        actual = pact_contains(s, nval, cells)
        if relation == 'inside' and not actual:
            return False
        if relation != 'inside' and actual:
            return False
        seen.add(key)
        rows.append({'state_id': state_id, 'probe_id': '', 'category': category,
                     'relation_to_pact': relation, 'cell_triangle_id': cell['triangle_id'],
                     'eta1': raw[0], 'eta2': raw[1], 'eta3': raw[2],
                     'eta_key_float64': key, 'bary_a': lam[0], 'bary_b': lam[1], 'bary_c': lam[2],
                     'normal_offset': off, 'new_continuations_needed': 64})
        return True

    # Two central interior points.
    for cell, lam in zip(central, ([1/3, 1/3, 1/3], [0.4, 0.3, 0.3])):
        if len([r for r in rows if r['category'] == 'INSIDE_CENTRAL']) >= 2:
            break
        add('INSIDE_CENTRAL', cell, lam, 0.0, 'inside')
    # Two near tangential boundaries.  Small positive barycentric weights keep
    # them strictly inside the selected triangle.
    edge_count = Counter()
    owner = {}
    for cell in cells:
        v = cell['vertex_indices']
        for local, edge in enumerate(((v[0], v[1]), (v[1], v[2]), (v[0], v[2]))):
            e = tuple(sorted(edge)); edge_count[e] += 1; owner.setdefault(e, (cell, local))
    for edge in sorted((e for e, n in edge_count.items() if n == 1)):
        cell, local = owner[edge]
        lam = ([0.49, 0.49, 0.02], [0.02, 0.49, 0.49], [0.49, 0.02, 0.49])[local]
        add('INSIDE_TANGENTIAL_BOUNDARY', cell, lam, 0.0, 'inside')
        if len([r for r in rows if r['category'] == 'INSIDE_TANGENTIAL_BOUNDARY']) == 2:
            break
    # Two near normal boundaries, preferring opposite sides.
    for side in ('plus', 'minus'):
        ordered = sorted(cells, key=lambda c: (-(c['delta_plus'] if side == 'plus' else c['delta_minus']), c['triangle_id']))
        for cell in ordered:
            delta = cell['delta_plus'] if side == 'plus' else cell['delta_minus']
            if delta <= 0:
                continue
            off = 0.95 * delta * (1 if side == 'plus' else -1)
            if add('INSIDE_NORMAL_BOUNDARY', cell, [1/3, 1/3, 1/3], off, 'inside'):
                break
    # If thickness is one-sided/zero, keep six inside tests by using distinct
    # strict tangential interiors on the center surface.
    for cell in central:
        for lam in ([0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]):
            if len([r for r in rows if r['relation_to_pact'] == 'inside']) >= 6:
                break
            add('INSIDE_NORMAL_BOUNDARY', cell, lam, 0.0, 'inside')
        if len([r for r in rows if r['relation_to_pact'] == 'inside']) >= 6:
            break
    # Three immediately outside the local normal envelope.
    for cell in sorted(cells, key=lambda c: (-max(c['delta_plus'], c['delta_minus']), c['triangle_id'])):
        for sign, delta in ((1, cell['delta_plus']), (-1, cell['delta_minus'])):
            if len([r for r in rows if r['category'] == 'OUTSIDE_NORMAL']) >= 3:
                break
            add('OUTSIDE_NORMAL', cell, [1/3, 1/3, 1/3], sign * (delta + 0.0125), 'outside_normal')
        if len([r for r in rows if r['category'] == 'OUTSIDE_NORMAL']) >= 3:
            break
    # Three omitted-region points from rejected Delaunay triangles.  Their
    # surface is defined only for probe placement, never added to PACT.
    by_key = {c['triangle_id']: c for c in cells}
    template = cells[0]
    for rej in sorted(rejected, key=lambda r: (0 if r['rejection_reason'] == 'nonB63_on_surface' else 1,
                                               -float(r['area']), int(r['triangle_id']))):
        if len([r for r in rows if r['category'] == 'OUTSIDE_TANGENTIAL_GAP']) >= 3:
            break
        # Recover the rejected triangle from vertex keys using cached full
        # coordinates embedded in the caller-provided temporary fields.
        if '_S' not in rej:
            continue
        pseudo = dict(template)
        pseudo['triangle_id'] = int(rej['triangle_id'])
        pseudo['S'] = rej['_S']; pseudo['N'] = rej['_N']
        pseudo['delta_plus'] = pseudo['delta_minus'] = 0.0
        add('OUTSIDE_TANGENTIAL_GAP', pseudo, [1/3, 1/3, 1/3], 0.0, 'outside_tangential')
    for i, row in enumerate(rows):
        row['probe_id'] = f'{state_id}__PACT{i:02d}'
    return rows[:12]


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    if sha(BASIS) != '51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38':
        raise RuntimeError('authoritative basis hash mismatch')
    states = json.load(open(COMP / 'frozen_8state_manifest.json'))['attempted_states']
    frames = {r['state_id']: r for r in read_csv(MAN / 'affine_plane_fit.csv')}
    exact = [r for r in read_csv(SHAPE / 'cached_q64_manifest.csv') if r['Q64_available'] == 'True']
    if len(exact) != 877:
        raise RuntimeError(f'unexpected exact Q64 total {len(exact)}')
    write_csv(HERE / 'exact_q64_manifest.csv', exact)
    eq = np.asarray(json.load(open(MULTI / 'geometry_constants.json'))['halfspaces'], float)

    all_candidate_rows, all_metric_rows = [], []
    selected_cells, selected_metrics, rejected_by_state = {}, {}, {}
    for st in states:
        sid = st['state_id']; rows = [r for r in exact if r['state_id'] == sid]
        indep = [r for r in rows if r['B63'] == 'True' and 'coverage64' in r['phases']]
        train = [r for r in rows if r['B63'] == 'True' and 'coverage64' not in r['phases']]
        neg = [r for r in rows if r['B63'] != 'True']
        builds = []
        for scale in BRIDGE_SCALES:
            result = build_scale(sid, train, neg, indep, frames[sid], scale)
            cells, metric, candidates = result
            all_metric_rows.append(metric); all_candidate_rows.extend(candidates)
            builds.append((cells, metric, candidates))
        eligible = [b for b in builds if b[1].get('eligible')]
        if not eligible:
            selected_cells[sid] = []; selected_metrics[sid] = {'state_id': sid, 'selected': False}
            rejected_by_state[sid] = []
            continue
        eligible.sort(key=lambda b: (-float(b[1]['independent_B63_recall']),
                                     float(b[1]['bridge_scale']), int(b[1]['triangle_count'])))
        cells, metric, candidates = eligible[0]
        metric = dict(metric); metric['selected'] = True
        two_d = metric['tangential_eigen_ratio'] >= 0.10 and metric['support_hull_area'] >= 2 * metric['d5'] ** 2
        broad = metric['support_hull_area'] >= 4 * metric['d5'] ** 2 and metric['tangential_diameter'] >= 4 * metric['d5']
        nontrivial = metric['nonzero_thickness_cell_fraction'] > 0 or broad
        metric['genuinely_2d'] = two_d; metric['broad_tangential_support'] = broad
        metric['shape_usable_before_fresh'] = bool(metric['cached_false_inclusions'] == 0 and
                                                   metric['independent_B63_recall'] >= 0.30 and
                                                   two_d and nontrivial)
        selected_cells[sid] = cells; selected_metrics[sid] = metric
        rejected_by_state[sid] = [r for r in candidates if not r['accepted']]

    write_csv(HERE / 'pact_candidates.csv', all_candidate_rows)
    write_csv(HERE / 'pact_cached_metrics.csv', all_metric_rows + list(selected_metrics.values()))
    cell_rows = []
    for sid, cells in selected_cells.items():
        for cell in cells:
            row = {k: v for k, v in cell.items() if k not in ('S', 'N', 'center', 'U', 'normal', 'vertex_indices', 'vertex_keys')}
            row.update({'vertex_indices': ';'.join(map(str, cell['vertex_indices'])),
                        'vertex_keys': ';'.join(cell['vertex_keys']),
                        's_vertices': json.dumps(cell['S'].tolist(), separators=(',', ':')),
                        'n_vertices': json.dumps(cell['N'].tolist(), separators=(',', ':'))})
            cell_rows.append(row)
    write_csv(HERE / 'pact_cells.csv', cell_rows)

    # Attach rejected triangle geometry for deterministic gap placement.
    for sid, rejected in rejected_by_state.items():
        frame = frames[sid]
        center = np.array([float(frame[f'center_t{i}']) for i in (1, 2, 3)])
        U = np.array([[float(frame[f'u1_t{i}']) for i in (1, 2, 3)],
                      [float(frame[f'u2_t{i}']) for i in (1, 2, 3)]]).T
        normal = np.array([float(frame[f'normal_t{i}']) for i in (1, 2, 3)])
        key_to_row = {r['eta_key_float64']: r for r in exact if r['state_id'] == sid}
        for rej in rejected:
            keys = rej['vertex_keys'].split(';')
            pts = [nt([key_to_row[k]['eta1'], key_to_row[k]['eta2'], key_to_row[k]['eta3']]) for k in keys]
            rej['_S'] = np.array([(x - center) @ U for x in pts])
            rej['_N'] = np.array([float((x - center) @ normal) for x in pts])

    validation = []
    existing_keys = defaultdict(set)
    for row in exact:
        existing_keys[row['state_id']].add(row['eta_key_float64'])
    for st in states:
        sid = st['state_id']
        if selected_metrics[sid].get('shape_usable_before_fresh'):
            validation += selected_validation_points(sid, selected_cells[sid], rejected_by_state[sid], eq, existing_keys[sid])
    write_csv(HERE / 'pact_validation_manifest.csv', validation)
    for st in states:
        sid = st['state_id']
        dump(HERE / f'validation_targets_{sid}.json', [r for r in validation if r['state_id'] == sid])

    # Repository-wide true-t0 inventory: 300 exact h0 vectors from the frozen
    # fresh-WIDE cohort plus any compatible OrthoFlow3 Q-v2 t0 states.
    fresh = {}
    for path in sorted((MARGIN / 'raw/fresh_wide').glob('shard*.jsonl')):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line); sid = row['source_id']
            item = fresh.setdefault(sid, {'state_id': sid, 'source_group': sid,
                                          'source_trajectory': sid, 'true_t0': True,
                                          'h0_available': bool(row.get('h0')),
                                          'h0_dim': len(row.get('h0') or []),
                                          'feature_sha256': row.get('feature_sha256', ''),
                                          'provenance': 'orthoflow3_basin_margin_learning_v1/fresh_wide',
                                          'single_episode_controller_records': 0})
            item['single_episode_controller_records'] += 1
            if row.get('h0'):
                item['h0_available'] = True
                item['h0_dim'] = len(row['h0'])
                item['feature_sha256'] = row.get('feature_sha256', item.get('feature_sha256', ''))
    fixed_by_group = {st['source_group']: st for st in states}
    exact_by_state = defaultdict(list)
    for row in exact:
        exact_by_state[row['state_id']].append(row)
    inventory = []
    for sid, item in sorted(fresh.items()):
        group_match = next((st for group, st in fixed_by_group.items() if group == sid), None)
        if group_match:
            qrows = exact_by_state[group_match['state_id']]
            item.update({'audit_state_id': group_match['state_id'],
                         'h_conditioning_identifier': group_match['h_conditioning_identifier'],
                         'exact_q64_eta': len(qrows),
                         'confirmed_B63_eta': sum(r['B63'] == 'True' for r in qrows),
                         'confirmed_nonB63_eta': sum(r['B63'] != 'True' for r in qrows),
                         'verified_ball': True, 'pact_compatible_cloud': True,
                         'robust_center_target': True})
        else:
            item.update({'audit_state_id': '', 'h_conditioning_identifier': '',
                         'exact_q64_eta': 0, 'confirmed_B63_eta': 0,
                         'confirmed_nonB63_eta': 0, 'verified_ball': False,
                         'pact_compatible_cloud': False, 'robust_center_target': False})
        inventory.append(item)
    qmanifest = json.load(open(QV2 / 'eligible_state_manifest.json'))
    for st in qmanifest['selected_states']:
        if int(st.get('absolute_step', -1)) != 0:
            continue
        if any(r['feature_sha256'] == st['feature_sha256'] for r in inventory):
            continue
        inventory.append({'state_id': st['state_id'], 'source_group': st['source_group'],
                          'source_trajectory': st['source_trajectory'], 'true_t0': True,
                          'h0_available': True, 'h0_dim': 214,
                          'feature_sha256': st['feature_sha256'],
                          'h_conditioning_identifier': st['h_conditioning_identifier'],
                          'provenance': 'orthoflow3_q_learnability_v2',
                          'single_episode_controller_records': 0,
                          'exact_q64_eta': 0, 'confirmed_B63_eta': 0,
                          'confirmed_nonB63_eta': 0, 'verified_ball': False,
                          'pact_compatible_cloud': False, 'robust_center_target': False,
                          'note': 'only lower-seed eta evidence; no exact B63'})
    # Older startup/deadlock manifests contain additional true-t0 state
    # snapshots.  Their historical eta labels use an incompatible
    # representation, so they are inventoried but never promoted to current
    # OrthoFlow3 supervision.  S_r000_p00 is already represented by Q-v2.
    historical = json.load(open(DIRECTG / 'dataset_424_manifest.json'))['states']
    for st in historical:
        if int(st.get('absolute_step', -1)) != 0:
            continue
        prior = next((r for r in inventory if r['state_id'] == st['state_id'] and
                      r.get('source_group') == st.get('leakage_group')), None)
        if prior:
            prior['provenance'] += ';direct_eta_basin_geometry_audit_v1'
            continue
        inventory.append({'state_id': st['state_id'],
                          'source_group': st.get('leakage_group', st['state_id']),
                          'source_trajectory': st.get('source_trajectory', ''),
                          'true_t0': True, 'h0_available': False, 'h0_dim': 0,
                          'feature_sha256': '', 'h_conditioning_identifier': '',
                          'provenance': 'direct_eta_basin_geometry_audit_v1',
                          'single_episode_controller_records': 0,
                          'exact_q64_eta': 0, 'confirmed_B63_eta': 0,
                          'confirmed_nonB63_eta': 0, 'verified_ball': False,
                          'pact_compatible_cloud': False, 'robust_center_target': False,
                          'note': 'true-t0 state snapshot; historical eta representation incompatible and no frozen current h0/Q64 label'})
    write_csv(HERE / 'unique_t0_state_inventory.csv', inventory)
    dump(HERE / 'state_deduplication.json', {
        'unique_compatible_true_t0_states': len(inventory),
        'fresh_wide_states': len(fresh),
        'additional_nonfresh_t0_records': len(inventory) - len(fresh),
        'dedup_keys': ['feature_sha256', 'h_conditioning_identifier', 'source_group'],
        'state_level_rule': 'many eta evaluations for one h0 count as one state',
        'exact_q64_state_count': sum(int(r['exact_q64_eta']) > 0 for r in inventory),
    })

    # Pre-fresh readiness candidates.  Final set usability is updated after
    # validation; point usability is already authoritative.
    split_map = {}
    fixed_ids = [st['state_id'] for st in states]
    for i, sid in enumerate(fixed_ids):
        split_map[sid] = 'train' if i < 5 else ('val' if i == 5 else 'test')
    set_rows, point_rows = [], []
    for st in states:
        sid = st['state_id']; metric = selected_metrics[sid]
        set_rows.append({'state_id': sid, 'source_group': st['source_group'],
                         'split': split_map[sid], 'shape_usable_before_fresh': metric.get('shape_usable_before_fresh', False),
                         'verified_pact_usable': 'PENDING_FRESH' if metric.get('shape_usable_before_fresh') else False,
                         'selected_bridge_scale': metric.get('bridge_scale', ''),
                         'independent_B63_recall': metric.get('independent_B63_recall', '')})
        ball = next((r for r in read_csv(COMP / 'completed_t0_balls.csv') if r['state_id'] == sid), None)
        point_rows.append({'state_id': sid, 'source_group': st['source_group'], 'split': split_map[sid],
                           'point_usable': True, 'target_priority': 1,
                           'target_kind': 'existing_independently_verified_robust_center',
                           'target_eta1': ball['c1'], 'target_eta2': ball['c2'],
                           'target_eta3': ball['c3'], 'center_Q64': ball['center_Q64']})
    write_csv(HERE / 'set_dataset_candidates.csv', set_rows)
    write_csv(HERE / 'point_dataset_candidates.csv', point_rows)
    point_counts = Counter(r['split'] for r in point_rows if r['point_usable'])
    dump(HERE / 'point_dataset_gate.json', {
        'pass': False, 'counts': dict(point_counts), 'required': {'train': 24, 'val': 8, 'test': 8},
        'missing': {k: max(0, v - point_counts[k]) for k, v in {'train': 24, 'val': 8, 'test': 8}.items()},
        'reason': 'only eight unique true-t0 states have defensible exact-B63 point targets'})
    dump(HERE / 'set_dataset_gate.json', {'pass': False, 'status': 'PENDING_PACT_FRESH_VALIDATION',
                                          'required': {'train': 24, 'val': 8, 'test': 8}})
    dump(HERE / 'pact_preflight.json', {
        'exact_q64_eta': len(exact), 'B63_eta': sum(r['B63'] == 'True' for r in exact),
        'nonB63_eta': sum(r['B63'] != 'True' for r in exact),
        'shape_usable_before_fresh': sum(m.get('shape_usable_before_fresh', False) for m in selected_metrics.values()),
        'validation_eta': len(validation), 'projected_new_continuations': 64 * len(validation),
        'max_eta_per_state': max(Counter(r['state_id'] for r in validation).values(), default=0)})
    (HERE / 'protocol.md').write_text(
        '# PACT encoding and true-t0 training readiness v1\n\n'
        'The authoritative exact-Q64 table is frozen before validation. Independent coverage64/Sobol B63 points are excluded from triangulation and used as held-out cached recall. '
        'For each bridge scale in {1.5,2.0,2.5,3.0} times d5, Delaunay triangles exceeding the edge limit or containing exact non-B63 evidence on their affine surface are rejected. '
        'Positive and negative thickness are selected independently from {0,.025,.05,.075,.10} subject to zero cached false inclusions. '
        'Selection maximizes independent recall, then prefers smaller bridge scale and fewer triangles. Shape usability requires recall>=.30, nondegenerate 2-D support, and nonzero thickness or broad tangential support. '
        'Fresh validation is at most 12 eta/state and all are evaluated directly at Q64. Independent state-level readiness counts unique h0, never eta rows.\n')
    print(json.dumps(json.load(open(HERE / 'pact_preflight.json')), indent=2))


if __name__ == '__main__':
    main()
