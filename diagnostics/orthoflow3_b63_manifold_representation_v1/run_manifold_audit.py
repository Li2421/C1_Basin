#!/usr/bin/env python3
"""Strictly offline low-complexity representation comparison for existing B63 clouds."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, Delaunay


ROOT = Path('/home/zhihan/research/Basin_C1')
DIAG = ROOT / 'diagnostics'
SRC = DIAG / 'orthoflow3_existing_b63_geometry_v1'
HERE = DIAG / 'orthoflow3_b63_manifold_representation_v1'
BASIS = DIAG / 'double_bottleneck_eta_basis_redesign/tools/bases.py'
BASIS_SHA = '51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38'
RIDGE_ALPHA = 1e-4
K_VALUES = (2, 3, 4)
KNN_K = 8
INDEPENDENT_MIN = 8
SECONDARY_TEST_FRACTION = 0.20
EPS = 1e-12


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, x: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(x, indent=2, sort_keys=True) + '\n')


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else ['status'])
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def pca_plane(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = X.mean(axis=0)
    _, _, vh = np.linalg.svd(X-c, full_matrices=False)
    # Deterministic sign convention: largest-magnitude coordinate positive.
    U = vh[:2].T.copy(); normal = vh[2].copy()
    for j in range(2):
        if U[np.argmax(np.abs(U[:, j])), j] < 0: U[:, j] *= -1
    if normal[np.argmax(np.abs(normal))] < 0: normal *= -1
    s = (X-c) @ U
    n = (X-c) @ normal
    return c, U, normal, np.column_stack([s, n])


def quadratic_features(s: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(s)), s[:, 0], s[:, 1], s[:, 0]**2, s[:, 0]*s[:, 1], s[:, 1]**2])


def fit_quadratic(s: np.ndarray, n: np.ndarray) -> np.ndarray:
    A = quadratic_features(s)
    reg = RIDGE_ALPHA * np.eye(A.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(A.T @ A + reg, A.T @ n)


def qeval(coef: np.ndarray, s: np.ndarray) -> np.ndarray:
    return quadratic_features(s) @ coef


def support2d(train_s: np.ndarray, test_s: np.ndarray) -> np.ndarray:
    if len(train_s) < 3:
        lo, hi = train_s.min(axis=0), train_s.max(axis=0)
        return np.all((test_s >= lo-1e-10) & (test_s <= hi+1e-10), axis=1)
    try:
        hull = Delaunay(train_s)
        return hull.find_simplex(test_s, tol=1e-10) >= 0
    except Exception:
        lo, hi = train_s.min(axis=0), train_s.max(axis=0)
        return np.all((test_s >= lo-1e-10) & (test_s <= hi+1e-10), axis=1)


def deterministic_kmeans(X: np.ndarray, K: int, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    # Farthest-first, lexicographically anchored initialization; no outcome-based choices.
    order = np.lexsort((X[:, 2], X[:, 1], X[:, 0]))
    centers = [X[order[0]]]
    while len(centers) < K:
        C = np.asarray(centers)
        d = np.min(((X[:, None, :] - C[None, :, :])**2).sum(axis=2), axis=1)
        best = np.flatnonzero(np.isclose(d, d.max()))
        centers.append(X[min(best)])
    centers = np.asarray(centers, dtype=float)
    labels = np.zeros(len(X), dtype=int)
    for _ in range(max_iter):
        D2 = ((X[:, None, :] - centers[None, :, :])**2).sum(axis=2)
        new_labels = np.argmin(D2, axis=1)
        new_centers = centers.copy()
        for k in range(K):
            ids = np.where(new_labels == k)[0]
            if len(ids):
                new_centers[k] = X[ids].mean(axis=0)
            else:
                # deterministic refill from globally most distant point
                nearest = np.min(D2, axis=1)
                new_centers[k] = X[int(np.argmax(nearest))]
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers, atol=1e-12):
            labels = new_labels; centers = new_centers; break
        labels, centers = new_labels, new_centers
    return centers, labels


def local_plane(X: np.ndarray, fallback: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(X) < 4:
        return fallback
    c, U, normal, _ = pca_plane(X)
    return c, U, normal


def summarize(residual: np.ndarray, diameter: float) -> dict:
    return {
        'median_reconstruction_error': float(np.median(residual)),
        'p90_reconstruction_error': float(np.quantile(residual, .90)),
        'max_reconstruction_error': float(np.max(residual)),
        'normalized_median_error': float(np.median(residual) / max(diameter, EPS)),
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    if sha(BASIS) != BASIS_SHA:
        raise RuntimeError('authoritative OrthoFlow3 hash mismatch')
    src_manifest = json.load(open(SRC / 'manifest.json'))
    if src_manifest['authoritative_orthoflow3_sha256'] != BASIS_SHA:
        raise RuntimeError('source cloud manifest hash mismatch')
    source_audit = json.load(open(SRC / 'source_compatibility_audit.json'))
    if source_audit['outcome_conflicts'] != 0:
        raise RuntimeError('source evidence contains conflicts')
    cloud_paths = sorted((SRC / 'per_state_b63_clouds').glob('*.csv'))
    if len(cloud_paths) != 8:
        raise RuntimeError('expected eight state clouds')
    dist = {r['state_id']: r for r in read_csv(SRC / 'pairwise_distance_stats.csv')}

    protocol = f'''# Existing B63 manifold representation audit\n\nThis is an offline-only comparison. It accepts only the prior exact B63 clouds and does not execute controller code, submit GPU work, query eta, or train a network. Independent Sobol B63 points are held out whenever at least {INDEPENDENT_MIN} exist. Otherwise a deterministic SHA256 eta-key 80/20 construction-point split is explicitly secondary evidence. R0 is PCA affine 2-D; R1 is its quadratic normal graph with ridge alpha={RIDGE_ALPHA}; R2 is deterministic farthest-first Lloyd K-means plus local PCA planes for K=2,3,4, selected by train-only BIC proxy n*log(MSE)+6K*log(n); R3 is a held-out local k={KNN_K} PCA diagnostic ceiling. Tube thickness is empirical B63 support only, never a verified success claim.\n'''
    (HERE / 'protocol.md').write_text(protocol)

    provenance_rows = []
    affine_rows = []; quadratic_rows = []; patch_rows = []; knn_rows = []
    recon_rows = []; thickness_rows = []; tangential_rows = []; complexity_rows = []; cross_rows = []
    model_data = {}

    for cloud_path in cloud_paths:
        rows = read_csv(cloud_path)
        sid = rows[0]['state_id']
        independent = [r for r in rows if 'coverage' in r['phases']]
        construction = [r for r in rows if r not in independent]
        if len(independent) >= INDEPENDENT_MIN:
            train_rows, test_rows, regime = construction, independent, 'PRIMARY_INDEPENDENT_SOBOL'
        else:
            ordered = sorted(construction, key=lambda r: hashlib.sha256(r['eta_key_float64'].encode()).hexdigest())
            ntest = max(1, int(math.ceil(SECONDARY_TEST_FRACTION * len(ordered))))
            test_rows, train_rows = ordered[:ntest], ordered[ntest:]
            regime = 'SECONDARY_DETERMINISTIC_CONSTRUCTION_SPLIT'
        for r in rows:
            provenance_rows.append({
                'state_id': sid, 'eta_key_float64': r['eta_key_float64'],
                'provenance': 'INDEPENDENT_SOBOL_B63' if r in independent else 'CONSTRUCTION_STRUCTURED_B63',
                'partition': 'TEST' if r in test_rows else 'TRAIN',
                'evaluation_regime': regime,
            })
        Xtr = np.asarray([[float(r['t1']), float(r['t2']), float(r['t3'])] for r in train_rows])
        Xte = np.asarray([[float(r['t1']), float(r['t2']), float(r['t3'])] for r in test_rows])
        diameter = float(dist[sid]['pairwise_max_farthest'])
        c, U, normal, coord = pca_plane(Xtr)
        Str, ntr = coord[:, :2], coord[:, 2]
        Ste = (Xte-c) @ U
        nte = (Xte-c) @ normal
        r0_tr = np.abs(ntr); r0_te = np.abs(nte)
        r0_support = support2d(Str, Ste)
        affine_rows.append({
            'state_id': sid, 'train_n': len(Xtr), 'test_n': len(Xte), 'evaluation_regime': regime,
            'center_t1': c[0], 'center_t2': c[1], 'center_t3': c[2],
            'u1_t1': U[0,0], 'u1_t2': U[1,0], 'u1_t3': U[2,0],
            'u2_t1': U[0,1], 'u2_t2': U[1,1], 'u2_t3': U[2,1],
            'normal_t1': normal[0], 'normal_t2': normal[1], 'normal_t3': normal[2],
            's1_min': Str[:,0].min(), 's1_max': Str[:,0].max(), 's2_min': Str[:,1].min(), 's2_max': Str[:,1].max(),
            **summarize(r0_te, diameter), 'train_residual_median': float(np.median(r0_tr)),
        })

        coef = fit_quadratic(Str, ntr)
        r1_tr = np.abs(ntr - qeval(coef, Str)); r1_te = np.abs(nte - qeval(coef, Ste))
        quadratic_rows.append({
            'state_id': sid, 'train_n': len(Xtr), 'test_n': len(Xte), 'evaluation_regime': regime,
            **{f'a{i}': coef[i] for i in range(6)},
            **summarize(r1_te, diameter), 'train_residual_median': float(np.median(r1_tr)),
            'median_gain_vs_affine': float((np.median(r0_te)-np.median(r1_te))/max(np.median(r0_te), EPS)),
            'p90_gain_vs_affine': float((np.quantile(r0_te,.9)-np.quantile(r1_te,.9))/max(np.quantile(r0_te,.9), EPS)),
        })

        patch_models = {}
        for K in K_VALUES:
            centers, labels = deterministic_kmeans(Xtr, K)
            local = []
            residual_tr = np.zeros(len(Xtr))
            for k in range(K):
                ids = np.where(labels == k)[0]
                lc, lU, ln = local_plane(Xtr[ids], (c, U, normal))
                Slocal = (Xtr[ids]-lc) @ lU
                local.append((lc, lU, ln, Slocal, len(ids)))
                residual_tr[ids] = np.abs((Xtr[ids]-lc) @ ln)
            test_labels = np.argmin(((Xte[:,None,:]-centers[None,:,:])**2).sum(axis=2), axis=1)
            residual_te = np.zeros(len(Xte)); support_te = np.zeros(len(Xte), dtype=bool)
            for j in range(len(Xte)):
                lc,lU,ln,Slocal,_ = local[int(test_labels[j])]
                residual_te[j] = abs(float((Xte[j]-lc) @ ln))
                support_te[j] = bool(support2d(Slocal, ((Xte[j]-lc) @ lU)[None,:])[0])
            mse = float(np.mean(residual_tr**2))
            bic = len(Xtr)*math.log(mse+1e-12) + 6*K*math.log(len(Xtr))
            patch_models[K] = {'centers': centers, 'labels': labels, 'local': local, 'rtr': residual_tr,
                               'rte': residual_te, 'support': support_te, 'bic': bic}
            patch_rows.append({
                'state_id': sid, 'K': K, 'selected_K': False, 'train_n': len(Xtr), 'test_n': len(Xte),
                'cluster_sizes': ';'.join(str(int(np.sum(labels==k))) for k in range(K)),
                'train_mse': mse, 'train_BIC_proxy': bic, **summarize(residual_te, diameter),
                'complexity_parameters_approx': 6*K,
            })
        selected_K = min(K_VALUES, key=lambda K: (patch_models[K]['bic'], K))
        for row in patch_rows:
            if row['state_id'] == sid and row['K'] == selected_K: row['selected_K'] = True
        r2_tr = patch_models[selected_K]['rtr']; r2_te = patch_models[selected_K]['rte']; r2_support = patch_models[selected_K]['support']

        # R3: each test point gets a local 2-D plane from its k nearest train points.
        r3_te = np.zeros(len(Xte))
        for j,x in enumerate(Xte):
            ids = np.argsort(((Xtr-x)**2).sum(axis=1))[:min(KNN_K,len(Xtr))]
            lc,lU,ln = local_plane(Xtr[ids], (c,U,normal))
            r3_te[j] = abs(float((x-lc) @ ln))
        knn_rows.append({'state_id': sid, 'k': KNN_K, 'train_n': len(Xtr), 'test_n': len(Xte),
                         'evaluation_regime': regime, **summarize(r3_te, diameter),
                         'complexity': 'stores local point-cloud neighborhoods; diagnostic ceiling only'})

        models = {
            'R0_AFFINE': (r0_tr, r0_te, r0_support),
            'R1_QUADRATIC': (r1_tr, r1_te, r0_support),
            f'R2_PATCH_K{selected_K}': (r2_tr, r2_te, r2_support),
        }
        for name,(rtr,rte,support) in models.items():
            d90,d95,dmax = np.quantile(rtr,.90),np.quantile(rtr,.95),np.max(rtr)
            thickness_rows.append({'state_id':sid,'representation':name,'selected_patch_K':selected_K if name.startswith('R2') else '',
                                   'delta90':d90,'delta95':d95,'delta_max':dmax,
                                   'heldout_within_delta90':float(np.mean(rte<=d90+1e-12)),
                                   'heldout_within_delta95':float(np.mean(rte<=d95+1e-12)),
                                   'heldout_within_delta_max':float(np.mean(rte<=dmax+1e-12))})
            tangential_rows.append({'state_id':sid,'representation':name,'selected_patch_K':selected_K if name.startswith('R2') else '',
                                    'heldout_tangential_support_fraction':float(np.mean(support)),
                                    'near_surface_outside_support_fraction':float(np.mean((rte<=d95+1e-12)&(~support))),
                                    'near_surface_outside_support_count':int(np.sum((rte<=d95+1e-12)&(~support))),
                                    'heldout_count':len(Xte)})
        for j, row in enumerate(test_rows):
            for name, residual, support in [
                ('R0_AFFINE',r0_te[j],r0_support[j]), ('R1_QUADRATIC',r1_te[j],r0_support[j]),
                (f'R2_PATCH_K{selected_K}',r2_te[j],r2_support[j]), ('R3_KNN_LOCAL',r3_te[j],None)]:
                recon_rows.append({'state_id':sid,'eta_key_float64':row['eta_key_float64'],'evaluation_regime':regime,
                                   'provenance':'INDEPENDENT_SOBOL_B63' if row in independent else 'CONSTRUCTION_STRUCTURED_B63',
                                   'representation':name,'reconstruction_error':residual,
                                   'normalized_error':residual/max(diameter,EPS),
                                   'inside_train_tangential_support': '' if support is None else bool(support)})
        complexity_rows += [
            {'representation':'R0_AFFINE','state_id':sid,'selected':True,'complexity':'center(3)+orthonormal 2-D orientation(3 effective)','parameter_count_approx':6},
            {'representation':'R1_QUADRATIC','state_id':sid,'selected':True,'complexity':'R0 + six quadratic normal-graph coefficients','parameter_count_approx':12},
            {'representation':'R2_PATCH','state_id':sid,'selected':True,'complexity':f'{selected_K} local centers/planes','parameter_count_approx':6*selected_K},
            {'representation':'R3_KNN_LOCAL','state_id':sid,'selected':False,'complexity':'local point-cloud neighborhood memory','parameter_count_approx':'nonparametric'},
        ]
        model_data[sid] = {'regime':regime,'selected_K':selected_K,'r0':r0_te,'r1':r1_te,'r2':r2_te,'r3':r3_te,
                           'diameter':diameter,'r0support':r0_support,'r2support':r2_support}
        cross_rows.append({'state_id':sid,'evaluation_regime':regime,'train_n':len(Xtr),'independent_heldout_n':len(independent),
                           'actual_heldout_n':len(Xte),'selected_patch_K':selected_K,
                           'diameter':diameter,
                           'affine_normal_t1':normal[0],'affine_normal_t2':normal[1],'affine_normal_t3':normal[2],
                           'r0_median':float(np.median(r0_te)),'r1_median':float(np.median(r1_te)),
                           'r2_median':float(np.median(r2_te)),'r3_median':float(np.median(r3_te)),
                           'affine_thickness95':float(np.quantile(r0_tr,.95)),
                           'quadratic_thickness95':float(np.quantile(r1_tr,.95)),
                           'patch_thickness95':float(np.quantile(r2_tr,.95)),
                           'r0_tangential_coverage':float(np.mean(r0_support)),
                           'r2_tangential_coverage':float(np.mean(r2_support))})

    write_csv(HERE/'provenance_split.csv', provenance_rows)
    write_csv(HERE/'affine_plane_fit.csv', affine_rows)
    write_csv(HERE/'quadratic_surface_fit.csv', quadratic_rows)
    write_csv(HERE/'local_patch_fit.csv', patch_rows)
    write_csv(HERE/'knn_tube_fit.csv', knn_rows)
    write_csv(HERE/'heldout_reconstruction.csv', recon_rows)
    write_csv(HERE/'tube_thickness.csv', thickness_rows)
    write_csv(HERE/'tangential_coverage.csv', tangential_rows)
    write_csv(HERE/'complexity_comparison.csv', complexity_rows)
    write_csv(HERE/'cross_state_representation.csv', cross_rows)

    primary = [x for x in cross_rows if x['evaluation_regime']=='PRIMARY_INDEPENDENT_SOBOL']
    def median_of(key): return float(statistics.median(float(x[key]) for x in primary))
    gains_r1 = [1-float(x['r1_median'])/max(float(x['r0_median']),EPS) for x in primary]
    gains_r2 = [1-float(x['r2_median'])/max(min(float(x['r0_median']),float(x['r1_median'])),EPS) for x in primary]
    gains_r3 = [1-float(x['r3_median'])/max(float(x['r2_median']),EPS) for x in primary]
    r0_good = sum(float(x['r0_median'])/float(x['diameter']) <= .05 and float(x['r0_tangential_coverage']) >= .75 for x in primary)
    r1_better = sum(g >= .20 for g in gains_r1)
    r2_better = sum(g >= .20 for g in gains_r2)
    r3_better = sum(g >= .25 for g in gains_r3)
    if len(primary) < 5:
        decision = 'B63_GEOMETRY_TOO_SPARSE_OR_INCONSISTENT'
    elif r0_good >= 4:
        decision = 'AFFINE_2D_TUBE_SUFFICIENT'
    elif r1_better >= 4 and r2_better < 4:
        decision = 'QUADRATIC_2D_TUBE_PREFERRED'
    elif r2_better >= 4 and r3_better < 4:
        decision = 'FEW_LOCAL_PATCHES_PREFERRED'
    elif r3_better >= 4:
        decision = 'GENERAL_LOCAL_MANIFOLD_REQUIRED'
    else:
        decision = 'B63_GEOMETRY_TOO_SPARSE_OR_INCONSISTENT'
    decision_obj = {
        'classification':decision,'primary_independent_states':len(primary),'secondary_states':8-len(primary),
        'primary_median_heldout_residuals':{'R0_affine':median_of('r0_median'),'R1_quadratic':median_of('r1_median'),
                                             'R2_selected_patch':median_of('r2_median'),'R3_knn_ceiling':median_of('r3_median')},
        'median_relative_gain':{'R1_vs_R0':float(statistics.median(gains_r1)),'R2_vs_best_R0_R1':float(statistics.median(gains_r2)),
                                'R3_vs_R2':float(statistics.median(gains_r3))},
        'criteria_counts':{'R0_good_states':r0_good,'R1_material_gain_states':r1_better,'R2_material_gain_states':r2_better,'R3_material_gain_states':r3_better},
        'fixed_protocol':{'ridge_alpha':RIDGE_ALPHA,'patch_K_values':list(K_VALUES),'knn_k':KNN_K,'independent_min':INDEPENDENT_MIN},
        'critical_limitation':'Success-only point-cloud fits cannot establish success-set connectedness, interpolation safety, absence of holes, or tube false-inclusion rate.',
        'new_rollouts':0,'gpu_jobs_submitted':0,'new_eta_queries':0,'networks_trained':0,
    }
    dump(HERE/'representation_decision.json',decision_obj)

    lines=[]
    for x in cross_rows:
        lines.append(f"| {x['state_id']} | {x['train_n']} | {x['independent_heldout_n']} | {x['evaluation_regime']} | {x['r0_median']:.4f} | {x['r1_median']:.4f} | {x['selected_patch_K']} / {x['r2_median']:.4f} | {x['r3_median']:.4f} | {x['r0_tangential_coverage']:.2f} |")
    report=f'''# Existing B63 low-complexity manifold representation audit\n\n## Scope\n\nThis audit ran **zero** new rollouts, submitted **zero** GPU jobs, queried **zero** new eta values, and trained **zero** networks. It used only the exact B63 point clouds and their archived conditioning/provenance manifest.\n\n## Held-out protocol\n\nIndependent Sobol B63 points are the primary held-out geometry set for 6/8 states. `ep0082` (6 points) and `ep0195` (4 points) use an explicitly secondary deterministic construction-point split because they do not meet the preregistered minimum of {INDEPENDENT_MIN} independent B63 points.\n\n| State | train | independent B63 | regime | R0 median | R1 median | R2 selected K / median | R3 median | R0 tangential coverage |\n|---|---:|---:|---|---:|---:|---:|---:|---:|\n{chr(10).join(lines)}\n\n## Representation comparison\n\nPrimary independent-held-out median residuals are R0 affine **{decision_obj['primary_median_heldout_residuals']['R0_affine']:.4f}**, R1 quadratic **{decision_obj['primary_median_heldout_residuals']['R1_quadratic']:.4f}**, R2 selected patches **{decision_obj['primary_median_heldout_residuals']['R2_selected_patch']:.4f}**, and R3 local kNN ceiling **{decision_obj['primary_median_heldout_residuals']['R3_knn_ceiling']:.4f}**. Median relative gains are R1/R0 **{100*decision_obj['median_relative_gain']['R1_vs_R0']:.1f}%**, R2/best(R0,R1) **{100*decision_obj['median_relative_gain']['R2_vs_best_R0_R1']:.1f}%**, and R3/R2 **{100*decision_obj['median_relative_gain']['R3_vs_R2']:.1f}%**.\n\nR0 has the lowest complexity: a center and oriented 2-D plane. R1 adds six normal-graph coefficients. R2 uses {K_VALUES} local planes and train-only BIC selection. R3 is nonparametric and is only a local-reconstruction ceiling. Empirical `delta90/delta95/delta_max` values are support thicknesses around B63 samples; they are **not certified success tubes**.\n\n## Decision\n\n**{decision}**\n\nThe decision uses only primary independent-held-out states where possible. A common class with state-dependent parameters can be used across states, but no fitted surface certifies interpolation or thickness safety.\n\n## Required next validation\n\nThe next experiment should use a small, frozen rollout set targeted to: (1) interpolation within fitted tangential supports, (2) normal offsets near empirical `delta95`, and (3) candidate gaps between nearby successful samples. It must measure both B63 success and false inclusion before any learning target is adopted.\n\n## Limitation\n\nThis task cannot establish success-set connectedness, interpolation safety, absence of holes, or a tube false-inclusion rate, because it used success points only and ran no new rollout.\n'''
    (HERE/'b63_manifold_representation_report.md').write_text(report)
    required=['protocol.md','provenance_split.csv','affine_plane_fit.csv','quadratic_surface_fit.csv','local_patch_fit.csv','knn_tube_fit.csv','heldout_reconstruction.csv','tube_thickness.csv','tangential_coverage.csv','complexity_comparison.csv','cross_state_representation.csv','representation_decision.json','b63_manifold_representation_report.md']
    dump(HERE/'manifest.json',{'experiment':'ORTHOFLOW3_EXISTING_B63_MANIFOLD_REPRESENTATION_AUDIT_V1',
        'created_utc':datetime.now(timezone.utc).isoformat(),'authoritative_orthoflow3_sha256':sha(BASIS),
        'source_manifest_sha256':sha(SRC/'manifest.json'),'new_rollouts':0,'gpu_jobs_submitted':0,'new_eta_queries':0,'networks_trained':0,
        'artifacts':{name:sha(HERE/name) for name in required}})
    print(json.dumps(decision_obj,indent=2))


if __name__=='__main__':
    main()
