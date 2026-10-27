"""Read-only post-hoc homogeneity audit of the 18 labelled cross-scene correction scenes.

Inputs are read from the live experiment directory; nothing is written there.
design/ridge_fit/ridge_predict/knn_predict/select are copied verbatim from that
experiment's models.py so the reduced-set refits use the frozen learner exactly.
"""
import json, sys
from pathlib import Path
import numpy as np

EXP = Path('/home/zhihan/research/Basin_C1/diagnostics/cross_scene_correction_v1')
ACTIONS = Path('/home/zhihan/research/Basin_C1/diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/best_correction_search_v1/ACTIONS.json')
OUT = Path(__file__).resolve().parent

# ---- copied verbatim from models.py ----
LAMBDAS=(.1,1.,10.,100.)
KS=(1,2,3,5)


def design(features,etas):
    f=np.asarray(features,float);e=np.asarray(etas,float)
    return np.column_stack([np.repeat(f[None],len(e),axis=0),e,e*e,(f[None,:,None]*e[:,None,:]).reshape(len(e),-1)])


def ridge_fit(source_features,source_tables,etas,penalty):
    matrices=[design(f,etas)for f in source_features]
    X=np.vstack(matrices);y=np.asarray(source_tables,float).reshape(-1)
    mu=X.mean(0);sd=X.std(0);sd=np.where(sd>1e-12,sd,1.)
    Z=(X-mu)/sd;ym=y.mean()
    coef=np.linalg.solve(Z.T@Z+penalty*np.eye(Z.shape[1]),Z.T@(y-ym))
    return dict(mean=mu.tolist(),scale=sd.tolist(),coefficient=coef.tolist(),intercept=float(ym),penalty=float(penalty))


def ridge_predict(source_features,source_tables,target_features,etas,penalty):
    state=ridge_fit(source_features,source_tables,etas,penalty)
    return ((design(target_features,etas)-state['mean'])/state['scale'])@np.asarray(state['coefficient'])+state['intercept']


def knn_predict(source_features,source_tables,target_features,k):
    X=np.asarray(source_features,float);mu=X.mean(0);sd=X.std(0);sd=np.where(sd>1e-12,sd,1.)
    distance=np.sum(((X-target_features)/sd)**2,axis=1)
    order=np.argsort(distance,kind='stable')[:min(k,len(X))]
    return np.asarray(source_tables)[order].mean(0)


def select(features,tables,target,etas):
    f=np.asarray(features,float);q=np.asarray(tables,float)
    ridge_cv={};knn_cv={}
    for penalty in LAMBDAS:
        scores=[]
        for i in range(len(f)):
            train=[j for j in range(len(f))if j!=i]
            choice=int(np.argmax(ridge_predict(f[train],q[train],f[i],etas,penalty)))
            scores.append(float(q[i,choice]))
        ridge_cv[str(penalty)]=float(np.mean(scores))
    for k in KS:
        scores=[]
        for i in range(len(f)):
            train=[j for j in range(len(f))if j!=i]
            choice=int(np.argmax(knn_predict(f[train],q[train],f[i],k)))
            scores.append(float(q[i,choice]))
        knn_cv[str(k)]=float(np.mean(scores))
    best=max(ridge_cv.values());penalty=next(p for p in LAMBDAS if abs(ridge_cv[str(p)]-best)<=1e-12)
    best=max(knn_cv.values());k=next(k for k in KS if abs(knn_cv[str(k)]-best)<=1e-12)
    state=ridge_fit(f,q,etas,penalty)
    rp=((design(target,etas)-state['mean'])/state['scale'])@np.asarray(state['coefficient'])+state['intercept']
    kp=knn_predict(f,q,target,k)
    return dict(ridge=int(np.argmax(rp)),knn=int(np.argmax(kp)),global_fixed=int(np.argmax(q.mean(0))),penalty=penalty,k=k)
# ---- end of copy ----

KEYS = ['safety'] + [f'eta{i}' for i in range(16)] + [f'sE{i:02d}' for i in range(32)]


def load():
    scenes = [s for s in json.load(open(EXP/'SCENES.json')) if s['family'] != 'extrapolation']
    tables = json.load(open(EXP/'LABEL_TABLES.json'))
    rows = json.load(open(ACTIONS))['lit_swap8_free']
    etas = np.asarray([rows[k]['eta_a'] for k in KEYS])
    names = [s['name'] for s in scenes]
    f = np.asarray([s['features'] for s in scenes], float)
    y = {n: np.asarray(tables[n]['y'], float) for n in names}
    q = np.asarray([y[n].mean((0, 1)) for n in names])
    fam = {s['name']: s['family'] for s in scenes}
    return names, fam, f, y, q, etas


def loso(idx, names, f, q, etas):
    out = {}
    for i in idx:
        train = [j for j in idx if j != i]
        out[names[i]] = select(f[train], q[train], f[i], etas)
    return out


def rank(v):
    order = np.argsort(v, kind='stable'); r = np.empty(len(v)); r[order] = np.arange(len(v))
    # average ranks for ties
    for val in np.unique(v):
        m = v == val
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def boot_diff(idx, names, y, choices, method, reps=5000, seed=20261025):
    rng = np.random.default_rng(seed)
    per = []
    for i in idx:
        n = names[i]; c = choices[n]
        per.append((y[n][:, :, c[method]] - y[n][:, :, c['global_fixed']]))
    point = float(np.mean([d.mean() for d in per]))
    draws = np.empty(reps)
    for b in range(reps):
        s = rng.integers(len(per), size=len(per)); vals = []
        for j in s:
            d = per[j]; C, S = d.shape
            ct = rng.integers(C, size=C); st = rng.integers(S, size=(C, S))
            vals.append(d[ct[:, None], st].mean())
        draws[b] = np.mean(vals)
    return point, np.percentile(draws, [2.5, 97.5]).tolist()


def main():
    names, fam, f, y, q, etas = load()
    N = len(names); allidx = list(range(N))
    report = {}

    # 1. reproduce the frozen LOSO choices
    frozen = json.load(open(EXP/'LOSO_CHOICES.json'))
    full = loso(allidx, names, f, q, etas)
    mism = [(n, m) for n in names for m in ('ridge', 'knn', 'global_fixed') if full[n][m] != frozen[n][m]]
    report['reproduction_mismatches'] = mism
    print('reproduction mismatches:', mism)

    # 2. response-profile similarity
    best = q.argmax(1)
    rho = np.array([[spearman(q[i], q[j]) for j in range(N)] for i in range(N)])
    regret = np.array([[q[i, best[i]] - q[i, best[j]] for j in range(N)] for i in range(N)])  # use j's best on i
    gall = int(q.mean(0).argmax())
    print('\nper scene: in-sample best, its rate, all-18 global best rate, gap (pp), nearest by rho, rho, cross-regret i<-j / j<-i')
    for i in range(N):
        order = [j for j in np.argsort(-rho[i]) if j != i]
        j = order[0]
        print(f'{names[i]:22s} {KEYS[best[i]]:6s} {100*q[i,best[i]]:5.1f}  global {100*q[i,gall]:5.1f}  gap {100*(q[i,best[i]]-q[i,gall]):5.1f} | nearest {names[j]:20s} rho {rho[i,j]:.2f}  regret {100*regret[i,j]:4.1f}/{100*regret[j,i]:4.1f}')
    report['all18_global_best'] = KEYS[gall]
    report['scene_best'] = {names[i]: KEYS[best[i]] for i in range(N)}

    # 3. pre-stated redundancy rule: a pair is redundant when rho >= 0.9 and each scene's
    # in-sample best loses <= 2 pp on the other.  Greedy: highest-rho redundant pair first;
    # drop the member with the higher mean rho to the remaining scenes (the more typical one).
    keep = set(allidx); dropped = []
    while True:
        pairs = [(rho[i, j], i, j) for i in keep for j in keep if i < j and rho[i, j] >= 0.9
                 and regret[i, j] <= 0.02 + 1e-12 and regret[j, i] <= 0.02 + 1e-12]
        if not pairs: break
        _, i, j = max(pairs)
        mi = np.mean([rho[i, k] for k in keep if k != i]); mj = np.mean([rho[j, k] for k in keep if k != j])
        d = i if mi >= mj else j; o = j if d == i else i
        keep.discard(d); dropped.append((names[d], names[o], round(float(rho[i, j]), 3)))
    print('\nredundant scenes dropped (dropped, kept twin, rho):')
    for row in dropped: print('  ', row)
    report['dropped'] = dropped
    kidx = sorted(keep)
    print('kept', len(kidx), [names[i] for i in kidx])

    # 4. LOSO on the full and reduced sets
    for label, idx, ch in (('full18', allidx, full), ('reduced', kidx, None)):
        ch = ch if ch is not None else loso(idx, names, f, q, etas)
        diffs = {names[i]: float(q[i, ch[names[i]]['ridge']] - q[i, ch[names[i]]['global_fixed']]) for i in idx}
        kd = {names[i]: float(q[i, ch[names[i]]['knn']] - q[i, ch[names[i]]['global_fixed']]) for i in idx}
        d = np.array(list(diffs.values()))
        pr, cr = boot_diff(idx, names, y, ch, 'ridge')
        pk, ck = boot_diff(idx, names, y, ch, 'knn')
        rates = {m: float(np.mean([q[i, ch[names[i]][m]] for i in idx])) for m in ('ridge', 'knn', 'global_fixed')}
        ins = float(np.mean([q[i].max() for i in idx]))
        print(f'\n[{label}] n={len(idx)} ridge {100*rates["ridge"]:.1f} knn {100*rates["knn"]:.1f} global {100*rates["global_fixed"]:.1f} in-sample {100*ins:.1f}')
        print(f'  ridge-global {100*pr:+.2f} CI [{100*cr[0]:+.2f}, {100*cr[1]:+.2f}]  median {100*np.median(d):+.2f}  wins {int((d>0).sum())} losses {int((d<0).sum())} ties {int((d==0).sum())}')
        print(f'  knn-global   {100*pk:+.2f} CI [{100*ck[0]:+.2f}, {100*ck[1]:+.2f}]')
        print('  per-scene ridge-global pp:', {k: round(100*v, 1) for k, v in diffs.items()})
        report[label] = dict(n=len(idx), rates=rates, in_sample=ins, ridge_minus_global=pr, ridge_ci=cr,
                             knn_minus_global=pk, knn_ci=ck, per_scene_ridge=diffs, per_scene_knn=kd,
                             global_choice={names[i]: KEYS[ch[names[i]]['global_fixed']] for i in idx},
                             ridge_choice={names[i]: KEYS[ch[names[i]]['ridge']] for i in idx},
                             penalty={names[i]: ch[names[i]]['penalty'] for i in idx})
    json.dump(dict(report=report, rho=rho.tolist(), regret=regret.tolist(), names=names),
              open(OUT/'HOMOGENEITY_AUDIT.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
