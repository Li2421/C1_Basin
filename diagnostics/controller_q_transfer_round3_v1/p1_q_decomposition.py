"""Descriptive level/ranking decomposition after full P1 postflight.

Gate statistics are solely in p1_analyze.py. This file never affects a split,
controller qualification, or a model-selection decision.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent


def main():
    assert json.loads((HERE/'P1_POSTFLIGHT.json').read_text())['missing']==0
    z=np.load(HERE/'P1_LABELS.npz')
    Y=z['Y'];U=z['U'];names=list(z['controller_names'])
    assert Y.shape==(14,24,16,4)
    q=np.divide(Y.sum(-1),4-U.sum(-1),out=np.full((14,24,16),np.nan),where=U.sum(-1)<4)
    means=np.nanmean(q,(1,2));eta_means=np.nanmean(q,1)
    # Paired state cluster resamples, keeping all controller/eta/seed records
    # of a physical state together. Unknowns are bounded separately.
    draw=np.random.default_rng(2026101210).integers(24,size=(5000,24))
    lower=Y.mean((2,3));upper=(Y+U).mean((2,3))
    mean_ci={name:np.quantile(lower[c,draw].mean(1),[.025,.975]).tolist() for c,name in enumerate(names)}
    pair=[]
    for a,b in combinations(range(14),2):
        rho=spearmanr(eta_means[a],eta_means[b]).statistic
        d=lower[a]-lower[b]
        pair.append(dict(a=names[a],b=names[b],absolute_level_difference=float(abs(means[a]-means[b])),
                         signed_level_difference_lower=float(d.mean()),
                         signed_level_difference_ci95_lower=np.quantile(d[draw].mean(1),[.025,.975]).tolist(),
                         eta_rank_spearman=None if not np.isfinite(rho) else float(rho),
                         best_eta_differs=bool(int(np.argmax(eta_means[a]))!=int(np.argmax(eta_means[b])))))
    result=dict(status='EXPLORATORY COMPLETE DEVELOPMENT; NO SPLIT OR GATE EFFECT',
                controller_Q_levels=dict(zip(names,map(float,means))),
                controller_Q_lower=dict(zip(names,map(float,lower.mean(1)))),
                controller_Q_upper=dict(zip(names,map(float,upper.mean(1)))),
                controller_Q_lower_state_cluster_ci95=mean_ci,
                controller_best_hindsight_eta=dict(zip(names,map(int,np.argmax(eta_means,axis=1)))),
                controller_eta_mean_Q=eta_means.tolist(),
                between_controller_level_sd=float(means.std()),
                median_pairwise_absolute_level_difference=float(np.median([p['absolute_level_difference'] for p in pair])),
                median_pairwise_eta_rank_spearman=float(np.median([p['eta_rank_spearman'] for p in pair if p['eta_rank_spearman'] is not None])),
                fraction_controller_pairs_different_best_eta=float(np.mean([p['best_eta_differs'] for p in pair])),
                pairwise=pair,unknown_count=int(U.sum()),
                scope='24 opened development states, K16 Q4; true continuous eta basin is not identified')
    (HERE/'P1_Q_DECOMPOSITION.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print({k:result[k] for k in ('between_controller_level_sd','median_pairwise_eta_rank_spearman','fraction_controller_pairs_different_best_eta')})


if __name__=='__main__':main()
