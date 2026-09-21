"""Fit invertible preprocessing using only the actual training mixture."""
import numpy as np


def fit_normalization(dataset, recovery=None, recovery_fraction=0.5):
    sources = [(dataset, 1.)] if recovery is None else [(dataset, 1-recovery_fraction), (recovery, recovery_fraction)]
    if any(d.split != 'train' for d, _ in sources):
        raise ValueError('Statistics must use train only')
    result = {'normalize': True}
    for field, prefix in [('observations', 'obs'), ('actions', 'act')]:
        moments = []
        for source, weight in sources:
            a = getattr(source, field).reshape(len(source), -1).astype(np.float64)
            moments.append((weight, a.mean(0), (a*a).mean(0)))
        mean = sum(w*m for w,m,_ in moments)
        variance = np.maximum(sum(w*v for w,_,v in moments)-mean*mean, 0.)
        result[prefix+'_mean'] = mean.tolist()
        result[prefix+'_scale'] = np.maximum(np.sqrt(variance), .01).tolist()
    return result
