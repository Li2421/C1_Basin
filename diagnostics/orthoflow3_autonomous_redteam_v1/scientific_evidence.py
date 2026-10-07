"""Versioned minimal report repairs; no physical/controller/model mutation.

Use only complete canonical seed identities as requested; numerical rows remain
observations of solver non-certification, never manufactured task outcomes.
"""
import math

def summarize_canonical16(rows):
    by={}
    for r in rows:
        seed=int(r['future_index'])
        if seed not in range(16):raise ValueError('noncanonical seed')
        if seed in by:raise ValueError('duplicate seed')
        if r.get('numerical_failure') and r.get('success'):
            raise ValueError('uncertified success')
        by[seed]=r
    s=sum(bool(r['success']) for r in by.values() if not r.get('numerical_failure'))
    f=sum(not r['success'] for r in by.values() if not r.get('numerical_failure'))
    u=16-s-f
    return {'successes':s,'valid_failures':f,'uncertified_or_missing':u,
            'Q16':s/16 if u==0 else None,'Q_lower':s/16,'Q_upper':(s+u)/16,
            'robust':True if s>=15 else False if f>=2 else None}

def exact_label_or_proxy(request,label):
    keys=('state_uid','eta_uid','controller_uid')
    exact=all(request[k]==label[k] for k in keys)
    return {'identity_match':exact,'evidence_type':'EXACT_TUPLE' if exact else 'NEAREST_LABEL_PROXY',
            'robust':label['robust_15of16'] if exact else None,
            'proxy_robust':None if exact else label['robust_15of16'],
            'eligible_for_readiness_decision':exact}

def finite_logit_argmax(logits):
    """Rank the same finite candidates without sigmoid's finite-precision ties.

    Mirrors already-written Phase-A code, not a new deployment intervention.
    Preserves existing first-index tie rule for genuinely equal raw logits.
    """
    if not logits or not all(math.isfinite(float(x)) for x in logits):
        raise ValueError('empty/nonfinite finite proposal scores')
    return max(range(len(logits)),key=lambda i:(float(logits[i]),-i))
