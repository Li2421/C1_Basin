"""Paired unknown bounds, preserving cancellation of identical physical IDs."""
from common import *
from actions import TIMES

def difference(a,b):
    # Each endpoint is (lower, upper, exact physical UID).
    same=a[2]==b[2]
    return np.asarray([np.where(same,0,a[0]-b[1]).mean(),np.where(same,0,a[1]-b[0]).mean()])

def endpoint(d,key,draw,seed=None):
    ix=(draw,slice(None)) if seed is None else (draw,seed)
    return d[0][key][ix],d[1][key][ix],d['uids'][key][ix]

def full_bounds(d,draw):
    old=endpoint(d,'old_selected',draw);prog=endpoint(d,'selected_progress',draw);human=endpoint(d,'human_progress',draw)
    lower=np.stack([d[0]['selected_'+str(t)][draw] for t in TIMES],-1);chosen=np.argmax(lower.mean(0),-1)
    best=[]
    for q in (0,1):best.append(endpoint(d,'selected_'+str(TIMES[chosen[1-q]]),draw,q))
    fixed=tuple(np.stack([best[q][k] for q in (0,1)],axis=-1) for k in range(3))
    return np.stack([difference(human,old),difference(prog,old),difference(fixed,old),difference(human,prog),*[difference(human,endpoint(d,'human_'+str(t),draw)) for t in TIMES]],axis=-1)

def proposal_choices(labels,draw):
    total=labels[draw].sum(0);counts=np.bincount(draw,minlength=len(labels));picks=[]
    for sid in draw:
        scores=total-counts[sid]*labels[sid]
        picks.append(0 if len(draw)==counts[sid] else int(np.argmax(scores)))
    return np.asarray(picks,int)

def coverage_bounds(d,draw):
    old=endpoint(d,'old_selected',draw,0);prog=endpoint(d,'selected_progress',draw,0);human=endpoint(d,'human_progress',draw,0);picks=proposal_choices(d[0]['proposal'],draw)
    best=(d[0]['proposal'][draw,picks],d[1]['proposal'][draw,picks],d['uids']['proposal'][draw,picks])
    maximum=np.asarray([d[b]['proposal'][draw].max(-1).mean() for b in (0,1)])
    return np.stack([difference(human,old),difference(prog,old),difference(best,prog),difference(human,best),maximum],axis=-1)
