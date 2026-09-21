"""Finite-state mathematical audit; no learned model or robot execution."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def probability(P, deadlock, success):
    """Reference recursion with mutually exclusive absorbing event labels."""
    P=np.asarray(P,float);D=np.asarray(deadlock,bool);S=np.asarray(success,bool)
    T,n,_=P.shape
    assert D.shape==S.shape==(T+1,n) and not np.any(D&S)
    assert np.all(P>=0) and np.allclose(P.sum(-1),1.)
    q=np.zeros((T+1,n));q[-1]=D[-1]
    for t in range(T-1,-1,-1):
        q[t]=np.where(D[t],1.,np.where(S[t],0.,P[t]@q[t+1]))
    return q


def certificate(P,D,S,b):
    """Uniform positive Bellman residual bound, requiring exact transitions."""
    b=np.asarray(b,float)
    assert np.isfinite(b).all() and np.all(b>=0) and np.all(b[D]>=1.)
    eps=np.zeros(len(P)+1)
    eps[-1]=max(0.,float(np.max(D[-1].astype(float)-b[-1])))
    for t in range(len(P)):
        alive=~(D[t]|S[t])
        eps[t]=max(0.,float(np.max((P[t]@b[t+1]-b[t])[alive]))) if alive.any() else 0.
    budget=np.cumsum(eps[::-1])[::-1]
    return np.minimum(1.,b+budget[:,None]),eps


def main():
    out=ROOT/'results/c1_bellman_certificate_audit_v1.json'
    if out.exists():raise FileExistsError(out)
    # States: two detector-history states with identical possible actor inputs,
    # a deadlock absorbing state, and a successful absorbing state.
    P=np.array([[[0.,0.,.8,.2],[0.,0.,.1,.9],[0.,0.,1.,0.],[0.,0.,0.,1.]]]*3)
    D=np.tile([False,False,True,False],(4,1));S=np.tile([False,False,False,True],(4,1))
    q=probability(P,D,S)
    np.testing.assert_allclose(q[0],[.8,.1,1.,0.])
    exact_bound,exact_eps=certificate(P,D,S,q)
    np.testing.assert_allclose(exact_bound,q);np.testing.assert_allclose(exact_eps,0.)
    # An underestimating value at state0 is not a certificate by itself.
    bad=q.copy();bad[:-1,0]=.3
    corrected,eps=certificate(P,D,S,bad)
    assert np.all(corrected+1e-12>=q) and bad[0,0]<q[0,0]
    # Score error alone cannot bound its parameter derivative.
    # q(theta)=.5 locally, estimate=.5+delta*sin(k*theta).
    delta=1e-4;k=1e6
    result=dict(scope='finite-state reference only; no robot transition certificate',
        exact_probabilities=q.tolist(),exact_residuals=exact_eps.tolist(),
        underestimated_value=float(bad[0,0]),corrected_bound=float(corrected[0,0]),
        positive_residuals=eps.tolist(),
        omitted_history_counterexample={'same_actor_observation_possible':True,'risks':[.8,.1]},
        value_gradient_counterexample={'uniform_value_error':delta,'true_gradient':0.,'estimated_gradient_at_zero':delta*k},
        learned_models_added=0,robot_replays=0,training_launched=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
