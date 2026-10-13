"""Meaningful synthetic tests: paired unknown bounds and max-T invariance."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from common import *
import numpy as np
from analyze import contrast,max_t

def main():
    rng=np.random.default_rng(2026101320);x=rng.choice([-1,0,1],size=(32,16,4));y=(x==1).mean(-1);u=(x==-1).mean(-1)
    a=rng.integers(16,size=(3,32));b=rng.integers(16,size=(3,32));lo,hi=contrast(y[None],u[None],a,b)
    zero=contrast(y[None],u[None],a,a);assert np.array_equal(zero[0],np.zeros((3,32))) and np.array_equal(zero[1],np.zeros((3,32)))
    for i in range(1000):
        complete=np.where(x==-1,rng.integers(2,size=x.shape),x).mean(-1)
        actual=np.take_along_axis(complete[None],a[...,None],-1)[...,0]-np.take_along_axis(complete[None],b[...,None],-1)[...,0]
        assert np.all(actual>=lo-1e-12) and np.all(actual<=hi+1e-12)
    matrix=rng.normal(size=(32,4));obs,interval,c=max_t(matrix);same=max_t(matrix[:,[2,0,3,1]])
    assert np.allclose(obs[[2,0,3,1]],same[0]) and np.allclose(interval[[2,0,3,1]],same[1]) and c==same[2]
    degenerate=max_t(np.zeros((32,4)));assert np.array_equal(degenerate[1],np.zeros((4,2)))
    rec=dict(status='PASS',unknown_completion_draws=1000,identical_eta_unknown_cancellation=True,max_t_column_permutation_invariance=True,max_t_degenerate_zero=True,
        statistics_code_sha256=sha(HERE/'analyze.py'),utc=utc(),does_not_read_rollout_labels=True)
    write(HERE/'STATISTICS_VERIFICATION.json',rec);print(json.dumps(rec))
if __name__=='__main__':main()
