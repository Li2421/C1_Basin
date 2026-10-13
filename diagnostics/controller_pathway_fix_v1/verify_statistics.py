"""Verify unknown pairing and simultaneous interval families before endpoints."""
from util import *
from evaluate import contrast,choose,simultaneous
def main():
    rng=np.random.default_rng(606);outcomes=rng.integers(-1,2,size=(32,16,4));Y=(outcomes==1).mean(-1);U=(outcomes==-1).mean(-1)
    a=rng.integers(16,size=(3,32));b=rng.integers(16,size=(3,32));lo,hi=contrast(Y[None],U[None],a,b)
    for i in range(200):
        filled=outcomes.copy();unknown=filled==-1;filled[unknown]=rng.integers(2,size=unknown.sum());truth=filled.mean(-1)
        diff=np.take_along_axis(truth[None],a[...,None],-1)[...,0]-np.take_along_axis(truth[None],b[...,None],-1)[...,0]
        assert np.all(diff>=lo-1e-12) and np.all(diff<=hi+1e-12)
    l,h=contrast(Y[None],U[None],a,a);assert np.array_equal(l,np.zeros_like(l)) and np.array_equal(h,np.zeros_like(h))
    logits=np.full((3,2,32,16),-3.);logits[:,:,:,10]=3.
    assert np.all(choose(logits,'conservative',10)==10)
    logits[0,:,:,4]=4.;assert np.all(choose(logits,'conservative',10)==10)
    logits[:,:,:,4]=5.;assert np.all(choose(logits,'conservative',10)==10),'Mean Q difference less than .1 should fall back'
    logits[:,:,:,10]=0.;assert np.all(choose(logits,'conservative',10)==4)
    draws=rng.integers(32,size=(5000,32));M=rng.normal(size=(4,32));C,k=simultaneous(M,M+.1,draws);C2,k2=simultaneous(M[::-1],(M+.1)[::-1],draws)
    assert np.allclose(C,C2[::-1]) and abs(k-k2)<1e-12
    write(HERE/'STATISTICS_VERIFICATION.json',dict(status='PASS',utc=utc(),tests=['unknown contrast valid under 200 completions','same-eta unknown cancels exactly','unanimous conservative selector and .10 margin','joint max-T invariant to method-column order'],statistics_code_sha256=sha(HERE/'evaluate.py')))
    print('PASS statistics verification')
if __name__=='__main__':main()
