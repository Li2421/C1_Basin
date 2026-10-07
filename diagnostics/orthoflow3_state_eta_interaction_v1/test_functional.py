"""Synthetic invariants for the audit, not learned-performance evidence."""
import numpy as np
from scipy.special import expit
from .audit import contrasts,derange
def run():
    q=np.array([[1.,0.],[0.,1.]])
    met,_=contrasts(q,q)
    assert met['reversal_direction_accuracy']==1 and met['strong_sign_accuracy']==1
    a=np.array([[.2],[.4]]);b=np.array([[.1,.3]])
    met,_=contrasts(q,a+b)
    assert met['delta_pred_quantiles']==[0.]*5 and met['reversal_recall']==0
    met,_=contrasts(q,expit(a+b))
    assert met['reversal_recall']==0,'Sigmoid additive logits can change contrast but never ranking'
    for seed in range(20):assert np.all(derange(8,seed)!=np.arange(8))
    bad=q.copy();bad[0,0]=np.nan
    assert contrasts(bad,q)[0]['quadruples']==0
    return {'passed':True,'checks':['exact interaction','additive Q','sigmoid additive logits','derangements','missing Q excluded']}
if __name__=='__main__':print(run())
