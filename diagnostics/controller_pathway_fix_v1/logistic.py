from util import *
from scipy.optimize import minimize
from scipy.special import softmax,logsumexp
def fit_task(t):
    out=HERE/'logistic'/task_name(dict(t,arm='F4'));out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    d=Data(t);S=np.array([d.coordinates(n,'F1_PCA2') for n in d.sources]);mean=S.mean(0);sd=S.std(0);scale=np.where(sd>1e-6,sd,1.)
    X=np.c_[np.ones(len(S)),(S-mean)/scale];y=np.array([list(REGIMES).index(d.regimes[n]) for n in d.sources]);mask=np.unique(y)
    # Absent classes get exact zero probability rather than target information.
    Y=np.eye(len(mask))[np.searchsorted(mask,y)]
    def fn(flat):
        W=flat.reshape(3,len(mask));z=X@W;p=softmax(z,axis=1);val=(logsumexp(z,axis=1)-(Y*z).sum(1)).sum()+.5*(W[1:]**2).sum()
        grad=X.T@(p-Y);grad[1:]+=W[1:];return val,grad.ravel()
    res=minimize(fn,np.zeros(3*len(mask)),jac=True,method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-12,'gtol':1e-8});assert res.success,(res.message,t)
    W=res.x.reshape(3,len(mask));H=np.c_[np.ones(len(d.held)),(np.array([d.coordinates(n,'F1_PCA2') for n in d.held])-mean)/scale];p=softmax(H@W,axis=1);probs=np.zeros((len(d.held),4));probs[:,mask]=p
    choice=np.where(probs.max(1)>=.6,REGIMES[probs.argmax(1)],d.src);choices=np.tile(choice[None,:,None],(3,1,len(d.test_states)))
    np.savez_compressed(out/'predictions.npz',choices=choices,regime_probabilities=probs,held_controller_names=d.held)
    np.savez_compressed(out/'classifier.npz',weights=W,classes=REGIMES[mask],mean=mean,scale=scale,pca_mean=d.pca_mean,pca_scale=d.pca_scale,pca_components=d.components[:2])
    write(out/'complete.json',dict(status='COMPLETE',task=t,arm='F4',regularization=1.,threshold=.6,max_iterations=2000,solver_iterations=int(res.nit),source_controller_names=d.sources,source_regimes=d.regimes,held_predictions=choice.tolist(),confidence=probs.max(1).tolist(),target_labels_used=False,test_labels_read=False,not_a_success_probability_predictor=True,training_deterministic=True))
def main(bed):
    for t in task_list(bed):fit_task(t)
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'blend')
