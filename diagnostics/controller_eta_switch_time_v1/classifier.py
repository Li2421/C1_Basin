"""C=1 physical-state logistic regression, exact state leave-one-out."""
from core import *
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

def metrics(y,p,w=None):
    w=np.ones(len(y)) if w is None else w;n=w.sum();pos=(w*y).sum();neg=(w*(1-y)).sum();order=np.argsort(p);sp=p[order];sy=y[order];sw=w[order];starts=np.r_[0,np.flatnonzero(np.diff(sp))+1];wp=np.add.reduceat(sw*sy,starts);wn=np.add.reduceat(sw*(1-sy),starts)
    auc=float(np.sum(wp*(np.cumsum(wn)-wn+.5*wn))/(pos*neg)) if pos and neg else None
    ll=-np.sum(w*(y*np.log(np.clip(p,1e-15,1))+(1-y)*np.log(np.clip(1-p,1e-15,1))))/n
    return dict(AUROC=auc,logloss=float(ll),Brier=float(np.sum(w*(p-y)**2)/n),accuracy=float(np.sum(w*((p>=.5)==y))/n),balanced_accuracy=float(.5*((w*y*(p>=.5)).sum()/pos+(w*(1-y)*(p<.5)).sum()/neg)) if pos and neg else None)

def main():
    prereg();gate();z=np.load(HERE/'CLASSIFIER_INPUT.npz');X=z['X'];y=z['y'];state=z['state'];ctl=z['controller'];assert X.shape[1]==25;assert len(np.unique(state))==64;assert np.isfinite(X).all() and np.isin(y,[0,1]).all();probs=np.empty((3,len(y)));folds=[]
    for s in range(64):
        train=state!=s;test=~train;mean=X[train].mean(0);sd=X[train].std(0);sd[sd<1e-6]=1;scaled=(X-mean)/sd
        probs[2,test]=y[train].mean();fold=dict(state=s,source_n=int(train.sum()),test_n=int(test.sum()),normalization_mean=mean.tolist(),normalization_sd=sd.tolist(),fits=[])
        for model in (0,1):
            f=scaled if model==0 else np.c_[scaled,np.eye(4)[ctl]];f=np.c_[np.ones(len(f)),f];A=f[train];b=y[train]
            def loss(theta):
                q=A@theta;cost=np.sum(np.logaddexp(0,q)-b*q)+.5*(theta[1:]@theta[1:]);gradient=A.T@(expit(q)-b);gradient[1:]+=theta[1:];return cost,gradient
            fitted=minimize(loss,np.zeros(f.shape[1]),jac=True,method='L-BFGS-B',options=dict(maxiter=2000,gtol=1e-8,ftol=1e-12));assert fitted.success,(s,model,fitted.message);probs[model,test]=expit(f[test]@fitted.x);fold['fits'].append(dict(model='physical' if model==0 else 'physical_ID',success=bool(fitted.success),iterations=int(fitted.nit),message=str(fitted.message),coefficients=fitted.x.tolist()))
        folds.append(fold)
    names=['physical','physical_ID','constant'];point={n:metrics(y,p) for n,p in zip(names,probs)};rng=np.random.default_rng(202610130902);bs=[]
    for i in range(5000):
        draws=rng.integers(0,64,64);w=np.bincount(draws,minlength=64)[state];m=[metrics(y,p,w) for p in probs];bs.append([m[1]['AUROC']-m[0]['AUROC'],m[0]['logloss']-m[1]['logloss'],m[2]['Brier']-m[0]['Brier']])
    bs=np.array(bs);CI=np.quantile(bs,[.025,.975],axis=0);sufficient=point['physical']['AUROC']>=.75 and point['constant']['Brier']-point['physical']['Brier']>=.02;small=CI[1,0]<=.02 and CI[1,1]<=.01;status='SUPPORTED' if sufficient and small else 'REJECTED' if CI[0,0]>.02 or CI[0,1]>.01 else 'INCONCLUSIVE'
    result=dict(DEVELOPMENT=True,C=1,features=25,onehot_features=4,eligible_rows=len(y),eligible_states=64,metrics=point,ID_AUROC_gain=float(point['physical_ID']['AUROC']-point['physical']['AUROC']),ID_AUROC_gain_CI=CI[:,0].tolist(),ID_logloss_gain=float(point['physical']['logloss']-point['physical_ID']['logloss']),ID_logloss_gain_CI=CI[:,1].tolist(),Brier_gain_over_constant_CI=CI[:,2].tolist(),state_prediction_sufficient=sufficient,ID_gain_equivalent_small=bool(small),H_state_mediated=status,bootstrap_draws=5000,interpretation='Predictive comparison among four known controllers on survivor switch states; not formal causal mediation or held-out-controller zero-shot certification')
    np.savez_compressed(HERE/'CLASSIFIER_OOF.npz',y=y,state=state,controller=ctl,physical=probs[0],physical_ID=probs[1],constant=probs[2],bootstrap_gain=bs);write(HERE/'CLASSIFIER_FOLDS.json',folds);write(HERE/'CLASSIFIER_RESULTS.json',result);print(json.dumps(result))
if __name__=='__main__':main()
