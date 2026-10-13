"""Zero-rollout descriptive regime audit. Old artifacts are read-only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from common import *
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,expit

def feature_path(name):
    a=PREVIOUS/'static_features'/f'{name}.npz'
    return a if a.exists() else PREVIOUS/'stage_b/static_features'/f'{name}.npz'

def logistic(x,y,test):
    classes=np.unique(y);n,d=x.shape
    if len(classes)==1:return int(classes[0])
    yy=(y[:,None]==classes[None]).astype(float)
    xx=np.c_[x,np.ones(n)];tt=np.r_[test,1.]
    def objective(v):
        w=v.reshape(d+1,len(classes));a=xx@w;p=np.exp(a-logsumexp(a,axis=1,keepdims=True))
        val=np.sum(logsumexp(a,axis=1)-(a*yy).sum(1))+.5*np.square(w[:-1]).sum()
        g=xx.T@(p-yy);g[:-1]+=w[:-1]
        return val,g.ravel()
    z=minimize(objective,np.zeros((d+1)*len(classes)),jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12})
    assert z.success,z.message
    return int(classes[(tt@z.x.reshape(d+1,len(classes))).argmax()])

def main():
    receipt(HERE/'PREREGISTRATION_STAGE_0.md')
    a=read(PREVIOUS/'protocol.json');b=read(PREVIOUS/'stage_b/protocol.json')
    names=a['controller_order']+b['controller_order']
    labels=np.concatenate((np.load(PREVIOUS/'LABELS.npz')['outcomes'][:,:128,:,:4],np.load(PREVIOUS/'stage_b/LABELS.npz')['outcomes']),axis=0)
    mask=(labels!=-2).any(-1);assert np.all(mask.sum(1)==64)
    valid=((labels==1)|(labels==0)).sum(-1);succ=(labels==1).sum(-1)
    q=np.divide(succ,valid,out=np.full_like(succ,np.nan,dtype=float),where=valid>0)
    curve=np.nanmean(q,axis=1);best=curve.argmax(-1)
    lower=(labels==1).mean(-1)*mask;unknown=(labels==-1).mean(-1)*mask
    def curves(ids):
        mm=mask[:,ids].sum(1)
        return lower[:,ids].sum(1)/mm,unknown[:,ids].sum(1)/mm
    lo,uu=curves(np.arange(128));cols=[(c,j) for c in range(14) for j in (10,13)]
    def contrasts(l,u):
        return np.array([l[c,best[c]]-l[c,j]-(u[c,j] if best[c]!=j else 0) for c,j in cols])
    obs=contrasts(lo,uu);rng=np.random.default_rng(2026101301)
    draws=rng.integers(128,size=(5000,128));boots=np.array([contrasts(*curves(ids)) for ids in draws])
    se=boots.std(0,ddof=1);critical=float(np.quantile(np.max(np.abs((boots-obs)/np.where(se>1e-12,se,1)),axis=1),.95))
    rows=[]
    for c,name in enumerate(names):
        comparisons={}
        for j in (10,13):
            k=cols.index((c,j));upper=lo[c,best[c]]+uu[c,best[c]]-lo[c,j] if best[c]!=j else 0.
            comparisons[str(j)]=dict(conservative_difference=float(obs[k]),difference_upper=float(upper),ci95_simultaneous=[float(obs[k]-critical*se[k]),float(obs[k]+critical*se[k])])
        rows.append(dict(controller=name,best_eta=int(best[c]),observed_valid_curve=curve[c].tolist(),success_lower=lo[c].tolist(),success_upper=(lo[c]+uu[c]).tolist(),comparisons=comparisons,requested=int(mask[c].sum()*4),unknown=int((labels[c]==-1).sum())))
    spaces={};classes=[2,4,10,13]
    for arm in ('R3','R4_action','R3_R4_action'):
        z=[]
        for name in names:
            f=np.load(feature_path(name))
            assert list(f['state_uids'][:128])==[s['state_uid'] for s in a['states'][:128]]
            z.append(np.c_[f['R3'][:128],f['R4_action'][:128]] if arm=='R3_R4_action' else f[arm][:128])
        z=np.array(z,dtype=float);assert np.isfinite(z).all()
        nn=[];lg=[];major=[];dist=[];nearest=[]
        for held in range(14):
            src=[c for c in range(14) if c!=held];mu=z[src].mean((0,1));sd=z[src].std((0,1));sd=np.where(sd>1e-6,sd,1.)
            x=((z[src]-mu)/sd).mean(1);h=((z[held]-mu)/sd).mean(0);dd=np.linalg.norm(x-h,axis=1)
            near=src[int(dd.argmin())];nearest.append(names[near]);nn.append(int(best[near]));dist.append(dict(zip([names[c] for c in src],dd.tolist())))
            lg.append(logistic(x,best[src],h));major.append(int(np.bincount(best[src],minlength=16).argmax()))
        def summarize(pred):
            conf=np.zeros((4,4),int)
            for t,p in zip(best,pred):conf[classes.index(t),classes.index(p)]+=1
            return dict(accuracy=float(np.mean(best==pred)),predictions=pred,confusion=conf.tolist())
        spaces[arm]=dict(nearest_controller=nearest,source_normalized_distances=dist,NN=summarize(nn),logistic=summarize(lg),majority=summarize(major),classes=classes)
    priors={};decomp=[]
    for stage in ('stage_a','stage_b'):
        for fold in (1,4):
            priors[f'{stage}_fold{fold}']={}
            out=np.load(PREVIOUS/'LABELS.npz')['outcomes'][fold,128:]
            disc=out[:,:,:4];target=int(np.divide((disc==1).sum((0,2)),((disc==1)|(disc==0)).sum((0,2)),out=np.zeros(16),where=((disc==1)|(disc==0)).sum((0,2))>0).argmax())
            test=out[:,:,4:8];sy=(test==1).mean(-1);su=(test==-1).mean(-1)
            for arm in ('R3','R4_action','R3_R4_action'):
                pfile=PREVIOUS/'controller_prior'/f'{stage}_fold{fold}_{arm}.npz';pr=np.load(pfile)['eta_prior'];priors[f'{stage}_fold{fold}'][arm]=pr.tolist()
                fit=(PREVIOUS/'fits' if stage=='stage_a' else PREVIOUS/'stage_b/fits')/f'fold{fold}_{arm}'
                choice=np.load(fit/'predictions.npz')['held'].argmax(-1)
                selected=np.take_along_axis(sy[None],choice[...,None],axis=-1)[...,0]
                prior=sy[:,pr].T
                level=(sy[:,target][None]-prior).mean(0)
                state=(prior-selected).mean(0)
                assert np.allclose(level+state,(sy[:,target][None]-selected).mean(0))
                bounds=[]
                for component in ('regime','state'):
                    ca=np.full_like(choice,target) if component=='regime' else np.broadcast_to(pr[:,None],choice.shape)
                    cb=np.broadcast_to(pr[:,None],choice.shape) if component=='regime' else choice
                    yl=np.take_along_axis(sy[None],ca[...,None],-1)[...,0];yr=np.take_along_axis(sy[None],cb[...,None],-1)[...,0]
                    ul=np.take_along_axis(su[None],ca[...,None],-1)[...,0];ur=np.take_along_axis(su[None],cb[...,None],-1)[...,0]
                    v=(yl-yr-ur*(ca!=cb)).mean(0);up=(yl+ul*(ca!=cb)-yr).mean(0)
                    boot=v[np.random.default_rng(2026101302).integers(64,size=(5000,64))].mean(1)
                    bounds.append(dict(component=component,lower=float(v.mean()),upper=float(up.mean()),ci95_lower_bound=np.quantile(boot,[.025,.975]).tolist()))
                decomp.append(dict(stage=stage,fold=fold,controller=names[fold],arm=arm,target_reference_eta=target,learned_prior_eta=pr.tolist(),regime_component_lower_success_difference=float(level.mean()),state_component_lower_success_difference=float(state.mean()),total_lower_success_regret=float((level+state).mean()),paired_unknown_bounds=bounds))
    result=dict(status='COMPLETE EXPLORATORY',created_utc=utc(),new_rollouts=0,reused_training_seed_slots=int(mask.sum()*4),valid_seed_slots=int(((labels==0)|(labels==1)).sum()),unknown_seed_slots=int((labels==-1).sum()),controller_order=names,rows=rows,regime_counts={str(j):int((best==j).sum()) for j in classes},representations=spaces,frozen_controller_priors=priors,regret_decomposition=decomp,bootstrap_draws=5000,bootstrap_seed=2026101301,max_t_critical=critical,notes=['No retraining critic; logistic is an exploratory regime classifier','Best eta is selected on the same sparse TRAIN labels: descriptive and optimistic; not a new confirmation','Old background claim that all per-state point estimates are <= learned prior is too strong: Stage A combined representation +2.21pp CI[-0.01,4.44] pp; no reliable positive state residual'])
    write(HERE/'STAGE_0_RESULTS.json',result)
    write(HERE/'DATA_REUSE_INDEX.json',dict(files=[dict(path=str(p),sha256=sha(p)) for p in [PREVIOUS/'protocol.json',PREVIOUS/'stage_b/protocol.json',PREVIOUS/'LABELS.npz',PREVIOUS/'stage_b/LABELS.npz',PREVIOUS/'results.json',PREVIOUS/'SOURCE_REGIME_COVERAGE.json',PREVIOUS/'HANDOFF.md',PREVIOUS/'REPORT.md']],static_features=[dict(path=str(feature_path(n)),sha256=sha(feature_path(n))) for n in names]))
    print(json.dumps(dict(status=result['status'],regime_counts=result['regime_counts'],LOO={k:{m:v[m]['accuracy'] for m in ('NN','logistic','majority')} for k,v in spaces.items()},reused=result['reused_training_seed_slots'])))

if __name__=='__main__':main()
