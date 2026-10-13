"""Q-identification, mismatch and eta-effect diagnostics; DEVELOPMENT only."""
import numpy as np
from .study import HERE,read,write,FAMILIES
from .learn import FOLDS,CONDITIONS

def corr(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    return float(np.corrcoef(a,b)[0,1]) if a.std()>1e-12 and b.std()>1e-12 else None

def center(a,axes):
    for axis in axes:a=a-a.mean(axis,keepdims=True)
    return a

def gain_bounds(baseline,model,lo,hi):
    a=(baseline-lo)**2-(model-lo)**2;b=(baseline-hi)**2-(model-hi)**2
    return np.minimum(a,b).mean((1,2)),np.maximum(a,b).mean((1,2))

def run():
    rows=[];contrasts=[];boots={};saved={}
    for fold,source in FOLDS.items():
        predictions={}
        for condition in CONDITIONS:
            kind='eta_only' if condition=='eta_only' else 'state_field_compact_bilinear'
            root=HERE/'models'/fold/condition
            d=np.load(root/'data/source128/dataset.npz')
            z=[]
            for seed in (17,23,41):
                folder=root/'fits/source128'/kind/f'seed{seed}';assert (folder/'complete.json').exists()
                v=np.load(folder/'source_predictions.npz')['logits'];z.append(1/(1+np.exp(-np.clip(v,-80,80))))
            predictions[condition]=np.mean(z,0)
        for scene in sorted(set(d['scene'])):
            ix=np.flatnonzero((d['scene']==scene)&(d['original_split']=='validation')&np.isin(d['controller'],FAMILIES))
            ix=np.asarray(sorted(ix,key=lambda i:(d['state_uid'][i],d['controller'][i])))
            n=len(ix)//5;kk=np.asarray([0,5,10,15]);shape=(n,5,4)
            s=d['success'][ix][:,kk].reshape(shape);f=d['failure'][ix][:,kk].reshape(shape);u=d['unknown'][ix][:,kk].reshape(shape)
            assert np.all(s+f+u==4)
            lo=s/4;hi=(s+u)/4;y=s/np.maximum(s+f,1);valid=s+f>0
            boot=np.random.default_rng(2026101105).integers(n,size=(20000,n));p={c:v[ix][:,kk].reshape(shape) for c,v in predictions.items()}
            matched=p['operator'];off=np.arange(n)[:,None]!=np.arange(n)[None,:]
            mismatched=np.broadcast_to(matched[None],(n,n,5,4));basewrong=np.stack([np.mean(np.delete(matched,j,axis=1),axis=1) for j in range(5)],axis=1)
            # Average squared error over donor predictions, not error of the
            # donor-mean predictor; these are different estimands.
            matched_error=(matched-y)**2
            mismatch_error=((mismatched-y[:,None])**2*off[:,:,None,None]).sum(1)/(n-1)
            wrong_error=np.stack([np.mean((np.delete(matched,j,axis=1)-y[:,j,None])**2,axis=1) for j in range(5)],axis=1)
            values={'eta_only':p['eta_only'],'alias':p['alias'],'operator':matched}
            for condition,q in values.items():
                mse=((q-y)**2*valid).sum((1,2))/np.maximum(valid.sum((1,2)),1)
                selected=q.argmax(-1);success=np.take_along_axis(lo,selected[...,None],-1)[...,0]
                qq=center(q,[0]);yy=center(y,[0])
                ci,cj=np.triu_indices(5,1);contrast=(q[:,ci]-q[:,cj]);truth=(y[:,ci]-y[:,cj]);good=valid[:,ci]&valid[:,cj]
                rows.append(dict(fold=fold,scene=str(scene),panel='source_validation' if source is None or scene==source else 'held_scene_development',
                    condition=condition,N=n,controllers=5,candidates=4,Q_MSE=float(mse.mean()),
                    controller_contrast_MSE=float(np.mean((contrast-truth)[good]**2)),
                    controller_contrast_corr=corr(contrast[good],truth[good]),
                    state_effect_corr=corr(qq[valid],yy[valid]),
                    eta_centered_corr=corr(center(q,[-1])[valid],center(y,[-1])[valid]),
                    controller_eta_corr=corr(center(q,[1,2])[valid],center(y,[1,2])[valid]),
                    selected_verified_success=float(success.mean()),numerical_unknown=int(u.sum()),
                    selected_eta_counts=np.bincount(selected.ravel(),minlength=4).tolist()))
                saved[f'{fold}_{scene}_{condition}']=q
            def record(label,diff,lower,upper):
                dist=diff[boot].mean(1);dl=lower[boot].mean(1);dh=upper[boot].mean(1)
                name=f'{fold}_{scene}_{label}';boots[name]=dist;boots[name+'_lo']=dl;boots[name+'_hi']=dh
                contrasts.append(dict(fold=fold,scene=str(scene),comparison=label,N=n,
                    improvement=float(diff.mean()),CI95=np.quantile(dist,[.025,.975]).tolist(),
                    censor_robust_CI95=[float(np.quantile(dl,.025)),float(np.quantile(dh,.975))]))
            for name in ('eta_only','alias'):
                diff=(((p[name]-y)**2-matched_error)*valid).sum((1,2))/np.maximum(valid.sum((1,2)),1)
                lower,upper=gain_bounds(p[name],matched,lo,hi);record(name+'_minus_operator_Q_MSE',diff,lower,upper)
            for name,errors,donors in [('mismatched_state',mismatch_error,[(np.roll(matched,j,axis=0)) for j in range(1,n)]),
                                       ('wrong_controller',wrong_error,[np.roll(matched,j,axis=1) for j in range(1,5)])]:
                diff=((errors-matched_error)*valid).sum((1,2))/np.maximum(valid.sum((1,2)),1)
                bounds=[gain_bounds(q,matched,lo,hi) for q in donors]
                lower=np.mean([b[0] for b in bounds],axis=0);upper=np.mean([b[1] for b in bounds],axis=0)
                record(name+'_minus_matched_Q_MSE',diff,lower,upper)
            saved[f'{fold}_{scene}_within_scene_donors']=mismatched
            saved[f'{fold}_{scene}_success']=s;saved[f'{fold}_{scene}_unknown']=u
    np.savez_compressed(HERE/'development_predictions.npz',**saved)
    np.savez_compressed(HERE/'development_bootstrap.npz',**boots)
    write(HERE/'development_results.json',dict(rows=rows,contrasts=contrasts,
        warning='DEVELOPMENT only. Source-validation panels were used for checkpoint selection. No final generalization claim. Four fixed eta labels are mechanism probes, not formalK16/B15 evaluation.',
        causal_root_test='Same architecture,optimizer,labels and weighting; only static controller function information differs.'))
    print('Development root-repair diagnostics saved',flush=True)

if __name__=='__main__':run()
