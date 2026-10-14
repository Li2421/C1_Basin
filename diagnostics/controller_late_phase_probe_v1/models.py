from core import *
from scipy.stats import spearmanr
import csv
REPS=['R5-PCA2','R6-Delta','R6-PCA2','R5-PCA2+R6-Delta'];MENU=['F10','F13','S300','S500']
def csvwrite(name,rows):
    with (HERE/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(clean(rows))
def preprocessing(raw,source,pca=False):
    finite=np.isfinite(raw[source]).sum(0);keep=finite>0;assert keep.any();v=raw[:,keep];mean=np.nanmean(v[source],axis=0);filled=np.where(np.isfinite(v),v,mean);sd=filled[source].std(0);scale=np.where(sd<1e-6,1.,sd);z=(filled-mean)/scale;meta=dict(kept_columns=np.flatnonzero(keep),mean=mean,scale=scale,missing_cells_source=int((~np.isfinite(v[source])).sum()),target_imputation='source mean only')
    if pca:
        _,sing,vt=np.linalg.svd(z[source],full_matrices=False)
        for v in vt:
            if v[np.argmax(abs(v))]<0:v*=-1
        z=z@vt[:2].T;meta.update(components=vt[:2],variance_ratio=sing[:2]**2/(sing**2).sum())
    return z,meta

def fit_predict(x,y,source,targets):
    sx=x[source];mu=sx.mean(0);ym=y[source].mean(0);xc=sx-mu;beta=np.linalg.solve(xc.T@xc+np.eye(x.shape[1]),xc.T@(y[source]-ym));intercept=ym-mu@beta
    return intercept+x[targets]@beta,dict(beta=beta,intercept=intercept,source_indices=source)
def G(e,l):return 'F10' if e<.05 else 'F13' if l<.05 else 'S500'
def GE(e):return 'F10' if e<.05 else 'S500'
def main():
    prereg();frozen=labels_allowed();phase=read(ROOT/'diagnostics/controller_phase_decomposition_v1/results.json');r6=np.load(HERE/'R6_FEATURES.npz');allnames=r6['controller_names'].tolist();r5=np.load(R7/'FEATURES.npz');r5names=r5['controller_names'].tolist();keep=np.ones(160,bool);keep[105:110]=False;raw5=r5['R5'][[r5names.index(n) for n in allnames]][:,keep].astype(float);raw6=r6['raw'];delta=r6['delta'];names=read(R10/'protocol.json')['controller_order'];blend=read(HERE/'probe_protocol.json')['blend_order'];realids=[allnames.index(n) for n in names];blendids=[allnames.index(n) for n in blend]
    if not np.isfinite(delta).all():
        write(HERE/'STAGE_B.json',dict(status='INCONCLUSIVE',H_L='INCONCLUSIVE',reason='At least one controller has no complete paired probe; do not fabricate descriptor.'));raise RuntimeError('R6 descriptor missing: Stage B/C unavailable')
    y=np.array([[phase['effects'][n][k]['point'] for k in ['E_S500','L_S500','N']] for n in names]);z=np.load(R10/'LABELS.npz');anames=z['controllers'].tolist();arms=z['arms'].tolist();rates=z['success'].mean(axis=(2,3))[[anames.index(n) for n in names]];available=z['available'][[anames.index(n) for n in names]];source_mean_baseline=np.array([y[np.arange(14)!=i].mean(0) for i in range(14)]);baseline_metrics={label:dict(sign_accuracy=float((np.sign(source_mean_baseline[:,k])==np.sign(y[:,k])).mean()),RMSE=float(np.sqrt(np.mean((source_mean_baseline[:,k]-y[:,k])**2)))) for k,label in enumerate(['E','L','N'])}
    preds={};metrics={};folds=[];table=[];nn=[]
    def representation(rep,source):
        if rep=='R5-PCA2':return preprocessing(raw5,source,True)
        if rep=='R6-PCA2':return preprocessing(raw6,source,True)
        if rep=='R6-Delta':return preprocessing(delta,source)
        a,am=preprocessing(raw5,source,True);b,bm=preprocessing(delta,source);return np.c_[a,b],dict(R5=am,R6=bm)
    for rep in REPS:
        pred=np.zeros((14,3))
        for i in range(14):
            source=[j for j in range(14) if j!=i];sourceids=[realids[j] for j in source];x,prep=representation(rep,sourceids);p,model=fit_predict(x[realids],y,source,[i]);pred[i]=p[0];folds.append(dict(representation=rep,held=names[i],sources=[names[j] for j in source],preprocessing=prep,model=model))
            if rep==REPS[-1]:
                nearest=source[int(np.linalg.norm(x[sourceids]-x[realids[i]],axis=1).argmin())];chosen=MENU[int(np.argmax(rates[nearest,[arms.index(a) for a in MENU]]))];nn.append(dict(controller=names[i],nearest=names[nearest],action=chosen))
        assert np.max(abs(pred[:,0]-pred[:,1]-pred[:,2]))<1e-12;preds[rep]=pred;metrics[rep]={}
        for k,label in enumerate(['E','L','N']):
            rmse=float(np.sqrt(np.mean((pred[:,k]-y[:,k])**2)));metrics[rep][label]=dict(Spearman=float(spearmanr(pred[:,k],y[:,k]).statistic),sign_accuracy=float((np.sign(pred[:,k])==np.sign(y[:,k])).mean()),RMSE=rmse,RMSE_ratio_vs_source_mean=rmse/baseline_metrics[label]['RMSE'])
            for i,n in enumerate(names):table.append(dict(controller=n,representation=rep,effect=label,truth=y[i,k],prediction=pred[i,k],source_mean=source_mean_baseline[i,k]))
    candidates=[rep for rep in REPS if rep!='R5-PCA2'];winners=[rep for rep in candidates if metrics[rep]['L']['Spearman']>=.6 and metrics[rep]['L']['RMSE_ratio_vs_source_mean']<=.7];write(HERE/'STAGE_B.json',dict(status='COMPLETE',H_L='SUPPORTED' if winners else 'REJECTED',passing_representations=winners,metrics=metrics,source_mean_baseline=baseline_metrics,real_controller_order=names,folds=folds,label_access_after_R6_receipt=frozen['utc'],all_evidence_DEVELOPMENT=True))
    csvwrite('STAGE_B_PREDICTIONS.csv',table)
    # Real LOCO choices frozen before accessing any blend task labels.
    real_choices=[]
    for i,n in enumerate(names):
        e=preds['R5-PCA2'][i,0];l=preds['R6-Delta'][i,1];source=[j for j in range(14) if j!=i];source_action=MENU[int(rates[source][:,[arms.index(a) for a in MENU]].mean(0).argmax())]
        real_choices.append(dict(controller=n,Ehat=e,Lhat=l,G=G(e,l),G_E=GE(e),source_average=source_action,nearest_neighbor=nn[i]['action'],nearest_controller=nn[i]['nearest']))
    models={};blend_e=None;blend_l=None
    for rep,label in [('R5-PCA2','E'),('R6-Delta','L')]:
        x,prep=representation(rep,realids);yp,model=fit_predict(x[realids],y,list(range(14)),list(range(14)))
        beta=np.asarray(model['beta']);intercept=np.asarray(model['intercept']);targetpred=intercept+x[blendids]@beta;models[label]=dict(representation=rep,preprocessing=prep,model=model)
        if label=='E':blend_e=targetpred[:,0]
        else:blend_l=targetpred[:,1]
    source_action=MENU[int(rates[:,[arms.index(a) for a in MENU]].mean(0).argmax())];choices=[dict(controller=n,Ehat=blend_e[i],Lhat=blend_l[i],G=G(blend_e[i],blend_l[i]),G_E=GE(blend_e[i]),source_average=source_action) for i,n in enumerate(blend)]
    freeze(HERE/'SELECTOR_MODELS.json',dict(models=models,real_choices=real_choices,blend_choices=choices,menu=MENU,source_controller_order=names,R6_sha256=frozen['features_sha256'],rules=dict(E_threshold=.05,L_threshold=.05,main_L_representation='R6-Delta'),blend_task_labels_read=False))
    freeze(HERE/'SELECTOR_FROZEN.receipt.json',dict(utc=utc(),models_sha256=sha(HERE/'SELECTOR_MODELS.json'),stage_B_sha256=sha(HERE/'STAGE_B.json'),R6_receipt_sha256=sha(HERE/'R6_FROZEN.receipt.json'),blend_task_labels_read=False))
    print(json.dumps(clean(dict(H_L='SUPPORTED' if winners else 'REJECTED',metrics=metrics,blend_choices=choices))))
if __name__=='__main__':main()
