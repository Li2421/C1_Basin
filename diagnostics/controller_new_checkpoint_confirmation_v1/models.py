from core import *
import csv

def csvwrite(name,rows):
    with (HERE/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(clean(rows))

def preprocess(source,target):
    keep=np.isfinite(source).any(0);src=source[:,keep];tar=target[:,keep];mean=np.nanmean(src,axis=0);src=np.where(np.isfinite(src),src,mean);tar=np.where(np.isfinite(tar),tar,mean);sd=src.std(0);scale=np.where(sd<1e-6,1.,sd);xs=(src-mean)/scale;xt=(tar-mean)/scale;_,sing,vt=np.linalg.svd(xs,full_matrices=False)
    for v in vt:
        if v[np.argmax(abs(v))]<0:v*=-1
    pc=vt[:2];return xs@pc.T,xt@pc.T,dict(kept_columns=np.flatnonzero(keep),mean=mean,scale=scale,components=pc,variance_ratio=sing[:2]**2/(sing**2).sum(),source_missing_cells=int((~np.isfinite(source[:,keep])).sum()),target_missing_cells=int((~np.isfinite(target[:,keep])).sum()))

def fit(x,y):
    mu=x.mean(0);ym=y.mean();xc=x-mu;beta=np.linalg.solve(xc.T@xc+np.eye(x.shape[1]),xc.T@(y-ym));return beta,ym-mu@beta

def main():
    prereg();rf=read(HERE/'R5_FROZEN.receipt.json');assert sha(HERE/'R5_FINAL_FEATURES.npz')==rf['features_sha256'];base=read(HERE/'FINAL_BASE_PROFILES.json');names=base['best_controller_order'];z=np.load(R7/'FEATURES.npz');srcnames=z['controller_names'].tolist();oldraw=z['R5'][[srcnames.index(n) for n in names]].astype(float);keep=np.ones(160,bool);keep[105:110]=False;oldraw=oldraw[:,keep];new=np.load(HERE/'R5_FINAL_FEATURES.npz');finalnames=new['controller_names'].tolist();target=new['features'].astype(float)[:,keep];phase=read(ROOT/'diagnostics/controller_phase_decomposition_v1/results.json');y=np.array([phase['effects'][n]['E_S500']['point'] for n in names]);choices=[];models=[];distances=[]
    for i,n in enumerate(finalnames):
        sibling=base['profiles'][n]['parent_best_name'];bi=names.index(sibling);record=dict(controller=n,sibling_best=sibling,eligible=True)
        for scheme,source in [('primary',[j for j in range(14) if j!=bi]),('secondary',list(range(14)))]:
            x,h,prep=preprocess(oldraw[source],target[i:i+1]);beta,intercept=fit(x,y[source]);pred=float(intercept+h[0]@beta);action='F10' if pred<.05 else 'S500';record[scheme+'_Ehat']=pred;record[scheme+'_action']=action;record[scheme+'_source_mean_E']=float(y[source].mean());models.append(dict(controller=n,scheme=scheme,sources=[names[j] for j in source],excluded_sibling=sibling if scheme=='primary' else None,preprocessing=prep,beta=beta,intercept=intercept,prediction=pred,action=action))
        choices.append(record)
        # Descriptive distance only, using all14 source scaling; missing target cells use source means.
        mean=np.mean(oldraw,axis=0);sd=oldraw.std(0);scale=np.where(sd<1e-6,1.,sd);filled=np.where(np.isfinite(target[i]),target[i],mean);difference=filled-oldraw[bi];distances.append(dict(controller=n,sibling_best=sibling,R5_raw_L2=float(np.linalg.norm(difference)),R5_source_standardized_L2=float(np.linalg.norm(difference/scale)),missing_target_cells=int((~np.isfinite(target[i])).sum()),distance_imputation='all14 source means, descriptive only'))
    freeze(HERE/'SELECTOR_MODELS.json',dict(schema='round12_frozen_E_selector_v1',choices=choices,models=models,source_E_labels=y,source_controller_order=names,main='primary sibling excluded',threshold=.05,menu=['F10','F13','S300','S500'],oracle_only=['F2','F4'],source_average_action='S500',prior_action='F10',R5_features_sha256=rf['features_sha256'],source_R5_sha256=sha(R7/'FEATURES.npz'),source_E_file_sha256=sha(ROOT/'diagnostics/controller_phase_decomposition_v1/results.json'),final_success_labels_read_or_generated=False));write(HERE/'SIBLING_FEATURE_DISTANCES.json',distances);csvwrite('FROZEN_CHOICES.csv',choices)
    freeze(HERE/'SELECTOR_FROZEN.receipt.json',dict(utc=utc(),models_sha256=sha(HERE/'SELECTOR_MODELS.json'),R5_receipt_sha256=sha(HERE/'R5_FROZEN.receipt.json'),preregistration_sha256=sha(HERE/'PREREGISTRATION.md'),final_success_labels_read_or_generated=False,primary_sources_exclude_sibling=True))
    print(json.dumps(clean(dict(frozen=True,choices=choices))))
if __name__=='__main__':main()
