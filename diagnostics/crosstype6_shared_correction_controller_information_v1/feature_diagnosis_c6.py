"""Separate read-only feature diagnostics; task outcomes never become probe features."""
from common import *
import csv

def predict(X,y,types,source,test,alpha):
    train=np.isin(types,source);tx=X[train];tt=types[train];w=np.asarray([1/(len(source)*np.sum(tt==t)) for t in tt]);valid=np.isfinite(tx);fill=np.sum(np.where(valid,tx,0)*w[:,None],0)/np.maximum(np.sum(valid*w[:,None],0),1e-30);tx=np.where(valid,tx,fill);mu=np.sum(tx*w[:,None],0);scale=np.sqrt(np.sum((tx-mu)**2*w[:,None],0));scale=np.where(scale>1e-6,scale,1.);A=(tx-mu)/scale;target=y[train];center=float(np.sum(w*target));beta=np.linalg.solve((A*w[:,None]).T@A+alpha*np.eye(A.shape[1]),(A*w[:,None]).T@(target-center));B=(np.where(np.isfinite(X[test]),X[test],fill)-mu)/scale
    return center+B@beta

def main():
    assert (HERE/'CONTROLLER_FEATURES_V2.receipt.json').exists()
    with np.load(HERE/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:old={k:z[k] for k in z.files}
    with np.load(HERE/'CONTROLLER_FEATURES_V2.npz',allow_pickle=False) as z:new={k:z[k] for k in z.files}
    with np.load(D/'crosstype5_repair_gate_late_eta_v1/TRAINING_BUNDLE.npz',allow_pickle=False) as z:y=(z['labels'][:,:,26,:]==1).mean((1,2));unknown=(z['labels'][:,:,26,:]<0).mean((1,2))
    assert np.array_equal(old['names'],new['names']);types=new['types'];alltypes=sorted(set(types));results={};predictions={};grid=[1e-3,1e-2,.1,1.,10.,100.,1000.]
    for version,data in [('old24',old),('new39',new)]:
        X=data['features'];out=np.full(len(y),np.nan);folds=[]
        for held in alltypes:
            source=[t for t in alltypes if t!=held];curve=[]
            for alpha in grid:
                errors=[]
                for inner in source:
                    mask=types==inner;p=predict(X,y,types,[t for t in source if t!=inner],mask,alpha);errors.append(float(np.mean((p-y[mask])**2)))
                curve.append(dict(alpha=alpha,source_inner_type_equal_MSE=float(np.mean(errors))))
            chosen=min(curve,key=lambda r:(r['source_inner_type_equal_MSE'],-r['alpha']));mask=types==held;out[mask]=predict(X,y,types,source,mask,chosen['alpha']);folds.append(dict(held_type=held,source_types=source,alpha=chosen['alpha'],source_only_curve=curve,held_MSE=float(np.mean((out[mask]-y[mask])**2))))
        w=np.asarray([1/(len(alltypes)*np.sum(types==t)) for t in types]);center=np.sum(w*y);mse=float(np.sum(w*(out-y)**2));variance=float(np.sum(w*(y-center)**2));results[version]=dict(type_equal_MSE=mse,type_equal_R2=1-mse/variance,controller_equal_R2=1-np.mean((out-y)**2)/np.var(y),folds=folds,missing_original_controllers=data['names'][~np.isfinite(X).all(1)].tolist());predictions[version]=out
    oldnames=['strict_freeze','moving_no_progress',*[f'progress_{a}_{t}' for a in ('safety_baseline','F13','F10') for t in (50,100,200,300)],*[f'projection_{a}' for a in ('safety_baseline','F13','F10')],'rewrite_magnitude','mixed_circulation',*[f'eta13_minus_eta10_{t}' for t in (50,100,200,300)],'nominal_randomness'];correlations=[]
    for version,data,names in [('old24',old,oldnames),('new39',new,new['feature_names'])]:
        for j,name in enumerate(names):
            x=data['features'][:,j];valid=np.isfinite(x);r=float(np.corrcoef(x[valid],y[valid])[0,1]) if valid.sum()>2 and np.std(x[valid])>1e-12 else None;correlations.append(dict(version=version,feature=str(name),Pearson_r_with_training_Safety_success=r,controllers=int(valid.sum()),missing_excluded_not_imputed=int((~valid).sum())))
    raws=read(HERE/'PROBE_FEATURE_RAW.json');previousmissing=old['names'][~np.isfinite(old['features']).all(1)].tolist();retry=[]
    for name in previousmissing:
        rr=[r for r in raws if r['controller']==name];retry.append(dict(controller=name,new_attempts=len(rr),successful_700_step_probes=sum(not r['failed'] for r in rr),failures=[{k:r[k] for k in ('state_index','arm','reason','steps')} for r in rr if r['failed']]));assert len(rr)==32
    write(HERE/'FEATURE_CORRELATIONS.json',correlations);write(HERE/'FEATURE_LOTO_LINEAR_DIAGNOSIS.json',dict(utc=utc(),target='C5 training-state Safety q0/1 success lower, no fresh-evaluation outcomes',target_unknown_fraction_by_controller=unknown,versions=results,not_used_for_selector_training_or_checkpoint_selection=True));write(HERE/'PREVIOUS_MISSING_CONTROLLERS_REATTEMPT.json',retry)
    np.savez_compressed(HERE/'FEATURE_LINEAR_PREDICTIONS.npz',names=new['names'],types=types,target=y,**predictions)
    with (HERE/'FEATURE_CORRELATIONS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(correlations[0]));w.writeheader();w.writerows(correlations)
    seal('FEATURE_DIAGNOSIS.receipt.json',[HERE/n for n in ('FEATURE_CORRELATIONS.json','FEATURE_CORRELATIONS.csv','FEATURE_LOTO_LINEAR_DIAGNOSIS.json','PREVIOUS_MISSING_CONTROLLERS_REATTEMPT.json','FEATURE_LINEAR_PREDICTIONS.npz','feature_diagnosis_c6.py','CONTROLLER_FEATURES_V2.receipt.json')],no_fresh_outcomes=True)
    print('Feature diagnosis complete; old missing C attempted',len(previousmissing),flush=True)

if __name__=='__main__':main()
