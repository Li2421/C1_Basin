"""Small, auditable source-only alternatives for development iteration."""
import numpy as np
from selector import Regression,weights,ARMS,BASELINE,best_arm


class Transform:
    def fit(self,x,types,pca_dimensions=None):
        x=np.asarray(x,float);w=weights(types);finite=np.isfinite(x)
        self.fill=(np.where(finite,x,0)*w[:,None]).sum(0)/np.maximum((finite*w[:,None]).sum(0),1e-30)
        x=np.where(finite,x,self.fill);self.mean=(x*w[:,None]).sum(0)
        self.scale=np.sqrt(((x-self.mean)**2*w[:,None]).sum(0));self.scale=np.where(self.scale>1e-6,self.scale,1.)
        self.components=None
        if pca_dimensions is not None:
            _,_,v=np.linalg.svd(((x-self.mean)/self.scale)*np.sqrt(w[:,None]),full_matrices=False)
            self.components=v[:pca_dimensions]
        return self

    def apply(self,x):
        z=(np.where(np.isfinite(x),x,self.fill)-self.mean)/self.scale
        return z if self.components is None else z@self.components.T

    def to_dict(self):return {k:None if v is None else v.tolist() for k,v in vars(self).items()}

    @classmethod
    def from_dict(cls,record):
        obj=cls()
        for k,v in record.items():setattr(obj,k,None if v is None else np.asarray(v,float))
        return obj


class Predictor:
    def __init__(self,config,pca_dimensions=None):
        self.config=dict(config);self.pca_dimensions=pca_dimensions

    def fit(self,x,rates,types):
        x,rates,types=np.asarray(x,float),np.asarray(rates,float),np.asarray(types)
        assert rates.shape==(len(x),len(ARMS)) and np.isfinite(rates).all()
        self.source_types=sorted(set(types.tolist()));kind=self.config['kind']
        if kind=='ridge_margin':
            self.ridge=Regression(self.config['penalty'],self.pca_dimensions).fit(x,rates,types)
            return self
        self.transform=Transform().fit(x,types,self.pca_dimensions);z=self.transform.apply(x)
        target=rates-rates[:,BASELINE,None]
        if kind=='nearest_type':
            self.centers=np.stack([z[types==t].mean(0) for t in self.source_types])
            self.values=np.stack([target[types==t].mean(0) for t in self.source_types])
        elif kind=='shallow_tree':
            w=weights(types)
            def node(indices,depth):
                ww=w[indices];yy=target[indices];mean=(yy*ww[:,None]).sum(0)/ww.sum()
                record={'value':mean.tolist(),'controllers':len(indices)}
                if depth>=self.config['depth'] or len(indices)<6:return record
                parent=float(np.sum(ww[:,None]*(yy-mean)**2));best=None
                for feature in range(z.shape[1]):
                    ordered=indices[np.argsort(z[indices,feature],kind='stable')]
                    values=z[ordered,feature];cw=np.cumsum(w[ordered]);cy=np.cumsum(w[ordered,None]*target[ordered],axis=0)
                    cyy=np.cumsum(w[ordered,None]*target[ordered]**2,axis=0)
                    cuts=np.arange(3,len(indices)-2)
                    cuts=cuts[values[cuts-1]<values[cuts]]
                    if not len(cuts):continue
                    left_w=cw[cuts-1];right_w=cw[-1]-left_w;left_y=cy[cuts-1];right_y=cy[-1]-left_y
                    scores=np.sum(cyy[-1]-left_y**2/left_w[:,None]-right_y**2/right_w[:,None],axis=1)
                    j=int(np.argmin(scores));score=float(scores[j]);cut=int(cuts[j])
                    if score<parent-1e-12 and (best is None or score<best[0]-1e-12):
                        threshold=float((values[cut-1]+values[cut])/2)
                        best=(score,feature,threshold,ordered[:cut],ordered[cut:])
                if best is not None:
                    _,feature,threshold,left,right=best
                    record.update(feature=feature,threshold=threshold,left=node(left,depth+1),right=node(right,depth+1))
                return record
            self.tree=node(np.arange(len(x)),0)
        else:raise ValueError(kind)
        return self

    def predict(self,x):
        x=np.asarray(x,float);kind=self.config['kind']
        if kind=='ridge_margin':return self.ridge.predict(x)
        z=self.transform.apply(x)
        if kind=='nearest_type':
            distances=np.sqrt(np.mean((z[:,None,:]-self.centers[None])**2,axis=-1))
            k=min(self.config['k'],len(self.centers));idx=np.argsort(distances,axis=1,kind='stable')[:,:k]
            dd=np.take_along_axis(distances,idx,axis=1)
            w=np.ones_like(dd) if self.config['weight']=='equal' else 1/np.maximum(dd,1e-6)
            return np.sum(self.values[idx]*w[...,None],axis=1)/w.sum(1)[:,None]
        result=[]
        for row in z:
            node=self.tree
            while 'feature' in node:node=node['left'] if row[node['feature']]<=node['threshold'] else node['right']
            result.append(node['value'])
        return np.asarray(result,float)

    def choose(self,x):
        x=np.asarray(x,float);prediction=self.predict(x);choice=best_arm(prediction)
        advantage=prediction[np.arange(len(x)),choice]
        return np.where(np.isfinite(x).any(1)&(advantage>self.config['margin']+1e-12),choice,BASELINE)

    def to_dict(self):
        d=dict(model='source_only_candidate',config=self.config,pca_dimensions=self.pca_dimensions,source_types=self.source_types,
            fallback='All features unavailable or predicted advantage <= margin -> S500',arms=list(ARMS))
        if self.config['kind']=='ridge_margin':d['ridge']=self.ridge.to_dict()
        else:
            d['transform']=self.transform.to_dict()
            if self.config['kind']=='nearest_type':d.update(centers=self.centers.tolist(),values=self.values.tolist())
            else:d['tree']=self.tree
        return d

    @classmethod
    def from_dict(cls,d):
        obj=cls(d['config'],d['pca_dimensions']);obj.source_types=d['source_types']
        if obj.config['kind']=='ridge_margin':obj.ridge=Regression.from_dict(d['ridge'])
        else:
            obj.transform=Transform.from_dict(d['transform'])
            if obj.config['kind']=='nearest_type':obj.centers=np.asarray(d['centers']);obj.values=np.asarray(d['values'])
            else:obj.tree=d['tree']
        return obj


def grid(strategy):
    if strategy=='ridge_margin':
        return [dict(kind=strategy,penalty=p,margin=m) for p in (100.,10.,1.,.1,.01) for m in (.05,.02,.01,.005,0.)]
    if strategy=='nearest_type':
        return [dict(kind=strategy,k=k,weight=w,margin=m) for k in (7,3,2,1) for w in ('equal','inverse_distance') for m in (.02,.01,0.)]
    if strategy=='shallow_tree':
        return [dict(kind=strategy,depth=d,margin=m) for d in (1,2,3) for m in (.02,.01,0.)]
    raise ValueError(strategy)


def tune_candidate(x,rates,types,strategy,pca_dimensions=None):
    x,rates,types=np.asarray(x,float),np.asarray(rates,float),np.asarray(types)
    configs=grid(strategy);history=[]
    if len(set(types))<2:return configs[0],dict(status='single_source_type_fixed_first_config',history=[])
    for config in configs:
        folds=[]
        for held in sorted(set(types)):
            tr,va=types!=held,types==held
            model=Predictor(config,pca_dimensions).fit(x[tr],rates[tr],types[tr])
            prediction=model.predict(x[va]);choice=model.choose(x[va]);actual=rates[va][np.arange(va.sum()),choice]
            folds.append(dict(type=str(held),regret=float(np.mean(rates[va].max(1)-actual)),selected=choice.tolist(),
                squared_error=float(np.mean((prediction-(rates[va]-rates[va,BASELINE,None]))**2))))
        history.append(dict(config=config,folds=folds,mean_regret=float(np.mean([f['regret'] for f in folds])),
            mean_squared_error=float(np.mean([f['squared_error'] for f in folds]))))
    # Grid is ordered by simplicity/conservatism; retain first equal-regret model.
    chosen=min(history,key=lambda r:round(r['mean_regret'],12))
    return chosen['config'],dict(history=history,selection='type-equal inner regret; tie predeclared grid order')


def leave_one_type_candidate(x,rates,types,strategy,pca_dimensions=None):
    x,rates,types=np.asarray(x,float),np.asarray(rates,float),np.asarray(types)
    choices=np.full(len(x),-1,int);folds=[]
    for held in sorted(set(types)):
        tr,va=types!=held,types==held
        config,inner=tune_candidate(x[tr],rates[tr],types[tr],strategy,pca_dimensions)
        model=Predictor(config,pca_dimensions).fit(x[tr],rates[tr],types[tr]);choices[va]=model.choose(x[va])
        folds.append(dict(held_type=str(held),source_types=sorted(set(types[tr].tolist())),source_indices=np.flatnonzero(tr).tolist(),
            held_indices=np.flatnonzero(va).tolist(),config=config,inner=inner,model=model.to_dict()))
    return choices,folds


def fit_selected(x,rates,types,strategy='ridge',pca_dimensions=None):
    if strategy=='ridge':
        from selector import tune
        penalty,inner=tune(x,rates,types,pca_dimensions)
        return Regression(penalty,pca_dimensions).fit(x,rates,types),inner
    config,inner=tune_candidate(x,rates,types,strategy,pca_dimensions)
    return Predictor(config,pca_dimensions).fit(x,rates,types),inner


def restore(record):
    return Regression.from_dict(record) if record['model']=='relative_success_ridge' else Predictor.from_dict(record)
