"""Preserve within-type variation and avoid bad-arm losses dominating splits."""
import numpy as np
from selector import ARMS,BASELINE
from selector_candidates import Transform,Predictor as V1,fit_selected as fit_v1,restore as restore_v1


class Predictor(V1):
    def fit(self,x,rates,types):
        x,rates,types=np.asarray(x,float),np.asarray(rates,float),np.asarray(types)
        self.source_types=sorted(set(types.tolist()))
        if self.config['kind']=='armwise_tree':
            self.models={}
            for arm in range(len(ARMS)):
                if arm==BASELINE:continue
                target=np.zeros_like(rates);target[:,arm]=rates[:,arm]-rates[:,BASELINE]
                config=dict(kind='shallow_tree',depth=self.config['depth'],margin=0.)
                self.models[arm]=V1(config,self.pca_dimensions).fit(x,target,types)
        elif self.config['kind']=='type_kernel':
            self.transform=Transform().fit(x,types,self.pca_dimensions)
            self.x=self.transform.apply(x);self.types=types.copy();self.target=rates-rates[:,BASELINE,None]
        else:raise ValueError(self.config)
        return self

    def predict(self,x):
        x=np.asarray(x,float)
        if self.config['kind']=='armwise_tree':
            values=np.zeros((len(x),len(ARMS)))
            for arm,model in self.models.items():values[:,arm]=model.predict(x)[:,arm]
            return values
        z=self.transform.apply(x);distance=np.mean((z[:,None,:]-self.x[None])**2,axis=-1)
        values=[];type_distance=[]
        for t in self.source_types:
            mask=self.types==t;dd=distance[:,mask];minimum=dd.min(1)
            kernel=np.exp(-(dd-minimum[:,None])/(2*self.config['bandwidth']**2))
            local=kernel@self.target[mask]/kernel.sum(1)[:,None]
            shrink=self.config['shrinkage'];values.append((1-shrink)*local+shrink*self.target[mask].mean(0))
            type_distance.append(minimum)
        values=np.stack(values,axis=1);type_distance=np.stack(type_distance,axis=1)
        idx=np.argsort(type_distance,axis=1,kind='stable')[:,:min(self.config['k'],len(self.source_types))]
        return values[np.arange(len(x))[:,None],idx].mean(1)

    def to_dict(self):
        out=dict(model='source_only_candidate_v2',config=self.config,pca_dimensions=self.pca_dimensions,
            source_types=self.source_types,arms=list(ARMS),fallback='All features unavailable or predicted advantage <= margin -> S500')
        if self.config['kind']=='armwise_tree':out['models']={str(a):m.to_dict() for a,m in self.models.items()}
        else:out.update(transform=self.transform.to_dict(),x=self.x.tolist(),types=self.types.tolist(),target=self.target.tolist())
        return out

    @classmethod
    def from_dict(cls,record):
        obj=cls(record['config'],record['pca_dimensions']);obj.source_types=record['source_types']
        if obj.config['kind']=='armwise_tree':obj.models={int(a):V1.from_dict(m) for a,m in record['models'].items()}
        else:
            obj.transform=Transform.from_dict(record['transform']);obj.x=np.asarray(record['x']);obj.types=np.asarray(record['types']);obj.target=np.asarray(record['target'])
        return obj


def grid(strategy):
    if strategy=='armwise_tree':return [dict(kind=strategy,depth=d,margin=m) for d in (1,2,3) for m in (.02,.01,0.)]
    if strategy=='type_kernel':
        return [dict(kind=strategy,k=k,bandwidth=h,shrinkage=s,margin=m)
            for k in (7,3,1) for h in (2.,1.,.5,.25) for s in (.5,0.) for m in (.01,0.)]
    raise ValueError(strategy)


def tune_candidate(x,rates,types,strategy,pca_dimensions=None):
    x,rates,types=np.asarray(x,float),np.asarray(rates,float),np.asarray(types)
    configs=grid(strategy);history=[]
    if len(set(types))<2:return configs[0],dict(status='single_source_type_fixed_first_config',history=[])
    for config in configs:
        folds=[]
        for held in sorted(set(types)):
            tr,va=types!=held,types==held
            model=Predictor(config,pca_dimensions).fit(x[tr],rates[tr],types[tr]);prediction=model.predict(x[va]);choice=model.choose(x[va])
            actual=rates[va][np.arange(va.sum()),choice]
            folds.append(dict(type=str(held),regret=float(np.mean(rates[va].max(1)-actual)),selected=choice.tolist(),
                squared_error=float(np.mean((prediction-(rates[va]-rates[va,BASELINE,None]))**2))))
        history.append(dict(config=config,folds=folds,mean_regret=float(np.mean([f['regret'] for f in folds])),
            mean_squared_error=float(np.mean([f['squared_error'] for f in folds]))))
    chosen=min(history,key=lambda r:round(r['mean_regret'],12))
    return chosen['config'],dict(history=history,selection='Type-equal inner regret; tie predeclared grid order')


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
    if strategy not in ('armwise_tree','type_kernel'):return fit_v1(x,rates,types,strategy,pca_dimensions)
    config,inner=tune_candidate(x,rates,types,strategy,pca_dimensions)
    return Predictor(config,pca_dimensions).fit(x,rates,types),inner


def restore(record):
    return Predictor.from_dict(record) if record['model']=='source_only_candidate_v2' else restore_v1(record)
