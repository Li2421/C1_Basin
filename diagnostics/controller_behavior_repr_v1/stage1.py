"""I1--I4: consumes TRAIN-only labels and unlabeled representations."""
from core import *
from data import Representations,logistic
from scipy.stats import spearmanr,rankdata
import warnings
REPS=['R3','R3-PCA2','R4_action','R5','R5-PCA2','R5-Dprog','R3-PCA2+R5-PCA2']
STATS=['progress','projection_fraction','minimum_pair_distance','mean_speed','stalled_steps']

def rho(x,y):
    if np.ptp(x)==0 or np.ptp(y)==0:return float('nan')
    return float(spearmanr(x,y).statistic)
def cluster_ci(x,y,clusters,rng):
    ids=[np.where(np.array(clusters)==c)[0] for c in dict.fromkeys(clusters)];boot=[]
    for _ in range(5000):
        k=np.concatenate([ids[i] for i in rng.integers(len(ids),size=len(ids))]);boot.append(rho(x[k],y[k]))
    finite=np.array(boot)[np.isfinite(boot)]
    return dict(rho=rho(x,y),ci95=np.quantile(finite,[.025,.975]) if len(finite) else [None,None],bootstrap_valid=len(finite),bootstrap_degenerate=5000-len(finite),clusters=len(ids))

def main():
    gate();r=Representations();names=r.names;real=names[:14];blend=names[14:];old=read(R5/'STAGE_0_RESULTS.json')['rows'];labs=np.load(HERE/'TRAIN_ONLY.npz')['outcomes'];curves=np.array([next(x['observed_valid_curve'] for x in old if x['controller']==n) if i<14 else curve(labs[i,:96]) for i,n in enumerate(names)])
    ds=curves[:,13]-curves[:,10];regime=np.array([next(x['best_eta'] for x in old if x['controller']==n) if i<14 else int(REGIMES[np.argmax(curves[i,REGIMES])]) for i,n in enumerate(names)])
    best=curves.argmax(1);truth=[dict(controller=n,regime=regime[i],best_K16=best[i],delta_success=ds[i],curve=curves[i],source='Stage0 observed_valid_curve' if i<14 else '96 TRAIN state masked valid curve') for i,n in enumerate(names)]
    write(HERE/'TRAIN_TRUTH.json',truth);all_results={}
    for sens in (False,True):
        # I1 imputation uses only the controllers in that descriptive population;
        # fold-local I2/I3 imputation is recomputed separately below.
        I1=[];rng=np.random.default_rng(20261013071)
        for pop,ids,clusters in [('real',np.arange(14),[f'real{i}' for i in range(14)]),('blend',np.arange(14,26),[f'pair{i//3}' for i in range(12)]),('combined',np.arange(26),[f'real{i}' for i in range(14)]+[f'pair{i//3}' for i in range(12)])]:
            _,delta,_=r.behavior([names[i] for i in ids],sens)
            for j,stat in enumerate(STATS):I1.append(dict(population=pop,statistic=stat,primary=stat=='progress',**cluster_ci(delta[ids,j],ds[ids],clusters,rng)))
        I2=[];I3=[]
        for rep in REPS:
            for bed in ('real','blend'):
                predictions=np.full(14 if bed=='real' else 12,np.nan);folds=[]
                for fold in range(14 if bed=='real' else 4):
                    held=[real[fold]] if bed=='real' else blend[fold*3:(fold+1)*3];sources=[n for n in (real if bed=='real' else names) if n not in held]
                    z,meta=r.transform(rep,sources,targets=sources+held,sensitivity=sens);x=z[:len(sources)];h=z[len(sources):];y=ds[[names.index(n) for n in sources]]
                    mu=x.mean(0);ym=y.mean();xc=x-mu;hc=h-mu
                    # Dual ridge exactly equals primal alpha=1; avoids 256x256 solves.
                    w=xc.T@np.linalg.solve(xc@xc.T+np.eye(len(x)),y-ym);pred=ym+hc@w
                    loc=[(real if bed=='real' else blend).index(n) for n in held];predictions[loc]=pred
                    folds.append(dict(fold=fold,sources=sources,held=held,predictions=pred,preprocessing=meta))
                actual=ds[:14] if bed=='real' else ds[14:];I2.append(dict(representation=rep,bed=bed,spearman=rho(predictions,actual),sign_accuracy=float(np.mean(np.sign(predictions)==np.sign(actual))),predictions=predictions,truth=actual,folds=folds))
            nn=[];lg=[];major=[];details=[]
            for fold in range(14):
                sources=[n for n in real if n!=real[fold]];z,meta=r.transform(rep,sources,targets=sources+[real[fold]],sensitivity=sens);x=z[:-1];h=z[-1:];y=regime[[names.index(n) for n in sources]];near=int(np.linalg.norm(x-h,axis=1).argmin());nn.append(int(y[near]));p,model=logistic(x,y,h);lg.append(int(model['classes'][p.argmax(1)[0]]));major.append(int(np.bincount(y,minlength=16).argmax()));details.append(dict(held=real[fold],nearest=sources[near],probabilities=p[0],model=model))
            for clf,pred in [('1NN',nn),('logistic_C1',lg),('source_majority',major)]:
                classes=[2,4,10,13];cm=np.zeros((4,4),int)
                for a,b in zip(regime[:14],pred):cm[classes.index(a),classes.index(b)]+=1
                acc=np.mean(np.array(pred)==regime[:14]);I3.append(dict(representation=rep,classifier=clf,accuracy=acc,delta_vs_fixed_majority=acc-9/14,classes=classes,confusion=cm,predictions=pred,truth=regime[:14],fold_details=details if clf=='logistic_C1' else None))
        panel=read(R4/'protocol.json')['controller_order'];I4={};_,delta,_=r.behavior(real,sens)
        for rep in ('R3','R3-PCA2','R5-PCA2','R5-Dprog'):
            x,meta=r.transform(rep,real,targets=panel,sensitivity=sens);dist=np.linalg.norm(x[:,None]-x[None,:],axis=-1);nearest={n:[panel[j] for j in np.argsort(dist[i]) if j!=i] for i,n in enumerate(panel)}
            a=panel.index('base_u_macflow');b=panel.index('base_u_v10_local_macflow');c=panel.index('base_u_v9_local_macflow');prediction=dist[b,c]<dist[b,a] if rep.startswith('R3') else dist[a,b]<dist[a,c] and dist[a,b]<dist[b,c]
            I4[rep]=dict(controllers=panel,distances=dist,nearest_ranking=nearest,prediction_satisfied=bool(prediction),triplet_distances=dict(base_v10=dist[a,b],base_v9=dist[a,c],v10_v9=dist[b,c]),preprocessing=meta)
        I4['panel_delta']=[dict(controller=n,delta_progress=delta[names.index(n),0],sign=int(np.sign(delta[names.index(n),0]))) for n in panel]
        I4['H_static_recipe']='SUPPORTED' if all(I4[x]['prediction_satisfied'] for x in ('R3','R3-PCA2')) else 'REJECTED'
        I4['H_behavior_regime']='SUPPORTED' if all(I4[x]['prediction_satisfied'] for x in ('R5-PCA2','R5-Dprog')) else 'REJECTED'
        all_results['imputed' if sens else 'main']=dict(I1=I1,I2=I2,I3=I3,I4=I4)
        write(HERE/'STAGE1.json',dict(status='RUNNING',utc=utc(),evidence_level='DEVELOPMENT',results=all_results,endpoint_seeds_read=False))
    write(HERE/'STAGE1.json',dict(status='COMPLETE',utc=utc(),evidence_level='DEVELOPMENT',results=all_results,endpoint_seeds_read=False,limitation='Masked eta13/eta10 means use different state subsets; delta_success includes state composition differences.',bootstrap_seed=20261013071))
    print(json.dumps(clean(all_results['main']['I1'][:5])),flush=True)
if __name__=='__main__':main()
