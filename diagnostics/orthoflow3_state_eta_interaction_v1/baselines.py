"""Phase two: isolated analytical audit baselines, never production retraining."""
import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit
from .audit import OUT,PRE,SCENES,Frozen,folders,load,dump,ev,select_metrics,contrasts,corr

def features(e,omega,phase):
    return np.column_stack((np.ones(len(e)),e,e*e,np.sqrt(2/len(phase))*np.cos(e@omega+phase)))

def run():
    assert (OUT/'shuffle_metrics.json').exists(),'Frozen-model phase first'
    result=[];novel=[];cache={}
    for fold in SCENES:
        ps,xx=ev.inputs(fold);eta=np.asarray([p['eta'] for p in ps]);truth=load((ev.FIRST if fold=='ring' else PRE/fold)/'target_truth.json')
        for name,folder in folders(fold).items():
            m=Frozen(folder);h=m.encode(xx);original=m.score(h,eta)
            if str(folder) not in cache:
                rows=pq.read_table(folder/'source_pairs.parquet').to_pylist();rr=[r for r in rows if r['split']=='train'];ss=load(folder/'source_states.json');sx=dict(np.load(folder/'source_entities.npz'))
                si=np.array([r['state_index'] for r in rr]);eta_train=np.asarray([r['eta'] for r in rr]);e=(eta_train-m.center)/m.radius
                allh=m.encode(sx);ht=allh[si];ym=np.array([r['q'] for r in rr]);rng=np.random.default_rng(8301)
                omega=rng.normal(size=(3,128));phase=rng.uniform(0,2*np.pi,128);ef=features(e,omega,phase)
                scenes=sorted({r['scenario'] for r in rr});counts={sc:sum(r['scenario']==sc for r in rr) for sc in scenes}
                w=np.array([1/(len(scenes)*counts[r['scenario']]) for r in rr])
                hm=(w[:,None]*ht).sum(0);hs=np.sqrt((w[:,None]*(ht-hm)**2).sum(0));hs=np.maximum(hs,.01)
                hsfeat=(ht-hm)/hs;designs={'eta_only_RFF':ef,'additive_RFF':np.column_stack((ef,hsfeat))};beta={}
                for k,z in designs.items():
                    reg=np.eye(z.shape[1])*.01;reg[0,0]=1e-8
                    beta[k]=np.linalg.solve(z.T@(w[:,None]*z)+reg,z.T@(w*ym))
                cache[str(folder)]=(omega,phase,hm,hs,beta,rr,ss)
            omega,phase,hm,hs,beta,rr,ss=cache[str(folder)]
            e=(eta.reshape(-1,3)-m.center)/m.radius;ef=features(e,omega,phase);hr=np.repeat((h-hm)/hs,16,axis=0)
            sscores={k:((ef if k=='eta_only_RFF' else np.column_stack((ef,hr)))@b).reshape(-1,16) for k,b in beta.items()}
            sscores['frozen_eta_only_MLP']=Frozen(folder,'eta_only').score(np.zeros((len(h),0)),eta)
            sscores['frozen_full']=original
            trainetas={tuple(r['eta']) for r in rr};trainstates={r['state_uid'] for r in rr}
            novel.append({'fold':fold,'model':name,'train_state_overlap':len(trainstates&{p['state_uid'] for p in ps}),
                'exact_eta_seen_train':sum(tuple(e) in trainetas for e in eta.reshape(-1,3)),
                'total_eta':eta.size//3,'source_dataset':str(folder/'source_pairs.parquet')})
            for kind,z in sscores.items():
                # Preserve ordering; clipping may manufacture top-score ties.
                metric=select_metrics(z,truth,original)
                metric['prediction_outside_probability_range']=float(np.mean((z<0)|(z>1)))
                true=np.array([[np.nan if q is None else q for q in t['q']] for t in truth]);ok=np.isfinite(true)
                metric.update(Q_MSE=float(np.mean((np.clip(z[ok],0,1)-true[ok])**2)),Q_MAE=float(np.mean(abs(np.clip(z[ok],0,1)-true[ok]))))
                result.append({'fold':fold,'model':name,'baseline':kind,**metric})
            np.savez_compressed(OUT/f'baseline_{fold}_{name}.npz',**sscores,omega=omega,phase=phase,hmean=hm,hstd=hs,**{'beta_'+k:v for k,v in beta.items()})
    dump('baseline_metrics.json',result);dump('unseen_state_eta_audit.json',novel)
    dump('baseline_fit_protocol.json',{'production_retraining':False,'generator_modified':False,
       'new_fits':'TRAIN-only closed-form ridge diagnostic baselines, frozen before evaluating target labels',
       'objective':'scene-balanced pair-equal weighted squared Q error; fixed ridge .01; no hyperparameter selection',
       'eta_features':'constant + linear + squared +128 fixed random Fourier features, seed8301',
       'state_features':'TRAIN-standardized frozen full-critic embedding; no newly trained representation',
       'additive':'A(h)+B(eta) in probability space before clipping; clipping only for probability metrics, never interaction measurement',
       'limitation':'not a same-parameter-count new neural-network trial; frozen eta-only MLP is also reported; additive global ranking cannot reverse regardless of A flexibility'})

if __name__=='__main__':run()
