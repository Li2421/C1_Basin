"""Non-neural local predictability in native physical versus aggregate context.

Controller/eta are held fixed when selecting neighbors. This is a privileged
source diagnostic, not a deployable zero-shot model. K uses inner source VAL;
fresh seeds of outer families are never fitting or selection labels.
"""
import itertools,json
from pathlib import Path
import numpy as np
from ring_exchange.local_frame import local_observation
from ring_exchange.environment import Config
from .replication_analysis import OUT,SRC,load,read,write,csvwrite,correlation,block_b15

KINDS=('controller_eta_constant','native_ordered','native_permutation_invariant','aggregate_context','entity_response_ordered')
KS=(1,3,5,10,100000)

def main():
    states,fresh,keys=load();lookup={s['uid']:i for i,s in enumerate(states)}
    oldstates=read(SRC/'source_states.json');d=np.load(SRC/'source_data.npz');protocol=read(SRC/'protocol.json')
    cfg=Config(**read(SRC.parent/'ring_exchange_stage1/base_u_v10_local_dataset/manifest.json')['scenario_config'])
    permutations=list(itertools.permutations(range(4)))
    native=np.array([[local_observation(*[np.asarray(s['physical'][k])[list(p)] for k in ('positions','velocities','goals')],cfg) for p in permutations] for s in oldstates])
    # Fixed physical units commute with permutations. Thus minimizing over the
    # permutation group is a genuine invariant metric, not a relabeled vector.
    scale=np.full(23,4.);scale[[2,3,8,9,12,13,16,17]]=.82
    native=native/scale
    indices=np.array([[i*16+e for e in (10,15)] for i in range(46)])
    ctx=d['context'][:,indices]
    response=np.stack([np.load(OUT/'agent_response'/f'features_{c}.npz')['agent_response'].reshape(46,2,4,16) for c in ('alt','second')])
    ss=d['success'][:,indices];ff=d['failure'][:,indices]
    prediction={k:np.full(fresh.shape[:3],np.nan) for k in KINDS};selection=[]
    for fold,split in enumerate(protocol['folds']):
        fit=np.array(split['fit']);val=np.array(split['inner']);test=np.array(split['test'])
        for kind in KINDS:
            def dist(c,e,targets):
                if kind=='controller_eta_constant':return np.zeros((len(targets),len(fit)))
                if kind=='native_ordered':
                    a=native[targets,0].reshape(len(targets),-1);b=native[fit,0].reshape(len(fit),-1)
                    return np.mean((a[:,None,:]-b[None,:,:])**2,-1)
                if kind=='native_permutation_invariant':
                    a=native[targets,0].reshape(len(targets),-1);b=native[fit].reshape(len(fit),24,-1)
                    return np.mean((a[:,None,None,:]-b[None,:,:,:])**2,-1).min(-1)
                feature=ctx[c,:,e] if kind=='aggregate_context' else np.concatenate([ctx[c,:,e],response[c,:,e].reshape(46,-1)],axis=-1)
                center=feature[fit].mean(0);st=feature[fit].std(0);active=st>1e-6
                a=((feature-center)/np.maximum(st,.01))[:,active]
                return np.mean((a[targets,None,:]-a[None,fit,:])**2,-1)
            def predict(targets,k):
                pred=np.zeros((2,len(targets),2))
                for c in range(2):
                 for e in range(2):
                    order=np.argsort(dist(c,e,targets),axis=1,kind='stable')[:,:min(k,len(fit))]
                    ii=fit[order];s=ss[c,ii,e].sum(-1);f=ff[c,ii,e].sum(-1)
                    pred[c,:,e]=(s+.5)/(s+f+1)
                return pred
            options=(100000,) if kind=='controller_eta_constant' else KS;best=None
            for k in options:
                p=predict(val,k);s=ss[:,val];f=ff[:,val]
                nll=float(np.sum(-s*np.log(p)-f*np.log1p(-p))/np.sum(s+f))
                if best is None or nll<best[0]:best=(nll,k)
            selection.append(dict(kind=kind,fold=fold,k=best[1],inner_NLL=best[0]));p=predict(test,best[1])
            for j,i in enumerate(test):
                if oldstates[i]['uid'] in lookup:prediction[kind][:,lookup[oldstates[i]['uid']]]=p[:,j]
    s=np.nansum(fresh[...,16:],-1);f=np.nansum(1-fresh[...,16:],-1);q=s/(s+f);result=[]
    for kind,p in prediction.items():
        assert np.isfinite(p).all();choice=p.argmax(-1);selected=[];robust=oracle=0
        for c in range(2):
         for i in range(len(states)):
            selected.append(q[c,i,choice[c,i]])
            for b in range(1,4):
                good,bad,unknown=block_b15(fresh[c,i,:,b*16:(b+1)*16]);robust+=int(good[choice[c,i]]);oracle+=int(good.any())
        delta_corr=correlation((p[:,:,0]-p[:,:,1]).ravel(),(q[:,:,0]-q[:,:,1]).ravel())
        result.append(dict(kind=kind,fresh_NLL=float(np.sum(-s*np.log(p)-f*np.log1p(-p))/np.sum(s+f)),
            fresh_MAE=float(abs(p-q).mean()),selected_Q48=float(np.mean(selected)),B15_blocks=robust,oracle_B15_blocks=oracle,
            state_eta_contrast_Pearson=delta_corr['Pearson'] if delta_corr else None))
    csvwrite(OUT/'local_physical_metrics.csv',result);write(OUT/'local_physical_selection.json',selection)
    np.savez_compressed(OUT/'local_physical_predictions.npz',**prediction)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
