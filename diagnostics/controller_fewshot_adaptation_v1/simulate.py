"""Without-replacement adaptation: exclusively reads target collection pools."""
from core import *
from scipy.special import expit,betaln,logsumexp
from functools import lru_cache
import time

@lru_cache(maxsize=200000)
def beta_greater(a,b,c,d):
    """Exact P(Beta(a,b)>Beta(c,d)) for positive integer parameters."""
    i=np.arange(int(a),dtype=float)
    logs=betaln(c+i,b+d)-np.log(b+i)-betaln(1+i,b)-betaln(c,d)
    return float(np.clip(np.exp(logsumexp(logs)),0,1))

def winner(values,prior):
    best=np.max(values);tie=np.flatnonzero(values==best)
    return prior if prior in tie else int(tie[0])

def main(target):
    gate();start=time.time();protocol=read(HERE/'PROTOCOL.json');names=protocol['controller_order'];main=target<7;methods=METHODS if main else METHODS[:5];ns=64 if main else 24;prior=10
    out=HERE/'simulations'/f'{target}.npz';out.parent.mkdir(exist_ok=True)
    if out.exists() and out.with_suffix('.json').exists():return
    z=np.load(HERE/'pools'/f'{target}.npz');cells=z['cells'];labels=z['outcomes'];eta=cells[:,1];train=np.load(HERE/'SOURCE_TRAIN_ONLY.npz')['outcomes'];sources=[i for i in range(7 if main else 14) if i!=target]
    source=train[sources];sa=1+(source==1).sum((1,3));sb=1+(source==0).sum((1,3));nc=len(sources)
    # True TRAIN-only fixed prior calculation, distinct from the target outcome pool.
    succ=(source==1).sum(-1);valid=((source==1)|(source==0)).sum(-1);q=np.divide(succ,valid,out=np.full_like(succ,np.nan,dtype=float),where=valid>0);prior=int(np.nanmean(q,axis=1).mean(0).argmax());assert not main or prior==10
    banks=[np.flatnonzero(eta==k) for k in range(16)];restricted=np.flatnonzero(np.isin(eta,[2,4,10,13]));assert len(restricted)>=128 and all(len(k)>=128 for k in banks)
    if main:
        assert read(HERE/'PREDICTION_RECEIPT.json')['status']=='COMPLETE'
        p=np.load(HERE/'predictions'/f'{target}.npz');poolq=expit(np.stack([p['pool_R0'],p['pool_F2']]));heldq=expit(np.stack([p['held_R0'],p['held_F2']]))
    choices=np.zeros((3,7,len(methods),200,3,ns),np.int8);paths=np.full((3,200,128),-1,np.int16);sampled_y=np.full((3,200,128),-2,np.int8);S=np.zeros((3,200,7,16),np.int16);F=S.copy();W=np.zeros((3,200,7,nc),np.float64);prob=np.ones((3,200,7,2));offsets=np.zeros((3,200,7,2,3,16),np.float32) if main else None;unable=np.zeros((3,200,7),np.int16)
    for st,strategy in enumerate(STRATEGIES):
        for repeat in range(200):
            rng=np.random.default_rng(202610130800+target*10000+st*1000+repeat);s=np.zeros(16,int);f=s.copy();logw=np.full(nc,-np.log(nc));residual_sum=np.zeros((2,3,16),float) if main else None
            path=rng.permutation(len(labels))[:128] if strategy=='U' else rng.permutation(restricted)[:128] if strategy=='R' else None
            remaining=[list(rng.permutation(k)) for k in banks] if strategy=='A' else None
            for step in range(129):
                if step in BUDGETS:
                    bi=int(np.flatnonzero(BUDGETS==step)[0]);w=np.exp(logw-logsumexp(logw));hm=(w[:,None]*(sa+s)/(sa+sb+s+f)).sum(0);valid_target=s+f;em=(1+s)/(2+s+f);em[valid_target==0]=-np.inf
                    ec=winner(em,prior) if valid_target.sum() else prior;hc=winner(hm,prior)
                    ep=1. if ec==prior else beta_greater(1+s[ec],1+f[ec],1+s[prior],1+f[prior]);hp=1. if hc==prior else sum(w[j]*beta_greater(sa[j,hc]+s[hc],sb[j,hc]+f[hc],sa[j,prior]+s[prior],sb[j,prior]+f[prior]) for j in range(nc))
                    choices[st,bi,0,repeat]=prior;choices[st,bi,1,repeat]=ec;choices[st,bi,2,repeat]=ec if ep>=.9 else prior;choices[st,bi,3,repeat]=hc;choices[st,bi,4,repeat]=hc if hp>=.9 else prior;prob[st,repeat,bi]=[ep,hp]
                    S[st,repeat,bi]=s;F[st,repeat,bi]=f;W[st,repeat,bi]=w
                    if main:
                        offset=residual_sum/(valid_target+8)[None,None,:];offsets[st,repeat,bi]=offset;cal=np.clip(heldq+offset[:,:,None,:],0,1)
                        for backbone in range(2):
                            q=cal[backbone];a=q.argmax(-1);mu=q.mean(0);candidate=mu.argmax(-1);agreement=(a==candidate[None]).all(0);gain=picked(mu,candidate)-mu[:,prior];safe=np.where(agreement&(gain>=.10),candidate,prior)
                            ctl=q.mean(1);c=ctl.argmax(-1);m=ctl.mean(0);cand=int(m.argmax());ctlsafe=cand if (c==cand).all() and m[cand]-m[prior]>=.10 else prior
                            mi=5+4*backbone;choices[st,bi,mi,repeat]=a;choices[st,bi,mi+1,repeat]=safe[None];choices[st,bi,mi+2,repeat]=c[:,None];choices[st,bi,mi+3,repeat]=ctlsafe
                if step==128:break
                if strategy=='A':
                    available=np.array([len(x)>0 for x in remaining]);w=np.exp(logw-logsumexp(logw));j=int(rng.choice(nc,p=w));draw=rng.beta(sa[j]+s,sb[j]+f);draw[~available]=-np.inf
                    if not available.any():
                        for bi,budget in enumerate(BUDGETS):
                            if budget>step:unable[st,repeat,bi]=budget-step
                        break
                    k=int(draw.argmax());idx=remaining[k].pop()
                else:idx=int(path[step]);k=int(eta[idx])
                y=int(labels[idx]);paths[st,repeat,step]=idx;sampled_y[st,repeat,step]=y
                if y>=0:
                    predictive=(sa[:,k]+s[k])/(sa[:,k]+sb[:,k]+s[k]+f[k]);logw+=np.log(predictive if y==1 else 1-predictive);logw-=logsumexp(logw)
                    if main:residual_sum[:,:,k]+=y-poolq[:,:,int(cells[idx,0]),k]
                    if y==1:s[k]+=1
                    else:f[k]+=1
            assert len(np.unique(paths[st,repeat]))==128,'Collection must be without replacement'
        print(json.dumps(dict(target=target,strategy=strategy,seconds=time.time()-start)),flush=True)
    arrays=dict(choices=choices,paths=paths,sampled_outcomes=sampled_y,success_counts=S,failure_counts=F,H_component_weights=W,safety_posterior_probabilities=prob,unable_to_collect=unable,source_alpha=sa,source_beta=sb,source_target_indices=sources,methods=methods,budgets=BUDGETS,strategies=STRATEGIES,source_prior_eta=prior)
    if main:arrays['offsets']=offsets
    np.savez_compressed(out,**arrays)
    write(out.with_suffix('.json'),dict(utc=utc(),status='COMPLETE',target=target,controller=names[target],bed='main' if main else 'stageB_descriptive',source_names=[names[i] for i in sources],source_prior_eta=prior,repeats=200,choices_shape=list(choices.shape),sampled_unknown=int((sampled_y==-1).sum()),unable_count=int(unable.sum()),simulation_sha256=sha(out),new_rollouts=0,shared_db_writes=0,endpoint_labels_read=False,elapsed_seconds=time.time()-start,beta_probability_cache=beta_greater.cache_info()._asdict()))
if __name__=='__main__':main(int(sys.argv[1]))
