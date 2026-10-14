"""TRAIN-only data and fold-local representations; no endpoint outcome access."""
from core import *
from scipy.optimize import minimize
from scipy.special import softmax,logsumexp

def prepare():
    gate();p4=read(R4/'protocol.json');p5=read(R5/'stage_2/protocol.json');names=p5['anchor_order']+p5['controller_order'];means={k:[] for k in ('R3_96','R3_real','R4_action')}
    for n in names:
        paths=[R5/'stage_2/static_features'/f'{n}.npz']
        if n in p5['anchor_order']:
            old=R4/'static_features'/f'{n}.npz';paths.append(old if old.exists() else R4/'stage_b/static_features'/f'{n}.npz')
        mp={}
        for path in paths:
            z=np.load(path)
            for i,s in enumerate(z['state_uids']):mp[str(s)]={k:z[k][i] for k in ('R3','R4_action')}
        for key,states,col in [('R3_96',p5['states'][:96],'R3'),('R3_real',p4['states'][:128] if n in p5['anchor_order'] else p5['states'][:96],'R3'),('R4_action',p5['states'][:96],'R4_action')]:
            means[key].append(np.array([mp[s['state_uid']][col] for s in states],np.float32).mean(0).astype(float))
    r5=np.load(R5/'stage_2/probes/R5_FEATURES.npz')['features'].astype(float)
    np.savez_compressed(HERE/'FEATURES.npz',controller_names=names,R5=r5,**{k:np.array(v) for k,v in means.items()})
    ts=[];cover=[];lab=np.load(HERE/'TRAIN_ONLY.npz')['outcomes'];curves={n:curve(lab[i]) for i,n in enumerate(names)}
    for t in tasks():
        if t['bed']=='real':
            for cov in ('REAL6','REAL6+BLEND'):
                sources=t['sources'].copy();adds=[]
                if cov=='REAL6+BLEND':
                    for n in p5['controller_order']:
                        b=p5['controllers']['ring_exchange'][n]['payload']['blend']
                        if b['A'] in sources and b['B'] in sources:adds.append(n)
                    sources+=adds
                cover.append(dict(fold=t['fold'],held=t['held'][0],coverage=cov,added=[dict(controller=n,regime=int(REGIMES[np.argmax(curves[n][REGIMES])]),best_K16=int(curves[n].argmax())) for n in adds],source_prior=int(np.mean([curves[n] for n in t['sources']],0).argmax()),source_count=len(sources)))
                for rep in ('R3-PCA2','R5-PCA2','R5-Dprog'):
                    ts.append(dict(t,coverage=cov,sources=sources,real_sources=t['sources'],representation=rep,sensitivity=False,reuse=(cov=='REAL6' and rep=='R3-PCA2')))
        else:
            for sens in (False,True):
                for rep in (('R3-PCA2','R5-PCA2','R5-Dprog') if not sens else ('R5-PCA2','R5-Dprog')):
                    ts.append(dict(t,real_sources=[n for n in t['sources'] if n in p5['anchor_order']],representation=rep,sensitivity=sens,reuse=rep=='R3-PCA2'))
    write(HERE/'TASKS.json',ts);write(HERE/'COVERAGE.json',cover)
    fit=[t for t in ts if not t['reuse'] and (t['bed']=='real' or t['coverage']!='SPARSE' or t['fold']==0)]
    assert len(fit)*3==165
    write(HERE/'FIT_TASKS.json',fit);write(HERE/'PREPARATION.json',dict(utc=utc(),unique_fit_tasks=len(fit),independent_fits=3*len(fit),prediction_tasks=len(ts),real_R3_state_mean_count=128,blend_R3_state_mean_count=96))

def name(t):return f"{t['bed']}_{t['coverage']}_fold{t['fold']}_{t['representation']}"+('_imputed' if t['sensitivity'] else '')

class Representations:
    def __init__(self):
        z=np.load(HERE/'FEATURES.npz');self.names=z['controller_names'].tolist();self.arr={k:z[k] for k in z.files if k!='controller_names'}
    def behavior(self,sources,sensitivity=False):
        raw=self.arr['R5'].reshape(26,16,2,5).copy()
        if sensitivity:
            ids=[self.names.index(n) for n in sources if n!='base_u_v3_macflow'];assert len(ids)
            fill=raw[ids,10,1].mean(0);assert np.isfinite(fill).all();raw[9,10,1]=fill
            return raw.reshape(26,160),(raw[:,:,1]-raw[:,:,0]).mean(1),dict(imputation_source_names=[self.names[i] for i in ids],imputed_cell_mean=fill)
        keep=np.ones(160,bool);keep[105:110]=False;delta=(raw[:,:,1]-raw[:,:,0])[:,np.arange(16)!=10].mean(1)
        return raw.reshape(26,160)[:,keep],delta,dict(dropped_columns=list(range(105,110)),delta_dropped_state=10)
    def transform(self,rep,sources,targets=None,sensitivity=False,real_means=False):
        ids=[self.names.index(n) for n in sources];targets=self.names if targets is None else targets
        r5,delta,meta=self.behavior(sources,sensitivity)
        if rep=='R3-PCA2+R5-PCA2':
            a,am=self.transform('R3-PCA2',sources,targets,sensitivity,real_means);b,bm=self.transform('R5-PCA2',sources,targets,sensitivity,real_means)
            return np.c_[a,b],dict(R3=am,R5=bm)
        raw={'R3':self.arr['R3_real' if real_means else 'R3_96'],'R3-PCA2':self.arr['R3_real' if real_means else 'R3_96'],'R4_action':self.arr['R4_action'],'R5':r5,'R5-PCA2':r5,'R5-Dprog':delta[:,:1]}[rep]
        mean=raw[ids].mean(0);sd=raw[ids].std(0);scale=np.where(sd<1e-6,1.,sd);z=(raw-mean)/scale
        meta.update(mean=mean,scale=scale,source_names=sources,representation=rep)
        if rep.endswith('PCA2'):
            _,sing,vt=np.linalg.svd(z[ids],full_matrices=False)
            for v in vt:
                if v[np.argmax(np.abs(v))]<0:v*=-1
            components=vt[:2];z=z@components.T;meta.update(components=components,variance=(sing**2/(sing**2).sum()))
        assert np.isfinite(z).all()
        return np.array([z[self.names.index(n)] for n in targets]),meta

def logistic(x,y,h):
    classes=np.unique(y);n,d=x.shape;mean=x.mean(0);scale=x.std(0);scale=np.where(scale<1e-6,1.,scale)
    X=np.c_[np.ones(n),(x-mean)/scale];H=np.c_[np.ones(len(h)),(h-mean)/scale];Y=(y[:,None]==classes).astype(float)
    def fn(v):
        w=v.reshape(d+1,len(classes));z=X@w;p=softmax(z,axis=1);g=X.T@(p-Y);g[1:]+=w[1:]
        return (logsumexp(z,axis=1)-(Y*z).sum(1)).sum()+.5*(w[1:]**2).sum(),g.ravel()
    res=minimize(fn,np.zeros((d+1)*len(classes)),jac=True,method='L-BFGS-B',options=dict(maxiter=2000,ftol=1e-12,gtol=1e-8));assert res.success,res.message
    w=res.x.reshape(d+1,len(classes));p=softmax(H@w,axis=1)
    return p,dict(classes=classes,weights=w,mean=mean,scale=scale,iterations=res.nit,success=bool(res.success))

class Data:
    def __init__(self,t):
        gate();self.t=t;self.sources=t['sources'];self.held=t['held'];self.rep=t['representation'];self.p4=read(R4/'protocol.json');self.p5=read(R5/'stage_2/protocol.json');self.original=self.p5['anchor_order'];self.test_states=self.p4['states'][128:] if t['bed']=='real' else self.p5['states'][96:]
        self.val_states=read(P1/'P1_EXECUTION_PROTOCOL.json')['states'];self.valnames=t['real_sources'];z=np.load(HERE/'TRAIN_ONLY.npz');self.lab={str(n):z['outcomes'][i,:int(z['lengths'][i])] for i,n in enumerate(z['controller_names'])};self.curves={n:curve(self.lab[n]) for n in self.sources}
        prior_sources=t['real_sources'] if t['bed']=='real' else self.sources;self.src=int(np.mean([self.curves[n] for n in prior_sources],0).argmax());self.regimes={n:int(REGIMES[np.argmax(self.curves[n][REGIMES])]) for n in self.sources}
        v=np.load(P1/'P1_LABELS.npz');vn=v['controller_names'].tolist();self.SV=np.array([v['Y'][vn.index(n)].sum(-1) for n in self.valnames],np.float32);self.FV=np.array([(4-v['Y'][vn.index(n)]-v['U'][vn.index(n)]).sum(-1) for n in self.valnames],np.float32)
        reps=Representations();coords,self.rep_meta=reps.transform(self.rep,self.sources,sensitivity=t['sensitivity'],real_means=t['bed']=='real');self.coords=dict(zip(reps.names,coords))
    def states(self,n):return self.p4['states'][:128] if n in self.original else self.p5['states'][:96]
    def inputs(self,n,states):
        from diagnostics.controller_q_transfer_round3_v1.state_generator import physical23
        co=self.coords[n];v=np.zeros((len(states),513),np.float32);v[:,:len(co)]=co
        return np.c_[physical23(states),v].astype(np.float32)
    def build(self):
        raw=[self.inputs(n,self.states(n)) for n in self.sources];mean=np.concatenate(raw).mean(0);sd=np.concatenate(raw).std(0);scale=np.where(sd>1e-6,sd,1.)
        X=np.zeros((len(raw),128,536),np.float32);L=np.full((len(raw),128,16,4),-2,np.int8)
        for i,(n,x) in enumerate(zip(self.sources,raw)):X[i,:len(x)]=(x-mean)/scale;L[i,:len(x)]=self.lab[n]
        V=np.array([(self.inputs(n,self.val_states)-mean)/scale for n in self.valnames]);H=np.array([(self.inputs(n,self.test_states)-mean)/scale for n in self.held])
        return X,(L==1).sum(-1).astype(np.float32),(L==0).sum(-1).astype(np.float32),V,H,mean,scale,L

if __name__=='__main__':prepare()
