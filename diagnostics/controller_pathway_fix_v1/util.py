"""Read-only historical adapters; all new artifacts remain in this directory."""
import os,sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
R4=ROOT/'diagnostics/controller_loco_real_flow_v1';R5=ROOT/'diagnostics/controller_flow_blend_v1'
PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(R5))
from diagnostics.controller_flow_blend_v1.train import Features as OldFeatures,training_data as old_training_data
from diagnostics.controller_loco_real_flow_v1.train import Net,build as old4_build
from diagnostics.controller_q_transfer_round3_v1.state_generator import physical23
from diagnostics.controller_flow_blend_v1.stage0 import feature_path
from scipy.special import expit
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def task_name(t):return f"{t['bed']}_{t['coverage']}_fold{t['fold']}_{t['arm']}"
REGIMES=np.array([10,13,2,4])
NEW_ARMS=['F1_PCA1','F1_PCA2','F1_PCA3','F1_oracle_low','F2_PCA2','F2_oracle_low']
class Data:
    def __init__(self,t):
        self.t=t;self.bed=t['bed'];self.cache={};self.sources=t['sources'];self.held=t['held']
        self.p4=read(R4/'protocol.json');self.p5=read(R5/'stage_2/protocol.json')
        self.original=list(self.p4['controller_order']);self.extra=read(R4/'stage_b/protocol.json')['controller_order'];self.blends=self.p5['controller_order']
        a=np.load(R4/'LABELS.npz')['outcomes'];b=np.load(R4/'stage_b/LABELS.npz')['outcomes']
        self.lab=dict(zip(self.original,a[:,:128,:,:4]));self.lab.update(zip(self.extra,b))
        self.lab.update(zip(self.blends,np.load(R5/'stage_2/TRAIN_LABELS.npz')['outcomes']))
        self.common_states=self.p5['states'][:96] if self.bed=='blend' else self.p4['states'][:128]
        self.test_states=self.p5['states'][96:] if self.bed=='blend' else self.p4['states'][128:]
        self.val_states=read(ROOT/'diagnostics/controller_q_transfer_round3_v1/P1_EXECUTION_PROTOCOL.json')['states']
        self.valnames=[n for n in self.sources if n in self.original+self.extra]
        vf=np.load(ROOT/'diagnostics/controller_q_transfer_round3_v1/P1_LABELS.npz');names=list(vf['controller_names'])
        self.SV=np.array([vf['Y'][names.index(n)].sum(-1) for n in self.valnames],np.float32)
        self.FV=np.array([(4-vf['Y'][names.index(n)]-vf['U'][names.index(n)]).sum(-1) for n in self.valnames],np.float32)
        self.curves={n:self.curve(self.lab[n]) for n in self.sources}
        self.src=int(np.mean(list(self.curves.values()),axis=0).argmax())
        self.regimes={n:int(REGIMES[np.argmax(self.curves[n][REGIMES])]) for n in self.sources}
        self.means={n:self.static(n,self.common_states,'R3').mean(0).astype(float) for n in self.sources+self.held}
        centers=np.array([self.means[n] for n in self.sources]);self.pca_mean=centers.mean(0);sd=centers.std(0);self.pca_scale=np.where(sd>1e-6,sd,1.)
        z=(centers-self.pca_mean)/self.pca_scale;_,sing,vt=np.linalg.svd(z,full_matrices=False)
        for v in vt:
            if v[np.argmax(np.abs(v))]<0:v*=-1
        self.components=vt[:3];self.pca_variance=(sing**2/(sing**2).sum()).tolist()
    @staticmethod
    def curve(l):
        S=(l==1).sum(-1);V=((l==1)|(l==0)).sum(-1)
        return np.nanmean(np.divide(S,V,out=np.full_like(S,np.nan,dtype=float),where=V>0),axis=0)
    def states(self,n):return self.p4['states'][:128] if n in self.original+self.extra else self.p5['states'][:96]
    def static(self,n,states,key):
        if self.bed=='blend':
            if 'old_features' not in self.cache:self.cache['old_features']=OldFeatures(self.p5,'R3')
            return self.cache['old_features'].static(n,states,key)
        if n not in self.cache:
            z=np.load(R4/'static_features'/f'{n}.npz');v=np.load(R4/'static_features'/f'{n}_validation.npz')
            mapping={str(s):(z,i) for i,s in enumerate(z['state_uids'])};mapping.update({str(s):(v,i) for i,s in enumerate(v['state_uids'])});self.cache[n]=mapping
        return np.array([self.cache[n][s['state_uid']][0][key][self.cache[n][s['state_uid']][1]] for s in states])
    def coordinates(self,n,arm):
        if 'PCA' in arm:
            k=int(arm[-1]);return ((self.means[n]-self.pca_mean)/self.pca_scale)@self.components[:k].T
        if 'oracle_low' in arm:
            assert self.bed=='blend','Target regime unknown for held-out real Flow'
            b=self.p5['controllers']['ring_exchange'][n]['payload'].get('blend')
            A,B,lam=(b['A'],b['B'],b['lambda']) if b else (n,n,0.)
            assert A in self.sources and B in self.sources
            return np.r_[lam,REGIMES==self.regimes[A],REGIMES==self.regimes[B]].astype(float)
        raise ValueError(arm)
    def inputs(self,n,states,arm):
        co=self.coordinates(n,arm);v=np.zeros((len(states),513),np.float32);v[:,:len(co)]=co
        return np.c_[physical23(states),v].astype(np.float32)
    def build(self,arm):
        raw=[self.inputs(n,self.states(n),arm) for n in self.sources]
        mean=np.concatenate(raw).mean(0);sd=np.concatenate(raw).std(0);scale=np.where(sd>1e-6,sd,1.)
        X=np.zeros((len(raw),128,536),np.float32);L=np.full((len(raw),128,16,4),-2,np.int8)
        for i,(n,x) in enumerate(zip(self.sources,raw)):X[i,:len(x)]=(x-mean)/scale;L[i,:len(x)]=self.lab[n]
        V=np.array([(self.inputs(n,self.val_states,arm)-mean)/scale for n in self.valnames])
        H=np.array([(self.inputs(n,self.test_states,arm)-mean)/scale for n in self.held])
        return X,(L==1).sum(-1).astype(np.float32),(L==0).sum(-1).astype(np.float32),V,H,mean,scale,L
    def base_folder(self,arm='R0'):
        if self.bed=='blend':return R5/'stage_2/fits'/f"LOPO_fold{self.t['fold']}_{self.t['coverage']}_{arm}"
        return R4/'fits'/f"fold{self.t['fold']}_{arm}"
def task_list(bed):
    if bed=='blend':
        p=read(R5/'stage_2/protocol.json');return [dict(bed='blend',coverage=t['coverage'],fold=t['fold'],sources=t['sources'],held=t['held']) for t in read(R5/'stage_2/FIT_TASKS.json') if t['split']=='LOPO' and t['arm']=='R0']
    p=read(R4/'protocol.json');names=p['controller_order'];return [dict(bed='real',coverage='LOCO',fold=i,sources=[n for j,n in enumerate(names) if j!=i],held=[names[i]]) for i in range(7)]
def frozen_predict(folder,inputs):
    import jax;import jax.numpy as jnp
    from flax import serialization
    jax.config.update('jax_enable_x64',False)
    meta=read(folder/'complete.json');norm=np.load(folder/'normalization.npz');width=len(norm['mean']);net=Net();template=net.init(jax.random.PRNGKey(0),np.zeros((1,width),np.float32))
    fn=jax.jit(lambda pa,x:net.apply(pa,x));out=[]
    paths=meta['checkpoints']
    for path in paths:
        pa=serialization.from_bytes(template,Path(path).read_bytes());out.append(np.asarray(fn(pa,jnp.asarray(inputs,np.float32))))
    return np.array(out)
