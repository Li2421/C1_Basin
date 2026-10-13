"""Round4 Net/optimizer/protocol; new variable-controller source data adapter."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from common import *
import numpy as np,time
from diagnostics.controller_loco_real_flow_v1.train import Net
from diagnostics.controller_q_transfer_round3_v1.state_generator import physical23
from stage0 import feature_path
from shared_rollout_db.src.rollout_db import uid

class Features:
    def __init__(self,p,arm):
        self.p=p;self.arm=arm;self.cache={}
        if arm=='R5_behavior':
            f=np.load(HERE/'stage_2/probes/R5_FEATURES.npz');self.behavior=dict(zip(f['controller_names'],f['features']))
    def static(self,name,states,key):
        if name not in self.cache:
            new=np.load(HERE/'stage_2/static_features'/f'{name}.npz');mapping={str(s):(new,i) for i,s in enumerate(new['state_uids'])}
            if name in self.p['anchor_order']:
                old=np.load(feature_path(name));mapping.update({str(s):(old,i) for i,s in enumerate(old['state_uids'])})
                val=np.load(ROOT/'diagnostics/controller_q_transfer_round3_v1/p1_features'/f'{name}.npz');mapping.update({str(s):(val,i) for i,s in enumerate(val['state_uids'])})
            self.cache[name]=mapping
        return np.array([self.cache[name][s['state_uid']][0][key][self.cache[name][s['state_uid']][1]] for s in states],np.float32)
    def descriptor(self,name,states):
        a=np.zeros((len(states),513),np.float32)
        if self.arm=='R0':return a
        if self.arm in ('R3','R4_action'):
            f=self.static(name,states,self.arm);a[:,:f.shape[1]]=f
        elif self.arm=='R_oracle':
            b=self.p['controllers']['ring_exchange'][name]['payload'].get('blend')
            A,B,lam=(b['A'],b['B'],b['lambda']) if b else (name,name,0.)
            a[:,0]=lam;a[:,1:257]=self.static(A,states,'R3');a[:,257:513]=self.static(B,states,'R3')
        elif self.arm=='R5_behavior':a[:,:160]=self.behavior[name]
        else:raise ValueError(self.arm)
        assert np.isfinite(a).all(),('incomplete controller input',name,self.arm)
        return a
    def inputs(self,name,states):return np.c_[physical23(states),self.descriptor(name,states)].astype(np.float32)

def training_data(task,p,f):
    original=read(PREVIOUS/'protocol.json');extra=read(PREVIOUS/'stage_b/protocol.json')
    orig=np.load(PREVIOUS/'LABELS.npz')['outcomes'][:,:128,:,:4];add=np.load(PREVIOUS/'stage_b/LABELS.npz')['outcomes'];blend=np.load(HERE/'stage_2/TRAIN_LABELS.npz')['outcomes']
    source_names=task['sources'];lengths=[];tr=[];labs=[]
    for name in source_names:
        if name in original['controller_order']:states=original['states'][:128];lab=orig[original['controller_order'].index(name)]
        elif name in extra['controller_order']:states=extra['states'];lab=add[extra['controller_order'].index(name)]
        else:states=p['states'][:p['train_states']];lab=blend[p['controller_order'].index(name)]
        z=f.inputs(name,states);lengths.append(len(z));tr.append(z);labs.append(lab)
    mean=np.concatenate(tr).mean(0);std=np.concatenate(tr).std(0);scale=np.where(std>1e-6,std,1.)
    X=np.zeros((len(tr),128,536),np.float32);L=np.full((len(tr),128,16,4),-2,np.int8)
    for i,(x,l) in enumerate(zip(tr,labs)):X[i,:len(x)]=(x-mean)/scale;L[i,:len(l)]=l
    val=read(ROOT/'diagnostics/controller_q_transfer_round3_v1/P1_EXECUTION_PROTOCOL.json')['states']
    val_names=[n for n in p['anchor_order'] if n in source_names]
    V=np.array([(f.inputs(n,val)-mean)/scale for n in val_names])
    labels=np.load(ROOT/'diagnostics/controller_q_transfer_round3_v1/P1_LABELS.npz');vn=list(labels['controller_names'])
    SV=np.array([labels['Y'][vn.index(n)].sum(-1) for n in val_names],np.float32)
    FV=np.array([(4-labels['Y'][vn.index(n)]-labels['U'][vn.index(n)]).sum(-1) for n in val_names],np.float32)
    S=(L==1).sum(-1).astype(np.float32);F=(L==0).sum(-1).astype(np.float32)
    signature=uid('sourcefit',dict(arm=task['arm'],names=source_names,lengths=lengths,train_features=__import__('hashlib').sha256(X.tobytes()).hexdigest(),source_labels=__import__('hashlib').sha256(L.tobytes()).hexdigest(),validation_labels=__import__('hashlib').sha256(np.stack((SV,FV)).tobytes()).hexdigest(),protocol='Round4Net AdamW0.0008 12000steps fit17/23/41 source-val-NLL',x64=False))
    return source_names,X,S,F,V,SV,FV,mean,scale,lengths,signature

def main(index):
    import jax
    import jax.numpy as jnp
    import optax
    from flax import serialization
    jax.config.update('jax_enable_x64',False);assert any(d.platform=='gpu' for d in jax.devices())
    d=HERE/'stage_2';p=read(d/'protocol.json');task=read(d/'FIT_TASKS.json')[index];arm=task['arm']
    assert read(d/'LABEL_AUDIT.json')['status']=='COMPLETE EXACT UID LABEL TENSOR'
    name=f"{task['split']}_fold{task['fold']}_{task['coverage']}_{arm}";out=d/'fits'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    if arm=='R5_behavior' and read(d/'probes/POSTFLIGHT.json')['unknown_probe_uids']:
        write(out/'complete.json',dict(status='NOT EXECUTED — INCOMPLETE R5 INPUT',task=task,unknown_probe_uids=read(d/'probes/POSTFLIGHT.json')['unknown_probe_uids'],test_labels_read=False));return
    f=Features(p,arm);source_names,X,S,F,V,SV,FV,mean,scale,lengths,signature=training_data(task,p,f)
    shared=(d/'shared_sparse_fits'/arm) if task['coverage']=='SPARSE' else out
    shared.mkdir(parents=True,exist_ok=True);frozen=shared/'MODEL_FROZEN.json'
    net=Net();pred=jax.jit(lambda pa,x:net.apply(pa,x));start=time.time();optimized=False
    if not frozen.exists():
        assert task['coverage']!='SPARSE' or (task['split']=='LOPO' and task['fold']==0),'SPARSE source fitting must precede duplicate folds'
        optimized=True
        x=jnp.asarray(X);v=jnp.asarray(V);sj=jnp.asarray(S);fj=jnp.asarray(F);nc=len(source_names)
        banks=[np.argwhere(S[i]+F[i]>0).astype(np.int32) for i in range(nc)];assert all(len(b)>0 for b in banks)
        nmean=float(np.mean([S[i,j,k]+F[i,j,k] for i,b in enumerate(banks) for j,k in b]));valid=float((SV+FV).sum());assert valid>0
        opt=optax.chain(optax.clip_by_global_norm(5),optax.adamw(.0008,weight_decay=.0001))
        @jax.jit
        def block(pa,op,cc,ii,jj):
            def step(carry,ids):
                params,state=carry;c,i,j=ids
                def loss(pp):
                    z=pred(pp,x[c,i]);chosen=jnp.take_along_axis(z,j[:,None],1)[:,0]
                    return jnp.mean(sj[c,i,j]*jax.nn.softplus(-chosen)+fj[c,i,j]*jax.nn.softplus(chosen))/nmean
                _,grad=jax.value_and_grad(loss)(params);updates,state=opt.update(grad,state,params)
                return (optax.apply_updates(params,updates),state),None
            return jax.lax.scan(step,(pa,op),(cc,ii,jj))[0]
        schedule=[50,100]+list(range(200,12001,200));histories=[];snapshots=[]
        for seed in (17,23,41):
            rng=np.random.default_rng(seed);pa=net.init(jax.random.PRNGKey(seed),x[:1,:1]);op=opt.init(pa);prev=0;hist=[];snaps=[]
            for target in schedule:
                cc=rng.integers(nc,size=(target-prev,128));ii=np.empty_like(cc);jj=np.empty_like(cc)
                for code in range(nc):
                    where=cc==code;b=banks[code];draw=b[rng.integers(len(b),size=int(where.sum()))];ii[where]=draw[:,0];jj[where]=draw[:,1]
                pa,op=block(pa,op,jnp.asarray(cc),jnp.asarray(ii),jnp.asarray(jj))
                z=np.asarray(pred(pa,v.reshape(-1,536))).reshape(len(V),24,16)
                nll=float((SV*np.logaddexp(0,-z)+FV*np.logaddexp(0,z)).sum()/valid);hist.append(nll);snaps.append(serialization.to_bytes(pa));prev=target
            histories.append(hist);snapshots.append(snaps)
        selected=int(np.argmin(np.mean(histories,axis=0)));step=schedule[selected];paths=[]
        for seed,snap in zip((17,23,41),snapshots):
            path=shared/f'seed{seed}_step{step}.msgpack';path.write_bytes(snap[selected]);paths.append(str(path))
        np.savez_compressed(shared/'normalization.npz',mean=mean,scale=scale)
        write(frozen,dict(status='FROZEN',source_signature=signature,source_controller_names=source_names,source_state_lengths=lengths,
            selected_step=step,fit_seeds=[17,23,41],validation_nll_history=histories,checkpoints=paths,checkpoint_sha256=[sha(path) for path in paths],jax_enable_x64=False,
            source_only_normalization=True,validation_controller_names=[n for n in p['anchor_order'] if n in source_names],optimizer='AdamW0.0008 wd0.0001 clip5',maximum_steps=12000,
            imported_net_source=str(ROOT/'diagnostics/controller_loco_real_flow_v1/train.py'),imported_net_sha256=sha(ROOT/'diagnostics/controller_loco_real_flow_v1/train.py'),source_fit_seconds=time.time()-start))
    meta=read(frozen);assert meta['source_signature']==signature
    norm=np.load(shared/'normalization.npz');assert np.array_equal(norm['mean'],mean) and np.array_equal(norm['scale'],scale)
    assert set(task['held']).isdisjoint(source_names)
    test=p['states'][p['train_states']:];train=p['states'][:p['train_states']]
    h=np.array([(f.inputs(n,test)-mean)/scale for n in task['held']]);c=np.array([(f.inputs(n,train[:4])-mean)/scale for n in task['held']]);prior=np.array([(f.inputs(n,train)-mean)/scale for n in task['held']])
    wrong=np.array([[(f.inputs(n,test)-mean)/scale for n in source_names] for held in task['held']])
    # Same physical test state, wrong source descriptor. Independent of held UID.
    template=net.init(jax.random.PRNGKey(0),np.zeros((1,536),np.float32));held=[];mismatched=[];cal=[];pri=[];val=[]
    for path,expected in zip(meta['checkpoints'],meta['checkpoint_sha256']):
        assert sha(path)==expected;pa=serialization.from_bytes(template,Path(path).read_bytes())
        held.append(np.asarray(pred(pa,h.reshape(-1,536))).reshape(len(task['held']),32,16))
        mismatched.append(np.asarray(pred(pa,wrong.reshape(-1,536))).reshape(len(task['held']),len(source_names),32,16))
        cal.append(np.asarray(pred(pa,c.reshape(-1,536))).reshape(len(task['held']),4,16))
        pri.append(np.asarray(pred(pa,prior.reshape(-1,536))).reshape(len(task['held']),len(train),16))
        val.append(np.asarray(pred(pa,V.reshape(-1,536))).reshape(len(V),24,16))
    np.savez_compressed(out/'predictions.npz',held=np.array(held),mismatched=np.array(mismatched),target_calibration=np.array(cal),controller_prior=np.array(pri),source_validation=np.array(val),held_controller_names=np.array(task['held']),mismatch_controller_names=np.array(source_names))
    np.savez_compressed(out/'normalization.npz',mean=mean,scale=scale)
    write(out/'complete.json',dict(status='COMPLETE',task=task,source_signature=signature,source_controller_names=source_names,
        checkpoints=meta['checkpoints'],checkpoint_sha256=meta['checkpoint_sha256'],fit_seeds=[17,23,41],jax_enable_x64=False,
        optimization_performed=optimized,source_only_normalization=True,test_labels_read=False,selected_step=meta['selected_step'],elapsed_seconds=time.time()-start,train_code_sha256=sha(__file__)))
    print(json.dumps(dict(task=name,optimized=optimized,seconds=time.time()-start)),flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
