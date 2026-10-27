"""Source-only per-seed fits with the prescribed sampling, loss and early stopping."""
from common import *
from networks import Generator,critic,stable_int,log_prob,CENTER,RADIUS
from flax import serialization
import jax,jax.numpy as jnp,optax,numpy as np

def draw(mask,rng):
    counts=mask.sum(-1);assert np.all(counts>0)
    return np.argmax(np.cumsum(mask,axis=-1)>rng.integers(counts)[:,None],axis=-1)

def dataset(fold):
    assert not(set(fold['sources'])&set(fold['targets']))
    p=Path(fold['source_labels_path']);assert sha(p)==fold['training_data_sha256']
    norm=fold['normalization'];data={}
    with np.load(p,allow_pickle=False)as z:
        assert {k.split('__')[0]for k in z.files}==set(fold['sources'])
        for n in fold['sources']:
            h=(z[n+'__h']-norm['h_mean'])/norm['h_scale'];c=(z[n+'__c']-norm['c_mean'])/norm['c_scale']
            y=z[n+'__y'].transpose(1,0,2);q=y.mean(1);positive=q==q.max(1)[:,None];eligible=q.max(1)>0
            data[n]=dict(h=h.astype(np.float32),c=c.astype(np.float32),y=y.astype(np.float32),onehot=z[n+'__onehot'].astype(np.float32),
                positive=positive,negative=q==0,eligible=eligible,train=z[n+'__train'],val=z[n+'__val'],state_indices=z[n+'__state_indices'])
    return data

def gen_batch(d,rng,etas):
    ids=d['train'][d['eligible'][d['train']]];assert len(ids)>0
    i=ids[rng.integers(len(ids),size=64)];pm=d['positive'][i];nm=d['negative'][i];p=draw(pm,rng);has=nm.any(1)
    n=draw(np.where(has[:,None],nm,pm),rng);n=np.where(has,n,p)
    return d['h'][i],d['c'][i],etas[p],etas[n],has.astype(np.float32)

def crit_batch(d,rng,etas):
    i=d['train'][rng.integers(len(d['train']),size=96)];ctl=rng.integers(d['y'].shape[1],size=96);y=d['y'][i,ctl]
    pm=y==1;nm=y==0;pos=(rng.random(96)<.5)&pm.any(1)
    use=np.where(pos[:,None],pm,np.where(nm.any(1)[:,None],nm,pm));a=draw(use,rng)
    return d['h'][i],d['c'][i],d['onehot'][ctl],((etas[a]-CENTER)/RADIUS).astype(np.float32),y[np.arange(96),a]

def validation_arrays(data,kind,etas):
    args=[];weights=[];total_scenes=len(data)
    if kind=='critic':
        H=[];C=[];CTL=[];E=[];Y=[];W=[]
        for n,d in data.items():
            ids=d['val'];nc=d['y'].shape[1];ii,ct,aa=np.meshgrid(ids,np.arange(nc),np.arange(49),indexing='ij');ii,ct,aa=[v.ravel()for v in (ii,ct,aa)]
            H.append(d['h'][ii]);C.append(d['c'][ii]);CTL.append(d['onehot'][ct]);E.append((etas[aa]-CENTER)/RADIUS);Y.append(d['y'][ii,ct,aa])
            W.append(np.full(len(ii),1/(total_scenes*len(ids)*nc*49),np.float32))
        return tuple(jnp.asarray(np.concatenate(v),jnp.float32)for v in (H,C,CTL,E,Y,W))
    # Each validation state averages all tied positive candidates. Ranking uses
    # the same deterministic per-state stable_int indices as the historical fit.
    H=[];C=[];E=[];W=[];RH=[];RC=[];RP=[];RN=[];RW=[]
    for n,d in data.items():
        ids=d['val'][d['eligible'][d['val']]];assert len(ids)>0
        for i in ids:
            p=np.flatnonzero(d['positive'][i]);neg=np.flatnonzero(d['negative'][i]);weight=1/(total_scenes*len(ids))
            H.extend([d['h'][i]]*len(p));C.extend([d['c'][i]]*len(p));E.extend(etas[p]);W.extend([weight/len(p)]*len(p))
            if len(neg):
                uid_=f"{n}:{int(d['state_indices'][i])}"
                a=p[stable_int('val-pos',uid_)%len(p)];b=neg[stable_int('val-neg',uid_)%len(neg)]
                RH.append(d['h'][i]);RC.append(d['c'][i]);RP.append(etas[a]);RN.append(etas[b]);RW.append(weight)
    if not RH:
        RH=[np.zeros(6)];RC=[np.zeros(12)];RP=[etas[0]];RN=[etas[0]];RW=[0.]
    return tuple(jnp.asarray(v,jnp.float32)for v in (H,C,E,W,RH,RC,RP,RN,RW))

def fit(task):
    fold=read(HERE/'folds'/task['fold']/'definition.json');kind=task['kind'];mode=task.get('mode','generator');seed=task['seed']
    folder=HERE/'folds'/task['fold']/mode/f'seed{seed}';receipt=folder/'FIT.receipt.json'
    if receipt.exists():
        row=read(receipt);assert sha(row['checkpoint'])==row['sha256'];return row
    folder.mkdir(parents=True,exist_ok=True);data=dataset(fold);etas=np.asarray([a['eta_a']for a in actions()],np.float32)
    model=Generator()if kind=='generator'else critic(mode)
    # The historical train_generator/train_critic merge_initialized use key0;
    # seeds17/23/41 change their independent per-scene sampling streams.
    dummy=[jnp.zeros((1,6)),jnp.zeros((1,12))]
    if kind=='critic':dummy.extend([jnp.zeros((1,8)),jnp.zeros((1,3))])
    params=model.init(jax.random.PRNGKey(0),*dummy)
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(5e-4 if kind=='generator'else 1e-3,weight_decay=1e-4));state=opt.init(params)
    @jax.jit
    def step(p,st,args):
        def objective(q):
            if kind=='generator':
                h,c,ep,en,hn=args;raw=model.apply(q,h,c);lp=log_prob(raw,ep);ln=log_prob(raw,en)
                rank=jnp.sum(hn*jax.nn.softplus(.5+ln-lp),axis=1)/jnp.maximum(hn.sum(1),1)
                return jnp.mean(-lp.mean(1)+.25*rank)
            h,c,ct,e,y=args;logits=model.apply(q,h,c,ct,e)
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits,y))
        loss,grad=jax.value_and_grad(objective)(p);updates,st=opt.update(grad,st,p)
        return optax.apply_updates(p,updates),st,loss
    va=validation_arrays(data,kind,etas)
    @jax.jit
    def validation(p):
        if kind=='critic':
            h,c,ct,e,y,w=va;logits=model.apply(p,h,c,ct,e)
            return jnp.sum(w*optax.sigmoid_binary_cross_entropy(logits,y))
        h,c,e,w,rh,rc,rp,rn,rw=va;raw=model.apply(p,h,c);r=model.apply(p,rh,rc)
        return -jnp.sum(w*log_prob(raw,e))+.25*jnp.sum(rw*jax.nn.softplus(.5+log_prob(r,rn)-log_prob(r,rp)))
    rngs={n:np.random.default_rng(stable_int(kind,seed,n))for n in data};best=(float('inf'),None,0);history=[];stale=0
    max_steps=3000 if kind=='generator'else 4000;patience=8 if kind=='generator'else 10;minimum=1200 if kind=='generator'else 1500
    before=time.monotonic()
    for iteration in range(1,max_steps+1):
        batches=[gen_batch(d,rngs[n],etas)if kind=='generator'else crit_batch(d,rngs[n],etas)for n,d in data.items()]
        args=tuple(jnp.asarray(np.stack([v[k]for v in batches]))for k in range(len(batches[0])))
        params,state,loss=step(params,state,args)
        if iteration%100==0:
            val=float(validation(params));assert np.isfinite(val),(task,'NONFINITE_VALIDATION',val)
            history.append(dict(step=iteration,train=float(loss),validation=val,elapsed_seconds=time.monotonic()-before))
            if val<best[0]-1e-5:best=(val,serialization.to_bytes(params),iteration);stale=0
            else:stale+=1
            write(folder/'history.json',history)
            write(folder/'LIVE.json',dict(step=iteration,best_step=best[2],best_validation=best[0],stale=stale,elapsed_seconds=time.monotonic()-before))
            if stale>=patience and iteration>=minimum:break
    checkpoint=folder/'checkpoint.msgpack';checkpoint.write_bytes(best[1])
    row=dict(utc=utc(),fold=task['fold'],kind=kind,mode=mode,seed=seed,validation_loss=best[0],best_step=best[2],steps=iteration,
        elapsed_seconds=time.monotonic()-before,checkpoint=str(checkpoint),sha256=sha(checkpoint),
        training_scenes=fold['sources'],heldout_scenes=fold['targets'],training_data_sha256=fold['training_data_sha256'],
        normalization=fold['normalization'],initialization_key=0,source_only_early_stopping=True,heldout_labels_used=False,
        implementation_sources={str(HERE/n):sha(HERE/n)for n in ('train_model.py','networks.py')})
    write(receipt,row);print('FIT',task['fold'],mode,seed,iteration,'steps',round(row['elapsed_seconds'],2),'seconds',flush=True)
    return row

def main():fit(json.loads(sys.argv[1]))
if __name__=='__main__':main()
