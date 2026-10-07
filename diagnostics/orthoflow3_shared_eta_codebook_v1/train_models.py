#!/usr/bin/env python3
"""Train aligned feasibility baselines and small MLPs on binomial counts."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,time
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata
H=Path(__file__).parent;SEEDS=(17,23,41)
def read(p):return list(csv.DictReader(open(p)))
def write(name,rows,fields=None):
    p=H/name;p.parent.mkdir(parents=True,exist_ok=True);fields=fields or list(rows[0])
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(name,x):p=H/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Linear(nn.Module):
    M:int
    @nn.compact
    def __call__(self,x):return nn.Dense(self.M,kernel_init=nn.initializers.zeros,bias_init=nn.initializers.zeros)(x)
class MLP(nn.Module):
    M:int
    @nn.compact
    def __call__(self,x):x=nn.silu(nn.Dense(64)(x));x=nn.silu(nn.Dense(64)(x));return nn.Dense(self.M)(x)
def arrays():
    a=np.load(H/'state_features.npz');ids=a['state_ids'];sp=a['splits'];M=json.load(open(H/'codebook_manifest.json'))['M'];Y=np.zeros((len(ids),M));N=np.zeros((len(ids),M));idx={str(s):i for i,s in enumerate(ids)}
    for fn in ('train_mode_counts.csv','val_mode_counts.csv','test_mode_q64.csv'):
        for r in read(H/fn):i=idx[r['state_id']];m=int(r['mode_id']);Y[i,m]=float(r['successes'])/float(r['trials']);N[i,m]=float(r['trials'])
    if np.any(N==0):raise RuntimeError('incomplete matrix')
    return a['x'].astype(np.float32),Y.astype(np.float32),N.astype(np.float32),sp,ids
def metrics(logits,y):
    p=1/(1+np.exp(-np.clip(logits,-30,30)));eps=1e-7;nll=float(np.mean(-(y*np.log(p+eps)+(1-y)*np.log(1-p+eps))));brier=float(np.mean((p-y)**2));r=[]
    for a,b in zip(p,y):
        if np.std(a)>1e-12 and np.std(b)>1e-12:r.append(float(np.corrcoef(rankdata(a),rankdata(b))[0,1]))
    top=np.argmax(p,axis=1);t3=np.argsort(p,axis=1)[:,-3:]
    ece=0.
    for lo in np.linspace(0,1,11)[:-1]:
        z=(p>=lo)&(p<(lo+.1 if lo<.9 else 1.000001))
        if z.any():ece += z.mean()*abs(float(p[z].mean()-y[z].mean()))
    return dict(nll=nll,brier=brier,within_state_rank_correlation=float(np.mean(r)) if r else None,top1_empirical_Q=float(np.mean(y[np.arange(len(y)),top])),top3_empirical_Q=float(np.mean([y[i,t3[i]].max() for i in range(len(y))])),ece=float(ece))
def train_model(kind,seed):
    x,y,n,sp,ids=arrays();tr=np.where(sp=='train')[0];va=np.where(sp=='val')[0];M=y.shape[1];model=Linear(M) if kind=='linear' else MLP(M);p=model.init(jax.random.PRNGKey(seed),jnp.zeros((1,214)));opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(1e-3,weight_decay=1e-4));ost=opt.init(p);rng=np.random.default_rng(seed);best=None;bv=math.inf;be=-1;wait=0;hist=[];start=time.time()
    @jax.jit
    def step(p,ost,xb,yb):
        def lf(pp):return jnp.mean(optax.sigmoid_binary_cross_entropy(model.apply(pp,xb),yb))
        v,g=jax.value_and_grad(lf)(p);u,ost=opt.update(g,ost,p);return optax.apply_updates(p,u),ost,v
    maxep=1600 if kind=='linear' else 2400;pat=180
    for ep in range(maxep):
        order=rng.permutation(tr)
        for j in range(0,len(order),32):p,ost,_=step(p,ost,jnp.asarray(x[order[j:j+32]]),jnp.asarray(y[order[j:j+32]]))
        lt=np.asarray(model.apply(p,jnp.asarray(x[tr])));lv=np.asarray(model.apply(p,jnp.asarray(x[va])));mt=metrics(lt,y[tr]);mv=metrics(lv,y[va]);hist.append(dict(epoch=ep,train_nll=mt['nll'],val_nll=mv['nll'],train_brier=mt['brier'],val_brier=mv['brier'],train_top1_Q=mt['top1_empirical_Q'],val_top1_Q=mv['top1_empirical_Q']))
        if mv['nll']<bv-1e-7:bv=mv['nll'];be=ep;best=jax.tree_util.tree_map(lambda z:np.asarray(z),p);wait=0
        else:wait+=1
        if wait>=pat:break
    name='linear' if kind=='linear' else f'mlp_seed{seed}';d=H/name;d.mkdir(exist_ok=True);ck=d/'checkpoint.msgpack';ck.write_bytes(serialization.to_bytes(best));write(f'{name}/training_history.csv',hist);pred={s:np.asarray(model.apply(best,jnp.asarray(x[sp==s]))) for s in ('train','val','test')};sm=dict(model=name,kind=kind,seed=seed,best_epoch=be,best_val_nll=bv,epochs=len(hist),training_seconds=time.time()-start,checkpoint=str(ck),checkpoint_sha256=sha(ck),metrics={s:metrics(pred[s],y[sp==s]) for s in pred});dump(f'{name}/summary.json',sm);np.savez_compressed(d/'predictions.npz',**pred);print(json.dumps(sm))
def baselines():
    x,y,n,sp,ids=arrays();tr=sp=='train';prior=np.clip(y[tr].mean(0),1e-5,1-1e-5);pl=np.log(prior/(1-prior));out={}
    gp=H/'global_prior';gp.mkdir(exist_ok=True);gsm={'model':'global_prior','metrics':{s:metrics(np.tile(pl,(sum(sp==s),1)),y[sp==s]) for s in ('train','val','test')}};dump('global_prior/summary.json',gsm)
    nn=H/'nearest_neighbor';nn.mkdir(exist_ok=True);pred={}
    ti=np.where(tr)[0]
    for s in ('train','val','test'):
        qi=np.where(sp==s)[0];z=[]
        for i in qi:
            cand=ti[ti!=i] if s=='train' else ti;d=np.linalg.norm(x[cand]-x[i],axis=1);z.append(y[cand[np.argmin(d)]])
        pp=np.clip(np.asarray(z),1e-4,1-1e-4);pred[s]=np.log(pp/(1-pp))
    nsm={'model':'nearest_neighbor','metrics':{s:metrics(pred[s],y[sp==s]) for s in pred}};dump('nearest_neighbor/summary.json',nsm);np.savez_compressed(nn/'predictions.npz',**pred)
    print(json.dumps({'global':gsm['metrics']['val'],'nearest':nsm['metrics']['val']}))
def aggregate():
    x,y,n,sp,ids=arrays();summ=[json.load(open(H/'global_prior/summary.json')),json.load(open(H/'nearest_neighbor/summary.json')),json.load(open(H/'linear/summary.json'))]+[json.load(open(H/f'mlp_seed{s}/summary.json')) for s in SEEDS]
    rows=[]
    for q in summ:
        for s in ('train','val','test'):rows.append(dict(model=q['model'],split=s,**q['metrics'][s]))
    write('training_summary.csv',rows)
    # Primary family is the predeclared MLP; choose seed by VAL NLL then Brier.
    mlp=[q for q in summ if q['model'].startswith('mlp_')];win=min(mlp,key=lambda q:(q['metrics']['val']['nll'],q['metrics']['val']['brier'],q['seed']));dump('selected_model.json',win)
    lp=np.load(H/win['model']/'predictions.npz');vl=lp['val'];vy=y[sp=='val']
    def obj(logt):return metrics(vl/np.exp(logt),vy)['nll']
    z=minimize_scalar(obj,bounds=(-3,3),method='bounded');T=float(np.exp(z.x));raw=metrics(vl,vy);cal=metrics(vl/T,vy);use=cal['nll']<raw['nll']-1e-9
    dump('calibration.json',{'temperature':T,'used':use,'raw_val':raw,'calibrated_val':cal,'selection':'one scalar, VAL only'});w=json.load(open(H/'working_state.json'));w.update(status='MODELS_TRAINED',completed=w['completed']+['baselines','linear','mlp_seeds','temperature_calibration'],next_action='select VAL deployment threshold and evaluate TEST');dump('working_state.json',w);print(json.dumps({'selected':win['model'],'T':T,'use':use,'val':cal if use else raw}))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['baselines','train','aggregate']);ap.add_argument('--kind',choices=['linear','mlp']);ap.add_argument('--seed',type=int);a=ap.parse_args();{'baselines':baselines,'train':lambda:train_model(a.kind,a.seed),'aggregate':aggregate}[a.stage]()
if __name__=='__main__':main()
