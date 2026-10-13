"""Independent frozen-model confirmation with paired shared-state max-T CI."""
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
HERE=BASE/'stage_c'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def picked(mat,choice):return np.take_along_axis(mat,choice[...,None],axis=-1)[...,0]

def main():
    p=read(HERE/'protocol.json');stage_b=read(BASE/'stage_b/results.json')
    passed=p['passing_arms'];folds=p['held_folds'];names=p['controller_order']
    assert passed and len(folds)>=2 and set(passed)==set(stage_b['primary']['passing_arms'])
    out=np.load(HERE/'LABELS.npz')['outcomes'];rows=[];vectors={}
    source_rules={r['fold']:r for r in stage_b['rows'] if r['arm']=='R0'}
    for fold in folds:
        count=int(p['allocation'][str(fold)])
        x=out[fold,:count]
        assert np.all(x>=-1)
        discovery=x[:,:,:4];evaluate=x[:,:,4:]
        Y=(evaluate==1).mean(-1);U=(evaluate==-1).mean(-1)
        s=(discovery==1).sum((0,2));v=((discovery==1)|(discovery==0)).sum((0,2))
        target_eta=int(np.divide(s,v,out=np.full(16,-1.),where=v>0).argmax())
        src=int(source_rules[fold]['source_prior_eta']);near=int(source_rules[fold]['nearest_fixed_eta'])
        rp=np.load(HERE/'predictions'/f'fold{fold}_R0.npz')['held'][:,:count]
        r0=rp.argmax(-1)
        for arm in passed:
            z=np.load(HERE/'predictions'/f'fold{fold}_{arm}.npz')
            matched=z['held'][:,:count];wrong=z['mismatched'][:,:,:count]
            choice=matched.argmax(-1);wrong_choice=wrong.argmax(-1)
            sy=picked(Y[None],choice);su=picked(U[None],choice)
            contrasts={}
            for key,j in [('src',src),('near',near),('target_fixed',target_eta)]:
                contrasts[key]=(sy-Y[:,j][None]-U[:,j][None]*(choice!=j)).mean(0)
            contrasts['R0']=(sy-picked(Y[None],r0)-picked(U[None],r0)*(choice!=r0)).mean(0)
            wy=picked(Y[None,None],wrong_choice);wu=picked(U[None,None],wrong_choice)
            contrasts['mismatch']=(sy[:,None,:]-wy-wu*(choice[:,None,:]!=wrong_choice)).mean((0,1))
            vectors[fold,arm]=contrasts
            rows.append(dict(fold=fold,controller=names[fold],arm=arm,states=count,selected_success_lower=float(sy.mean()),
                selected_unknown=float(su.mean()),source_prior_eta=src,nearest_eta=near,target_discovery_fixed_eta=target_eta,
                source_prior_success_lower=float(Y[:,src].mean()),nearest_success_lower=float(Y[:,near].mean()),
                R0_success_lower=float(picked(Y[None],r0).mean()),target_fixed_success_lower=float(Y[:,target_eta].mean()),
                conservative_gains={k:float(v.mean()) for k,v in contrasts.items()}))
    cols=[(arm,c) for arm in passed for c in ('src','near','R0','mismatch')]
    obs=np.asarray([np.mean([vectors[f,arm][c].mean() for f in folds]) for arm,c in cols])
    rng=np.random.default_rng(2026101209);maxn=max(int(p['allocation'][str(f)]) for f in folds)
    draw=rng.integers(maxn,size=(5000,maxn));boot=[]
    for arm,c in cols:
        per=[]
        for f in folds:
            n=int(p['allocation'][str(f)]);v=vectors[f,arm][c]
            ids=draw<n;values=np.where(ids,v[np.minimum(draw,n-1)],0.)
            per.append(values.sum(1)/np.maximum(ids.sum(1),1))
        boot.append(np.mean(per,axis=0))
    boot=np.stack(boot,axis=1);se=boot.std(0,ddof=1)
    t=np.max(np.abs((boot-obs[None])/np.where(se>1e-12,se,1.)),axis=1)
    critical=float(np.quantile(t,.95));lower=obs-critical*se;upper=obs+critical*se
    contrasts=[dict(arm=arm,comparison=c,mean=float(obs[i]),ci95_simultaneous=[float(lower[i]),float(upper[i])]) for i,(arm,c) in enumerate(cols)]
    confirmed=[arm for arm in passed if all(lower[cols.index((arm,c))]>0 for c in ('src','near','R0','mismatch'))]
    result=dict(status='CONFIRMED' if confirmed else 'FAILED INDEPENDENT CONFIRMATION',
        frozen_stage_b_passing_arms=passed,confirmed_arms=confirmed,necessary_folds=folds,allocation=p['allocation'],
        rows=rows,contrasts=contrasts,bootstrap_draws=5000,bootstrap_seed=2026101209,
        simultaneous_method='max-T; common physical-state resampling across folds with differing allocated counts',
        unknown_policy='paired conservative lower; identical η outcomes cancel',
        new_rollouts=read(HERE/'CACHE_INVENTORY.json')['missing'])
    write(HERE/'results.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
