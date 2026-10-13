"""Independent-seed LOCO selection, paired unknown bounds and max-T CI."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit

HERE=Path(__file__).resolve().parent
ARMS=['R0','R3','R4_action','R4_openloop','R3_R4_action']
COMP=['src','near','R0','mismatch']
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def picked(mat,choice):
    return np.take_along_axis(mat,choice[...,None],axis=-1)[...,0]

def main(stage_b=False):
    p=read(HERE/'protocol.json');names=p['controller_order'];gate=read(HERE/'GATE_RESULTS.json')
    extra_names=read(HERE/'stage_b/protocol.json')['controller_order'] if stage_b else []
    all_names=names+extra_names
    fits_dir=HERE/'stage_b/fits' if stage_b else HERE/'fits'
    active=gate['necessary_folds'] if stage_b else list(range(7))
    out=np.load(HERE/'LABELS.npz')['outcomes'];test=out[:,128:,:16,4:8]
    Y=(test==1).mean(-1);U=(test==-1).mean(-1)
    qtrue=np.divide(Y,1-U,out=np.full_like(Y,np.nan),where=U<1)
    target_eta=[]
    for c in range(7):
        x=out[c,128:,:,:4];s=(x==1).sum((0,2));v=((x==1)|(x==0)).sum((0,2))
        target_eta.append(int(np.divide(s,v,out=np.full(16,-1.),where=v>0).argmax()))
    source_curve=[]
    train=out[:,:128,:,:4]
    if stage_b:train=np.concatenate((train,np.load(HERE/'stage_b/LABELS.npz')['outcomes']),axis=0)
    for c in range(len(all_names)):
        s=(train[c]==1).sum(-1);v=((train[c]==1)|(train[c]==0)).sum(-1)
        source_curve.append(np.nanmean(np.divide(s,v,out=np.full_like(s,np.nan,dtype=float),where=v>0),axis=0))
    source_curve=np.asarray(source_curve);own_eta=source_curve.argmax(-1)
    nearest={};distances={}
    for held in active:
        sources=[i for i in range(len(all_names)) if i!=held]
        z=np.stack([np.load((HERE/'static_features' if c<7 else HERE/'stage_b/static_features')/f'{all_names[c]}.npz')['R4_action'][:128] for c in sources])
        h=np.load(HERE/'static_features'/f'{names[held]}.npz')['R4_action'][:128]
        mu=z.mean((0,1));sd=np.where(z.std((0,1))>1e-6,z.std((0,1)),1.)
        means=((z-mu)/sd).mean(1);hm=((h-mu)/sd).mean(0)
        dd=np.linalg.norm(means-hm,axis=1);near=sources[int(dd.argmin())]
        nearest[held]=near;distances[held]=dd.tolist()
    fits={}
    for held in active:
        for arm in ARMS:
            d=fits_dir/f'fold{held}_{arm}'
            assert (d/'complete.json').exists(),d
            z=np.load(d/'predictions.npz');fits[held,arm]=(z['held'],z['mismatched'])
            assert z['held'].shape==(3,64,16)
    rows=[];vectors={}
    for held in active:
        sources=[i for i in range(len(all_names)) if i!=held]
        src=int(source_curve[sources].mean(0).argmax());near=own_eta[nearest[held]]
        # Pair each fit seed with the same R0 seed, then average over fits.
        r0choice=fits[held,'R0'][0].argmax(-1)
        r0Y=picked(Y[held][None],r0choice).mean(0)
        r0U=picked(U[held][None],r0choice).mean(0)
        base={k:Y[held,:,j] for k,j in [('src',src),('near',near),('target_fixed',target_eta[held])]}
        base['R0']=r0Y
        for arm in ARMS:
            matched,wrong=fits[held,arm];choice=matched.argmax(-1)
            calpred=np.load(fits_dir/f'fold{held}_{arm}'/'predictions.npz')['target_calibration']
            sy=picked(Y[held][None],choice);su=picked(U[held][None],choice)
            success=sy.mean(0);unknown=su.mean(0)
            contrasts={}
            for label,j in [('src',src),('near',near),('target_fixed',target_eta[held])]:
                yy=Y[held,:,j][None];uu=U[held,:,j][None]
                contrasts[label]=(sy-yy-uu*(choice!=j)).mean(0)
            contrasts['R0']=(sy-picked(Y[held][None],r0choice)-picked(U[held][None],r0choice)*(choice!=r0choice)).mean(0)
            wrong_choice=wrong.argmax(-1)
            wy=picked(Y[held][None,None],wrong_choice)
            wu=picked(U[held][None,None],wrong_choice)
            contrasts['mismatch']=(sy[:,None,:]-wy-wu*(choice[:,None,:]!=wrong_choice)).mean((0,1))
            # First k target-train states are outcome-blind and distinct from test.
            calibration={}
            targettrain=out[held,:128,:,:4]
            heldfeat=np.load(HERE/'static_features'/f'{names[held]}.npz')
            for k in (1,2,4):
                small=targettrain[:k]
                s=(small==1).sum((0,2));v=((small==1)|(small==0)).sum((0,2))
                empirical=np.divide(s,v,out=np.full(16,np.nan),where=v>0)
                prior_curve=source_curve[sources].mean(0)
                calibrated=np.where(np.isfinite(empirical),empirical,prior_curve)
                fixed=int(calibrated.argmax());fixed_success=float(Y[held,:,fixed].mean())
                # Per-head offset is estimated only from the k labelled target train
                # states; absent η heads retain source-model logits unchanged.
                measured=np.clip(empirical,1e-3,1-1e-3)
                predicted=np.clip(expit(calpred[:,:k]).mean((0,1)),1e-3,1-1e-3)
                offset=np.zeros(16)
                offset[v>0]=np.log(measured[v>0]/(1-measured[v>0]))-np.log(predicted[v>0]/(1-predicted[v>0]))
                calchoice=(matched+offset[None,None]).argmax(-1)
                cal_success=float(picked(Y[held][None],calchoice).mean())
                calibration[str(k)]=dict(fixed_eta=fixed,fixed_success_lower=fixed_success,
                    critic_offset_success_lower=cal_success,observed_eta_heads=int((v>0).sum()))
            mse=float(np.nanmean((expit(matched)-qtrue[held][None])**2))
            truepattern=np.nanmean(qtrue[held],axis=0)-source_curve[sources].mean(0)
            predictedpattern=expit(matched).mean((0,1))-expit(fits[held,'R0'][0]).mean((0,1))
            valid=np.isfinite(truepattern)
            corr=float(np.corrcoef(truepattern[valid],predictedpattern[valid])[0,1]) if valid.sum()>2 and np.std(predictedpattern[valid])>1e-10 else None
            row=dict(fold=held,held_controller=names[held],necessary=held in gate['necessary_folds'],arm=arm,
                source_prior_eta=int(src),nearest_source=all_names[nearest[held]],nearest_fixed_eta=int(near),
                target_fixed_eta=target_eta[held],selected_success_lower=float(success.mean()),
                selected_success_upper=float((success+unknown).mean()),selected_unknown=float(unknown.mean()),
                hindsight_state_oracle_success_lower=float(Y[held].max(-1).mean()),
                hindsight_state_oracle_regret_lower=float(Y[held].max(-1).mean()-success.mean()),
                hindsight_best_fixed_success_lower=float(Y[held].mean(0).max()),
                success_by_fit=sy.mean(1).tolist(),baseline_success_lower={k:float(v.mean()) for k,v in base.items()},
                conservative_gain={k:float(v.mean()) for k,v in contrasts.items()},
                q_mse=mse,controller_eta_pattern_correlation=corr,
                target_supervised_fixed_calibration=calibration)
            rows.append(row);vectors[held,arm]=contrasts
    necessary=gate['necessary_folds'];primary=None
    if len(necessary)>=2:
        cols=[(arm,comp) for arm in ARMS[1:] for comp in COMP]
        a=np.asarray([np.mean([vectors[held,arm][comp] for held in necessary],axis=0) for arm,comp in cols]).T
        obs=a.mean(0);rng=np.random.default_rng(2026101206)
        draws=rng.integers(64,size=(5000,64));boot=a[draws].mean(1)
        se=boot.std(0,ddof=1);denom=np.where(se>1e-12,se,1.)
        t=np.max(np.abs((boot-obs[None])/denom[None]),axis=1);critical=float(np.quantile(t,.95))
        lower=obs-critical*se;upper=obs+critical*se
        contrasts=[dict(arm=arm,comparison=comp,mean=float(obs[i]),ci95_simultaneous=[float(lower[i]),float(upper[i])]) for i,(arm,comp) in enumerate(cols)]
        pass_arms=[arm for arm in ARMS[1:] if all(lower[cols.index((arm,c))]>0 for c in COMP)]
        primary=dict(status=('DEVELOPMENT CONDITIONAL PASS' if stage_b else 'CONFIRMED INDEPENDENT-SEED EVALUATION') if pass_arms else 'NO ARM PASSED',
            necessary_folds=necessary,contrast_family_size=len(cols),bootstrap_draws=5000,bootstrap_seed=2026101206,
            simultaneous_method='paired physical-state max-T, same resampled state indices for all held controllers',
            unknown_policy='lower bound: selector unknown=0, comparator unknown=1 only when η differs; identical outcomes cancel',
            max_t_critical=critical,contrasts=contrasts,passing_arms=pass_arms)
    else:
        primary=dict(status='INCONCLUSIVE: FEWER THAN TWO NECESSARY FOLDS',necessary_folds=necessary,passing_arms=[])
    result=dict(status='COMPLETE',stage_b=stage_b,source_controller_count=(13 if stage_b else 6),
        test_states=64,endpoint_seeds=[4,5,6,7],fit_seeds=[17,23,41],
        rows=rows,nearest_source_distances={str(k):v for k,v in distances.items()},primary=primary,
        target_fixed_eta_rule='test seeds0-3 target supervised, reference only',
        calibration_rule='target S_train labels: first k natural state; missing η heads fall back to source pooled prior')
    write(HERE/'stage_b/results.json' if stage_b else HERE/'results.json',result)
    print(json.dumps(primary,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage-b',action='store_true')
    a=ap.parse_args();main(a.stage_b)
