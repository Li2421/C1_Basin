"""Exploratory prior-versus-state selection, never changes primary decisions."""
import json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
ARMS=['R0','R3','R4_action','R4_openloop','R3_R4_action']

def read(p):return json.loads(Path(p).read_text())
def picked(y,j):return np.take_along_axis(y,j[...,None],axis=-1)[...,0]

def panel(stage,folds):
    confirmation=stage=='stage_c'
    if confirmation:
        protocol=read(BASE/'stage_c/protocol.json')
        outcomes=np.load(BASE/'stage_c/LABELS.npz')['outcomes']
        arms=['R0']+protocol['passing_arms']
    else:
        outcomes=np.load(BASE/'LABELS.npz')['outcomes'][:,128:]
        arms=ARMS
    rows=[];vectors={}
    for fold in folds:
        count=int(protocol['allocation'][str(fold)]) if confirmation else 64
        x=outcomes[fold,:count,:,4:8];Y=(x==1).mean(-1);U=(x==-1).mean(-1)
        assert np.all(x>=-1)
        for arm in arms:
            source='stage_b' if confirmation else stage
            prior=np.load(BASE/'controller_prior'/f'{source}_fold{fold}_{arm}.npz')['eta_prior']
            if confirmation:
                logits=np.load(BASE/'stage_c/predictions'/f'fold{fold}_{arm}.npz')['held'][:,:count]
            else:
                folder=BASE/'fits' if stage=='stage_a' else BASE/'stage_b/fits'
                logits=np.load(folder/f'fold{fold}_{arm}'/'predictions.npz')['held']
            choice=logits.argmax(-1);j=np.broadcast_to(prior[:,None],choice.shape)
            selected=picked(Y[None],choice);baseline=picked(Y[None],j)
            diff=(selected-baseline-picked(U[None],j)*(choice!=j)).mean(0)
            vectors[fold,arm]=diff
            rows.append(dict(stage=stage,fold=fold,arm=arm,states=count,
                learned_controller_prior_eta_by_fit=prior.tolist(),selected_success_lower=float(selected.mean()),
                controller_prior_success_lower=float(baseline.mean()),state_gain_lower=float(diff.mean()),
                eta_selection_histogram_by_fit=[np.bincount(c,minlength=16).tolist() for c in choice],
                fraction_different_from_controller_prior=float((choice!=j).mean())))
    cols=[arm for arm in arms if arm!='R0'];n=len(vectors[folds[0],cols[0]])
    assert all(len(v)==n for v in vectors.values())
    rng=np.random.default_rng(2026101212);draw=rng.integers(n,size=(5000,n))
    obs=np.array([np.mean([vectors[f,arm].mean() for f in folds]) for arm in cols])
    boot=np.stack([np.mean([vectors[f,arm][draw].mean(1) for f in folds],axis=0) for arm in cols],axis=1)
    se=boot.std(0,ddof=1);critical=float(np.quantile(np.max(np.abs((boot-obs)/np.where(se>1e-12,se,1)),axis=1),.95))
    contrasts=[dict(arm=arm,state_selection_minus_learned_controller_prior_lower=float(obs[i]),
                    ci95_simultaneous=[float(obs[i]-critical*se[i]),float(obs[i]+critical*se[i])]) for i,arm in enumerate(cols)]
    return dict(status='EXPLORATORY',stage=stage,necessary_folds=folds,rows=rows,contrasts=contrasts,
                bootstrap_draws=5000,bootstrap_seed=2026101212,simultaneous_method='paired physical-state max-T within panel',
                unknown_policy='paired conservative lower; identical η cancels',
                prior_rule='mean predicted sigmoid Q over 128 outcome-blind original train initial states; no target labels')

def main():
    folds=read(BASE/'GATE_RESULTS.json')['necessary_folds']
    result={stage:panel(stage,folds) for stage in ['stage_a','stage_b']}
    if (BASE/'stage_c/results.json').exists():result['stage_c']=panel('stage_c',folds)
    p=BASE/'controller_prior/results.json';p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v['contrasts'] for k,v in result.items()},indent=2))
    return result

if __name__=='__main__':main()
