"""Paired Stage B (13-source) minus Stage A (6-source) selection change."""
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
HERE=BASE/'stage_b'
ARMS=['R0','R3','R4_action','R4_openloop','R3_R4_action']
def read(p):return json.loads(Path(p).read_text())
def pick(mat,choice):return np.take_along_axis(mat,choice[...,None],axis=-1)[...,0]

def main():
    gate=read(BASE/'GATE_RESULTS.json');folds=gate['necessary_folds']
    assert len(folds)>=2 and read(HERE/'results.json')['status']=='COMPLETE'
    labels=np.load(BASE/'LABELS.npz')['outcomes'][:,128:,:16,4:8]
    Y=(labels==1).mean(-1);U=(labels==-1).mean(-1)
    vectors=[]
    for arm in ARMS:
        per=[]
        for fold in folds:
            a=np.load(BASE/'fits'/f'fold{fold}_{arm}'/'predictions.npz')['held'].argmax(-1)
            b=np.load(HERE/'fits'/f'fold{fold}_{arm}'/'predictions.npz')['held'].argmax(-1)
            by=pick(Y[fold][None],b);ay=pick(Y[fold][None],a);au=pick(U[fold][None],a)
            per.append((by-ay-au*(a!=b)).mean(0))
        vectors.append(np.mean(per,axis=0))
    v=np.asarray(vectors).T;obs=v.mean(0)
    rng=np.random.default_rng(2026101210);draw=rng.integers(64,size=(5000,64));boot=v[draw].mean(1)
    se=boot.std(0,ddof=1);t=np.max(np.abs((boot-obs[None])/np.where(se>1e-12,se,1.)),axis=1)
    critical=float(np.quantile(t,.95));lower=obs-critical*se;upper=obs+critical*se
    result=dict(status='DEVELOPMENT SOURCE-COUNT COMPARISON',folds=folds,old_sources=6,new_sources=13,
                same_held_state_seed_panel=True,unknown_policy='paired conservative lower; same selected η cancels',
                bootstrap_draws=5000,bootstrap_seed=2026101210,simultaneous_method='max-T across five arms',
                rows=[dict(arm=arm,stage_b_minus_stage_a_lower=float(obs[i]),ci95_simultaneous=[float(lower[i]),float(upper[i])]) for i,arm in enumerate(ARMS)])
    (HERE/'SOURCE_COUNT_COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
