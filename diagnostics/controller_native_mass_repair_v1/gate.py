"""Frozen SOURCE-only joint repair gate; no TEST inspection."""
import numpy as np
from .study import HERE,BASE,SEEDS,read,write,sha


def scores(q,d,ix):
    p=q[ix];s,f=[d[k][ix] for k in ['success','failure']];y=s/(s+f);assert np.isfinite(y).all()
    def center(a):return a-a.mean(0,keepdims=True)-a.mean(1,keepdims=True)+a.mean()
    return dict(Q_MSE=float(np.mean((p-y)**2)),interaction_MSE=float(np.mean((center(p)-center(y))**2)))


def run():
    d=dict(np.load(BASE/'dataset.npz'));mask=(d['scene']=='ring_exchange')&(d['controller']=='native_flow')&(d['source']=='native_ring512')
    tr=np.flatnonzero(mask&(d['split']=='train'));va=np.flatnonzero(mask&(d['split']=='validation'));assert len(tr)==512 and len(va)==32
    rows=[]
    for seed in SEEDS:
        for root in [BASE,HERE]:
            for arm in ['eta_only','union_native_slots']:assert (root/'fits/joint'/arm/f'seed{seed}/complete.json').exists()
        for name,root in [('original_uniform',BASE),('native20',HERE)]:
            folder=root/'fits/joint/union_native_slots'/f'seed{seed}';done=read(folder/'complete.json');q=np.load(folder/'predictions.npz')['probability'];assert sha(folder/'checkpoint.msgpack')==done['checkpoint_sha256']
            rows.append(dict(seed=seed,method=name,train=scores(q,d,tr),validation=scores(q,d,va),global_validation_regret=done['selected']['regret'],selected_step=done['selected']['step']))
    old=[r for r in rows if r['method']=='original_uniform'];new=[r for r in rows if r['method']=='native20']
    ratio=np.mean([r['train']['interaction_MSE'] for r in new])/np.mean([r['train']['interaction_MSE'] for r in old])
    improved=sum(n['train']['interaction_MSE']<o['train']['interaction_MSE'] for n,o in zip(new,old))
    flags=dict(train_interaction_ratio_atmost_090=bool(ratio<=.9),atleast2of3_train_improve=improved>=2,
        validation_Q_improves=bool(np.mean([r['validation']['Q_MSE'] for r in new])<np.mean([r['validation']['Q_MSE'] for r in old])),
        validation_interaction_improves=bool(np.mean([r['validation']['interaction_MSE'] for r in new])<np.mean([r['validation']['interaction_MSE'] for r in old])),
        global_val_regret_within_001=bool(np.mean([r['global_validation_regret'] for r in new])<=np.mean([r['global_validation_regret'] for r in old])+.01))
    passed=all(flags.values());result=dict(passed=passed,flags=flags,native_train_interaction_ratio=float(ratio),improved_seeds=improved,rows=rows,
        scope='SOURCE TRAIN/VAL only. VAL selectedcheckpoints,notindependentconfirmation. NoTESTlabels/predictions used for this gate.',
        next='Runfourheldfoldsfrozenrecipe' if passed else 'Pruneweightrestorationas sufficientrepair;no weightsweeporfourfoldexpansion',new_rollouts=0)
    write(HERE/'source_gate.json',result);print({k:v for k,v in result.items() if k!='rows'},flush=True)

if __name__=='__main__':run()
