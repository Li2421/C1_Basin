"""Zero rollout TT LOSO replay. Target labels are opened only after alpha freeze."""
from __future__ import annotations
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
PROJECT=BASE.parents[1]
ORIGINAL=BASE.parent/'orthoflow3_loso_partial_count_v1'
TARGETS=BASE.parent/'orthoflow3_loso_root_cause_v1'
FOLDS=('toy','db','four','ring')
SEEDS=(17,23,41)
ALPHAS=(0.,.25,.5,.75,1.)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def csvout(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader();w.writerows(rows)

def register():
    assert not (BASE/'protocol.json').exists()
    files=[ORIGINAL/'all_pairs.parquet',ORIGINAL/'entities.npz',ORIGINAL/'target_predictions.json',ORIGINAL/'cached_truth.json',ORIGINAL/'protocol.json']
    for fold in FOLDS:
        files.extend((ORIGINAL/fold/'partial_count/models_frozen.json',ORIGINAL/fold/'normalization.json',TARGETS/'targets'/fold/'manifest.json'))
    write(BASE/'protocol.json',dict(question='Does VAL-constrained eta-only+Full rescue strict TT LOSO selection without target tuning?',
        controller='TT only; original frozen TT controller UID and current Ring safety labels',
        scope='Four historical strict LOSO TT folds; replay is diagnostic because target panels were previously evaluated',
        alphas=ALPHAS,mixture='probability convex combination (1-alpha)*p_eta+alpha*p_Full',
        alpha_selection='minimum mean source-scene observed-continuation NLL, then mean over 3 paired seeds; target labels never used',
        original_models='partial_count source-only shared + eta_only, seeds 17/23/41, immutable checkpoints',
        primary='original source-selected Full and eta-only checkpoints mixed with fold-specific source-VAL alpha',
        stability='three paired seeds using same alpha; endpoints alpha0 and alpha1 are original controls',
        no_extra_training=True,new_rollouts=0,TT_labels_never_used_as_FF=True,
        source_hashes={str(path):sha(path) for path in files}))

def freeze():
    """Pure source-side inference. Importing target prediction/truth is prohibited here."""
    import jax,jax.numpy as jnp
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import model_for,gather
    from diagnostics.orthoflow3_loso_partial_count_v1.train import data
    assert (BASE/'protocol.json').exists() and not (BASE/'selection_frozen.json').exists()
    pp=read(BASE/'protocol.json');prediction_files=[ORIGINAL/'target_predictions.json',ORIGINAL/'cached_truth.json']
    assert all(sha(p)==pp['source_hashes'][str(p)] for p in prediction_files)
    details=[];selection={}
    for fold in FOLDS:
        rows,x,e,s,f,binary,inv,groups=data(fold,'partial_count')
        si=np.asarray([r['state_index'] for r in rows]);sources=list(groups)
        scores={}
        for kind in ('shared','eta_only'):
            net=model_for(kind)
            for seed in SEEDS:
                model_record=read(ORIGINAL/fold/'partial_count'/kind/f'seed{seed}'/'training.json')
                assert model_record['target_labels_used'] is False and sha(model_record['checkpoint'])==model_record['sha256']
                initial=net.init(jax.random.PRNGKey(0),gather(x,[si[0]]),jnp.zeros((1,3)))
                params=serialization.from_bytes(initial,Path(model_record['checkpoint']).read_bytes())
                infer=jax.jit(lambda bx,be:jax.nn.sigmoid(net.apply(params,bx,be)))
                for scene in sources:
                    ix=groups[scene]['validation'];assert len(ix)>0
                    chunks=[np.asarray(infer(gather(x,si[j]),jnp.asarray(e[j]))) for j in (ix[i:i+256] for i in range(0,len(ix),256))]
                    scores[kind,seed,scene]=np.concatenate(chunks)
        by_alpha=[]
        for alpha in ALPHAS:
            seed_scores=[]
            for seed in SEEDS:
                losses=[]
                for scene in sources:
                    ix=groups[scene]['validation'];pe=scores['eta_only',seed,scene];pf=scores['shared',seed,scene]
                    p=np.clip((1-alpha)*pe+alpha*pf,1e-7,1-1e-7)
                    nll=float(-(s[ix]*np.log(p)+f[ix]*np.log1p(-p)).sum()/(s[ix]+f[ix]).sum())
                    losses.append(nll)
                    details.append(dict(fold=fold,source_scene=scene,seed=seed,alpha=alpha,VAL_observed_NLL=nll,VAL_pairs=len(ix),VAL_trials=int((s[ix]+f[ix]).sum())))
                seed_scores.append(float(np.mean(losses)))
            by_alpha.append(float(np.mean(seed_scores)))
        best=int(np.argmin(by_alpha));a=ALPHAS[best]
        frozen=read(ORIGINAL/fold/'partial_count/models_frozen.json')
        selection[fold]=dict(alpha=a,source_scenes=sources,VAL_NLL_by_alpha={str(x):y for x,y in zip(ALPHAS,by_alpha)},
            chosen_eta_seed=frozen['eta_only']['selected']['seed'],chosen_Full_seed=frozen['shared']['selected']['seed'],
            model_freeze_sha256=sha(ORIGINAL/fold/'partial_count/models_frozen.json'))
    csvout(BASE/'source_validation.csv',details)
    write(BASE/'selection_frozen.json',dict(folds=selection,target_predictions_used=False,target_truth_used=False,
        source_validation_sha256=sha(BASE/'source_validation.csv'),protocol_sha256=sha(BASE/'protocol.json')))
    print(json.dumps({f:r['alpha'] for f,r in selection.items()}),flush=True)

def measures(pred,truth):
    n=len(truth);j=np.asarray(pred).argmax(1);p=np.asarray(pred)[np.arange(n),j]
    chosen=[r['robust'][int(k)] for r,k in zip(truth,j)]
    lower=np.array([r['lower'][int(k)] for r,k in zip(truth,j)])
    upper=np.array([r['upper'][int(k)] for r,k in zip(truth,j)])
    oracle=sum(any(x is True for x in r['robust']) for r in truth)
    return dict(states=n,oracle_B15=oracle,B15=sum(x is True for x in chosen),unresolved=sum(x is None for x in chosen),
        B15_lower=sum(x is True for x in chosen),B15_upper=sum(x is True or x is None for x in chosen),
        selected_Q_lower=float(lower.mean()),selected_Q_upper=float(upper.mean()),
        mean_selected_prediction=float(p.mean()),
        severe_FP_p90=int(np.sum((p>.9)&(upper<=.5))),
        severe_FP_p95=int(np.sum((p>.95)&(upper<=.5))),
        prediction_values=np.asarray(pred),selected=j,selected_robust=chosen)

def evaluate():
    assert (BASE/'selection_frozen.json').exists() and not (BASE/'final_decision.json').exists()
    frozen=read(BASE/'selection_frozen.json');protocol=read(BASE/'protocol.json')
    assert frozen['target_predictions_used'] is False and frozen['target_truth_used'] is False
    assert sha(BASE/'protocol.json')==frozen['protocol_sha256']
    assert all(sha(Path(p))==h for p,h in protocol['source_hashes'].items())
    pred=read(ORIGINAL/'target_predictions.json')['folds'];truth=read(ORIGINAL/'cached_truth.json')
    summary=[];paired=[];all_alpha=[]
    for fold in FOLDS:
        target=read(TARGETS/'targets'/fold/'manifest.json');t=truth[fold];pr=pred[fold]
        assert [v['state_uid'] for v in target]==pr['state_uids']==[v['state_uid'] for v in t]
        assert sha(TARGETS/'targets'/fold/'manifest.json')==pr['proposals_sha256']
        a=frozen['folds'][fold]['alpha'];eta=np.asarray(pr['scores']['partial_count_eta_only']);full=np.asarray(pr['scores']['partial_count_shared'])
        kinds={'eta_only':eta,'Full':full,'prior_constrained_Full':(1-a)*eta+a*full}
        for seed in SEEDS:
            pe=np.asarray(pr['scores'][f'partial_count_eta_only_seed{seed}']);pf=np.asarray(pr['scores'][f'partial_count_shared_seed{seed}'])
            kinds[f'paired_seed{seed}_eta_only']=pe;kinds[f'paired_seed{seed}_Full']=pf
            kinds[f'paired_seed{seed}_prior_constrained_Full']=(1-a)*pe+a*pf
        metrics={name:measures(v,t) for name,v in kinds.items()}
        for name,m in metrics.items():
            summary.append(dict(fold=fold,method=name,alpha=a,states=m['states'],oracle_B15=m['oracle_B15'],
                B15=m['B15'],unresolved=m['unresolved'],B15_upper=m['B15_upper'],
                selected_Q_lower=m['selected_Q_lower'],selected_Q_upper=m['selected_Q_upper'],
                severe_FP_p90=m['severe_FP_p90'],severe_FP_p95=m['severe_FP_p95']))
        for alpha in ALPHAS:
            m=measures((1-alpha)*eta+alpha*full,t)
            all_alpha.append(dict(fold=fold,alpha=alpha,B15=m['B15'],unresolved=m['unresolved'],selected_Q_lower=m['selected_Q_lower'],selected_Q_upper=m['selected_Q_upper'],
                eligible_for_selection=False if alpha!=a else True))
        for base in ('eta_only','Full'):
            x=metrics['prior_constrained_Full']['selected_robust'];y=metrics[base]['selected_robust']
            rescue=sum(xx is True and yy is False for xx,yy in zip(x,y));broken=sum(xx is False and yy is True for xx,yy in zip(x,y))
            lower=sum((xx is True)-(yy is True or yy is None) for xx,yy in zip(x,y));upper=sum((xx is True or xx is None)-(yy is True) for xx,yy in zip(x,y))
            paired.append(dict(fold=fold,comparison='prior_constrained_Full_vs_'+base,rescue=rescue,break_count=broken,
                net_lower=lower,net_upper=upper,unresolved_union=sum(xx is None or yy is None for xx,yy in zip(x,y))))
    csvout(BASE/'results.csv',summary);csvout(BASE/'paired.csv',paired);csvout(BASE/'alpha_sensitivity_descriptive.csv',all_alpha)
    selected=[r for r in summary if r['method'] in ('eta_only','Full','prior_constrained_Full')]
    result=dict(scope='Existing four-fold TT LOSO K16 pools only; target outcomes previously opened in historical studies',
        controller='TT; original independent controller UIDs preserved; not FF zero-shot evidence',
        source_only_alpha_selection=True,new_model_training=0,new_rollouts=0,alpha_by_fold={f:x['alpha'] for f,x in frozen['folds'].items()},
        selected_results=selected,paired=paired,
        alpha0_is_eta_only=True,alpha1_is_Full=True,
        interpretation='Selected alpha is source-only. An alpha0 fold preserves eta-only but cannot prove state/context transfer.',
        selection_frozen_sha256=sha(BASE/'selection_frozen.json'),target_predictions_sha256=sha(ORIGINAL/'target_predictions.json'),
        target_truth_sha256=sha(ORIGINAL/'cached_truth.json'))
    write(BASE/'final_decision.json',result);print(json.dumps(result,allow_nan=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('register','freeze','evaluate'));action=parser.parse_args().action
    globals()[action]()
