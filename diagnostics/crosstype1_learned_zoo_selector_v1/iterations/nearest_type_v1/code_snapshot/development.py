"""Explicit development-only driver. Locked types are rejected at the boundary."""
from common import *
from selector import ARMS,BASELINE,Regression,leave_one_type,tune,crossfit_oracle,few_labels,paired_interval,paired_bounds
from features import NAMES
import argparse
import itertools

TYPES=('Flow','gaussian','deterministic','mixture','diffusion','chunking','recurrent','cvae')


def load(path):
    with np.load(path,allow_pickle=False) as source:
        data={k:source[k] for k in source.files}
    assert set(data['types']).issubset(TYPES),'Locked or unsupported types in development input'
    assert len(set(data['names']))==len(data['names'])
    assert list(data['arms'])==list(ARMS)
    assert list(data['feature_names'])==list(NAMES)
    assert list(data['state_uids'])==[s['state_uid'] for s in read(HERE/'EVAL_STATES.json')]
    assert data['labels'].shape==(len(data['names']),32,8,2)
    assert data['tuning'].shape==(len(data['names']),16,8)
    assert np.isin(data['labels'],[-1,0,1]).all(),'Missing labels (-2) cannot be treated as unknown (-1)'
    assert np.isin(data['tuning'],[-1,0,1]).all()
    assert data['features'].shape==(len(data['names']),len(NAMES))
    assert data['r5'].shape==(len(data['names']),160)
    return data


def take(labels,choice):
    nc,ns=labels.shape[:2]
    choice=np.asarray(choice)
    if choice.ndim==1:choice=np.broadcast_to(choice[:,None],(nc,ns))
    return labels[np.arange(nc)[:,None],np.arange(ns)[None,:],choice]


def equal_mean(values,types):
    per=values.mean(tuple(range(1,values.ndim)))
    return float(np.mean([per[types==t].mean() for t in sorted(set(types))]))


def run(path,iteration):
    prereg()
    data=load(path);folder=HERE/'iterations'/iteration
    assert not folder.exists(),'Keep all development iterations immutable'
    folder.mkdir(parents=True)
    labels=data['labels'];lo=(labels==1).astype(float);hi=(labels!=0).astype(float)
    rates=lo.mean((1,3));types=data['types'];nc=len(types)
    choices,folds=leave_one_type(data['features'],rates,types)
    r5choice,r5folds=leave_one_type(data['r5'],rates,types,pca_dimensions=2)
    method_choices={f'all_{a}':np.full(nc,j) for j,a in enumerate(ARMS)}
    method_choices.update(pipeline=choices,r5_pca2_pipeline=r5choice)
    for k in (2,4,8):method_choices[f'few_labels_{k}']=few_labels(data['tuning']==1,k)
    oracle,oracle_choice=crossfit_oracle(lo)
    method_choices['crossfit_oracle']=oracle_choice
    method_choices['in_sample_oracle']=np.argmax(rates,axis=1)
    typebest=np.empty(nc,int)
    for t in sorted(set(types)):
        mask=types==t;typebest[mask]=int(np.argmax(rates[mask].mean(0)))
    method_choices['type_best_fixed_oracle']=typebest
    values={name:(take(lo,c),take(hi,c)) for name,c in method_choices.items()}
    baseline=values['all_S500'][0];selected=values['pipeline'][0]
    summaries={}
    for group in [*sorted(set(types)),'development_combined']:
        mask=np.ones(nc,bool) if group=='development_combined' else types==group
        summaries[str(group)]={}
        for name,(low,high) in values.items():
            chosen=np.asarray(method_choices[name])
            if chosen.ndim==1:chosen=np.broadcast_to(chosen[:,None],(nc,32))
            same_schedule=(chosen==BASELINE)[...,None]
            same_safety=(chosen==0)[...,None]
            summaries[str(group)][name]=dict(
                lower_success_pct=100*equal_mean(low[mask],types[mask]),
                upper_success_pct=100*equal_mean(high[mask],types[mask]),
                paired_vs_schedule=paired_bounds(low[mask],high[mask],baseline[mask],values['all_S500'][1][mask],types[mask],same_cases=same_schedule[mask]),
                controller_equal_paired_vs_schedule=paired_bounds(low[mask],high[mask],baseline[mask],values['all_S500'][1][mask],types[mask],same_cases=same_schedule[mask],type_equal=False),
                crossfit_regret_pp=100*equal_mean((oracle-low)[mask],types[mask]),
                mean_controller_regret_pp=float((oracle-low)[mask].mean()*100),
                max_controller_regret_pp=float((oracle-low)[mask].mean((1,2)).max()*100),
                harm_count=int(np.sum(np.where(same_safety,0.,values['all_safety_baseline'][1]-low)[mask].mean((1,2))>=.1-1e-12)),
                harm_certain=int(np.sum(np.where(same_safety,0.,values['all_safety_baseline'][0]-high)[mask].mean((1,2))>=.1-1e-12)))
    combined=summaries['development_combined']['pipeline']
    pertype=[summaries[str(t)]['pipeline']['paired_vs_schedule']['point_pp'] for t in sorted(set(types))]
    complete=set(types)==set(TYPES) and all(np.sum(types==t)>=4 for t in TYPES)
    target=dict(complete_eight_types=complete,
       superiority=combined['paired_vs_schedule']['CI95_pp'][0]>0,
       regret_at_most_2pp=combined['crossfit_regret_pp']<=2,
       no_type_worse_than_3pp=min(pertype)>=-3,
       zero_harm=combined['harm_count']==0)
    target['passed']=all(target.values())
    write(folder/'OUTER_FOLDS.json',folds);write(folder/'R5_OUTER_FOLDS.json',r5folds)
    result=dict(utc=utc(),evidence_level='DEVELOPMENT',iteration=iteration,input_path=str(path),input_sha256=sha(path),
                development_targets=target,summaries=summaries,
                limitations=['Frozen round38 comparator and event table must be joined by the evidence assembler before final delivery.',
                            'Oracle CI requires selection recomputation inside the bootstrap; paired intervals here condition on outer choices.'])
    write(folder/'RESULTS.json',result)
    np.savez_compressed(folder/'CHOICES.npz',names=data['names'],types=types,**method_choices)
    seal(str(folder.relative_to(HERE)/'ITERATION.receipt.json'),[path,folder/'RESULTS.json',folder/'OUTER_FOLDS.json',
         folder/'R5_OUTER_FOLDS.json',folder/'CHOICES.npz',HERE/'selector.py',HERE/'features.py',HERE/'development.py'])
    with (HERE/'ITERATION_LOG.md').open('a') as stream:
        stream.write(f'\n## {iteration}\n\n机制特征按臂相对成功率差岭回归；源类型内嵌套留一类型选择正则。\n'
                     f"类型等权增益 {combined['paired_vs_schedule']['point_pp']:.3f}pp，"
                     f"95% CI {combined['paired_vs_schedule']['CI95_pp']}；交叉拟合oracle遗憾 "
                     f"{combined['crossfit_regret_pp']:.3f}pp；目标通过 {target['passed']}。全部折和候选参数保留于 iterations/{iteration}/。\n")
    print(json.dumps(target),flush=True)


def scale(path,penalty):
    data=load(path);types=data['types'];rates=(data['labels']==1).mean((1,3));rng=np.random.default_rng(202610180701)
    if set(types)!=set(TYPES):raise ValueError('Eight complete development types required for the specified m=7 curve')
    rows=[]
    for held in TYPES:
        source=[t for t in TYPES if t!=held];va=types==held
        for m in (1,2,3,5,7):
            for repetition in range(1 if m==7 else 20):
                selected=sorted(rng.choice(source,size=m,replace=False).tolist());tr=np.isin(types,selected)
                fold_penalty,inner=tune(data['features'][tr],rates[tr],types[tr])
                model=Regression(fold_penalty).fit(data['features'][tr],rates[tr],types[tr])
                choice=model.choose(data['features'][va]);value=rates[va][np.arange(va.sum()),choice]
                oracle=crossfit_oracle((data['labels'][va]==1).astype(float))[0].mean((1,2))
                rows.append(dict(held_type=held,m=m,repetition=repetition,source_types=selected,
                  source_controllers=int(tr.sum()),source_only_penalty=fold_penalty,inner=inner,held_names=data['names'][va].tolist(),
                  regret_pp=((oracle-value)*100).tolist()))
    write(HERE/'SCALE_CURVE_RAW.json',dict(seed=202610180701,hyperparameters='nested within selected source types only',rows=rows,
         same_type_control_status='PENDING matched-count feasibility audit; do not duplicate instances to fake extra data'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run','scale']);parser.add_argument('bundle')
    parser.add_argument('--iteration',default='ridge_v1');parser.add_argument('--penalty',type=float,default=1.)
    args=parser.parse_args()
    if args.command=='run':run(Path(args.bundle),args.iteration)
    else:scale(Path(args.bundle),args.penalty)
