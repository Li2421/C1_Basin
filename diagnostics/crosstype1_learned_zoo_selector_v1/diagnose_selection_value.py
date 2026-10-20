"""Separate better fixed repair from evidence for controller-specific selection."""
from common import *
from selector import ARMS,best_arm,crossfit_oracle
from scale_complete import paired_gain
from collections import Counter


def main():
    target=HERE/'DEVELOPMENT_SELECTION_VALUE.json';assert not target.exists()
    with np.load(HERE/'DEVELOPMENT_BUNDLE.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    types=data['types'];rates=(data['labels']==1).mean((1,3));n=len(types)
    fixed=np.zeros(n,int);folds=[]
    for held in sorted(set(types)):
        source=[t for t in sorted(set(types)) if t!=held]
        source_rate=np.stack([rates[types==t].mean(0) for t in source]).mean(0)
        arm=int(best_arm(source_rate));fixed[types==held]=arm
        folds.append(dict(held_type=str(held),source_types=source,arm=ARMS[arm],source_type_equal_success=source_rate))
    equal=lambda values:float(np.mean([values[types==t].mean() for t in sorted(set(types))]))
    uniform=int(best_arm(np.stack([rates[types==t].mean(0) for t in sorted(set(types))]).mean(0)))
    available=np.isfinite(data['features']).any(1);rows=[];cache={};paths=[]
    for folder in sorted((HERE/'iterations').iterdir()):
        receipt=folder/'ITERATION.receipt.json'
        if not receipt.exists():continue
        result=read(folder/'RESULTS.json')
        with np.load(folder/'CHOICES.npz') as z:choice=z['pipeline']
        key=choice.tobytes()
        if key not in cache:cache[key]=paired_gain(data,choice[:,None],fixed[:,None])
        rows.append(dict(iteration=folder.name,strategy=result['strategy'],
            choices={ARMS[int(k)]:int(v) for k,v in Counter(choice.tolist()).items()},
            unique_choices_on_available_probes=sorted({ARMS[int(v)] for v in choice[available]}),
            constant_on_available_probes=bool(len(set(choice[available]))==1),
            paired_vs_source_only_best_fixed=cache[key],development_targets=result['development_targets']))
        paths.extend([receipt,folder/'CHOICES.npz'])
    cross,_=crossfit_oracle((data['labels']==1).astype(float))
    record=dict(utc=utc(),evidence_level='DEVELOPMENT',source_only_best_fixed_folds=folds,iterations=rows,
        best_uniform_arm=ARMS[uniform],best_uniform_type_equal_success_pct=100*equal(rates[:,uniform]),
        in_sample_controller_oracle_type_equal_success_pct=100*equal(rates.max(1)),
        observed_in_sample_adaptation_headroom_pp=100*equal(rates.max(1)-rates[:,uniform]),
        crossfit_oracle_type_equal_success_pct=100*equal(cross.mean((1,2))),
        interpretation='Improvement over S500 alone does not establish behavior-adaptive selection; compare with source-only best fixed repair. The in-sample headroom is descriptive on this sample, not a population impossibility bound.',
        locked_outcomes_read=False,new_rollouts=0)
    write(target,record);seal('DEVELOPMENT_SELECTION_VALUE.receipt.json',[target,HERE/'DEVELOPMENT_BUNDLE.npz',HERE/'diagnose_selection_value.py',HERE/'scale_complete.py',*paths])
    print(json.dumps(dict(best_uniform_arm=record['best_uniform_arm'],headroom_pp=record['observed_in_sample_adaptation_headroom_pp'],iterations=rows),indent=2))


if __name__=='__main__':main()
