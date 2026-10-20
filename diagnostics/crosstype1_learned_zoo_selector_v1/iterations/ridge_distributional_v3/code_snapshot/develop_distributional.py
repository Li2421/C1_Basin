"""Complete development statistics; locked inputs rejected before label access."""
from common import *
from development import load,TYPES
from selector import ARMS,BASELINE,leave_one_type,few_labels,best_arm
from selector_candidates_v2 import leave_one_type_candidate
from analysis_statistics import evaluate
from transition_summary import summarize
import csv


def run(bundle,iteration,strategy='ridge'):
    prereg();data=load(bundle)
    packet=HERE/'DEVELOPMENT_DISTRIBUTIONAL_FEATURES.npz'
    packet_receipt=HERE/'DEVELOPMENT_DISTRIBUTIONAL_FEATURES.receipt.json'
    for path,digest in read(packet_receipt)['files'].items():assert sha(path)==digest
    with np.load(packet,allow_pickle=False) as z:
        for key in ('names','types','r5'):assert np.array_equal(data[key],z[key],equal_nan=True) if key=='r5' else np.array_equal(data[key],z[key])
        assert np.array_equal(data['features'],z['features'][:,:data['features'].shape[1]],equal_nan=True)
        data['features']=z['features'];data['feature_names']=z['feature_names']
    reference={'ridge':'ridge_v1','armwise_tree':'armwise_tree_v2','type_kernel':'type_kernel_v2'}[strategy]
    r5source=HERE/'iterations'/reference
    for path,digest in read(r5source/'ITERATION.receipt.json')['files'].items():assert sha(path)==digest
    prior=read(r5source/'RESULTS.json');assert prior['input_sha256']==sha(bundle) and prior['strategy']==strategy
    with np.load(r5source/'CHOICES.npz',allow_pickle=False) as z:
        assert np.array_equal(z['names'],data['names']);r5choice=z['r5_pca2_pipeline']
    r5folds=read(r5source/'R5_OUTER_FOLDS.json')
    feature_metadata=dict(feature_schema='distributional_v1',feature_packet_path=str(packet),
        feature_packet_sha256=sha(packet),r5_reused_from=reference)
    folder=HERE/'iterations'/iteration
    assert not folder.exists();folder.mkdir(parents=True)
    snapshot=folder/'code_snapshot';snapshot.mkdir()
    sources=('develop_distributional.py','development.py','selector.py','selector_candidates.py','selector_candidates_v2.py','features.py',
             'r5.py','analysis_statistics.py','transition_summary.py','distributional_features.py')
    for name in sources:(snapshot/name).write_bytes((HERE/name).read_bytes())
    write(folder/'RUN_STARTED.json',dict(utc=utc(),strategy=strategy,bundle_sha256=sha(bundle),
        sources={name:sha(snapshot/name) for name in sources},locked_outcomes_read=False,**feature_metadata))
    labels=data['labels'];rates=(labels==1).mean((1,3));types=data['types'];nc=len(types)
    if strategy=='ridge':
        choice,folds=leave_one_type(data['features'],rates,types)
    else:
        choice,folds=leave_one_type_candidate(data['features'],rates,types,strategy)
    blind_receipt=read(HERE/'DEVELOPMENT_COMPARATOR_CHOICES.receipt.json')
    for path,digest in blind_receipt['files'].items():assert sha(path)==digest
    comparators=read(HERE/'DEVELOPMENT_COMPARATOR_CHOICES.json')['methods']
    fixed={f'all_{a}':np.full(nc,j,int) for j,a in enumerate(ARMS)}
    fixed.update(pipeline=choice,r5_pca2_pipeline=r5choice)
    for method,record in comparators.items():
        fixed[method]=np.array([ARMS.index(record['choices'][str(n)]) if record['choices'][str(n)] in ARMS else -1 for n in data['names']])
    for k in (2,4,8):fixed[f'few_labels_{k}']=few_labels(data['tuning']==1,k)
    write(folder/'OUTER_FOLDS.json',folds);write(folder/'R5_OUTER_FOLDS.json',r5folds)
    np.savez_compressed(folder/'CHOICES.npz',names=data['names'],types=types,**fixed)
    summaries=evaluate(data,fixed)
    result=summaries['combined']['type_equal']['methods']['pipeline']
    differences=[summaries[str(t)]['type_equal']['methods']['pipeline']['gain_vs_schedule_pp'] for t in sorted(set(types))]
    targets=dict(complete_eight_types=set(types)==set(TYPES) and all(np.sum(types==t)>=4 for t in TYPES),
        superiority=result['gain_CI95_pp'][0]>0,regret_at_most_2pp=result['crossfit_regret_pp']<=2,
        no_type_worse_than_3pp=min(differences)>=-3,zero_harm=result['harm_possible']==0)
    targets['passed']=all(targets.values())
    write(folder/'RESULTS.json',dict(utc=utc(),iteration=iteration,strategy=strategy,evidence_level='DEVELOPMENT',
          input_path=str(bundle),input_sha256=sha(bundle),development_targets=targets,summaries=summaries,
          comparator_gaps={k:v for k,v in comparators.items() if not v['available_for_complete_cohort']},
          limitations=[],
          oracle_reselection_in_bootstrap=True,replicates=5000,**feature_metadata))
    rows=[]
    for group,aggregations in summaries.items():
        for aggregation,record in aggregations.items():
            for method,item in record['methods'].items():
                rows.append(dict(type=group,aggregation=aggregation,method=method,**item,evidence_level='DEVELOPMENT'))
    with (folder/'METHOD_TABLE.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    from selector import crossfit_oracle
    cross,crosschoices=crossfit_oracle((labels==1).astype(float))
    selected=labels[np.arange(nc)[:,None],np.arange(32)[None,:],choice[:,None]]==1
    details=[dict(controller=str(name),type=str(t),pipeline_arm=ARMS[int(a)],
             in_sample_best_arm=ARMS[int(best_arm(rates[i]))],success_pct=float(selected[i].mean()*100),
             crossfit_regret_pp=float((cross[i]-selected[i]).mean()*100),
             in_sample_regret_pp=float((rates[i].max()-selected[i].mean())*100),
             crossfit_choices=[ARMS[int(v)] for v in crosschoices[i]])
             for i,(name,t,a) in enumerate(zip(data['names'],types,choice))]
    write(folder/'CONTROLLER_DETAILS.json',details)
    transition_rows=[]
    for i,name in enumerate(data['names']):
        for s in range(32):
            for q in range(2):
                before='unknown' if data['nominal_unknown'][i,s,q] else 'collision' if data['nominal_collision'][i,s,q] else 'success' if data['nominal'][i,s,q]==1 else 'timeout' if data['nominal_timeout'][i,s,q] else 'other'
                after='unknown' if data['unknown'][i,s,0,q] else 'collision' if data['collision'][i,s,0,q] else 'success' if labels[i,s,0,q]==1 else 'timeout' if data['timeout'][i,s,0,q] else 'other'
                transition_rows.append(dict(controller=str(name),type=str(types[i]),state_uid=str(data['state_uids'][s]),seed=q,nominal_outcome=before,safety_outcome=after))
    with (folder/'NOMINAL_TO_SAFETY_TRANSITIONS.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=transition_rows[0].keys());writer.writeheader();writer.writerows(transition_rows)
    write(folder/'NOMINAL_TO_SAFETY_SUMMARY.json',summarize(transition_rows))
    seal(str(folder.relative_to(HERE)/'ITERATION.receipt.json'),[bundle,packet,packet_receipt,HERE/'distributional_features.py',r5source/'ITERATION.receipt.json',r5source/'CHOICES.npz',r5source/'R5_OUTER_FOLDS.json',folder/'RESULTS.json',folder/'OUTER_FOLDS.json',
        folder/'R5_OUTER_FOLDS.json',folder/'CHOICES.npz',folder/'METHOD_TABLE.csv',folder/'CONTROLLER_DETAILS.json',
        folder/'NOMINAL_TO_SAFETY_TRANSITIONS.csv',folder/'NOMINAL_TO_SAFETY_SUMMARY.json',HERE/'transition_summary.py',
        HERE/'develop_distributional.py',HERE/'analysis_statistics.py',HERE/'selector.py',HERE/'selector_candidates.py',HERE/'selector_candidates_v2.py',HERE/'features.py',
        folder/'RUN_STARTED.json',*[snapshot/name for name in sources]])
    with (HERE/'ITERATION_LOG.md').open('a') as out:
        out.write(f'\n## {iteration}\n\n探测分布特征（105维）选择结构 {strategy}，超参数在源类型内部嵌套选择；所有预处理仅拟合源类型，R5-PCA2对照使用同结构并复用相同数据/模型的既有预测。'
          f"类型等权增益{result['gain_vs_schedule_pp']:.3f}pp，95% CI {result['gain_CI95_pp']}；"
          f"交叉拟合oracle遗憾{result['crossfit_regret_pp']:.3f}pp；可能伤害数{result['harm_possible']}；目标通过{targets['passed']}。"
          'Oracle在5000次配对成簇bootstrap内部重选，保留原state奇偶；同时报告unknown上下界及controller等权汇总。\n')
    print(json.dumps(targets),flush=True)


if __name__=='__main__':run(Path(sys.argv[1]),sys.argv[2],sys.argv[3] if len(sys.argv)>3 else 'ridge')
