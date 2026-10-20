"""Whole-round upper-cap time estimate using historical records, no new rollouts."""
from common import *
from records import actual
from collections import Counter

def main():
    base=read(HERE/'BASE_PROTOCOL.json');counts=Counter(base['types'][n] for n in base['controller_order']);new={'transformer_bc','gradient_mpc','conflict_based_search'};times={t:[] for t in counts}
    sources=[D/'crosstype3_unseen_controller_schedule_v1/training_CASES.json',D/'crosstype3_unseen_controller_schedule_v1/new_evaluation_CASES.json']
    with rollout_db.connect(True) as db:
        for path in sources:
            for c in read(path):
                t=c['family']
                if t not in times or len(times[t])>=20:continue
                row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
                if row is None:continue
                e=actual(dict(row),verify=False)
                if e is not None and e['record'].get('elapsed_seconds') is not None:times[t].append(e['record']['elapsed_seconds'])
    assert all(times.values()),{t:len(x) for t,x in times.items()};train={t:n*48*(34 if t in new else 8) for t,n in counts.items()};enames=read(HERE/'OPENING_CONFIG.json')['B_evaluation'];evaluation={t:n*24*2*8 for t,n in Counter(base['types'][n] for n in enames).items()};A=read(HERE/'A_RUNTIME_ESTIMATE.json');Btrain=sum(train[t]*np.mean(times[t]) for t in train)/18/3600;Beval=sum(evaluation[t]*np.mean(times[t]) for t in evaluation)/18/3600
    hist=[]
    for mode in ('full','no_controller'):
        paths=sorted((D/'crosstype3_unseen_controller_schedule_v1/critic_fits'/mode).glob('*/HISTORY.json'))[:8]
        for p in paths:hist.append(read(p)[-1]['elapsed_seconds'])
    assert hist;GPU_mean=float(np.mean(hist))*382/2/3600*(199104/129792);raw_wall=A['ideal_wall_hours_at18']+Btrain+Beval+GPU_mean;range_hours=[raw_wall*1.15,raw_wall*2.0]
    record=dict(utc=utc(),estimated_total_wall_hours_before_extra_cache_dedup=range_hours,A_ideal_hours=A['ideal_wall_hours_at18'],B_training_ideal_hours_at18=Btrain,B_evaluation_ideal_hours_at18=Beval,critic_ideal_hours_at2GPU=GPU_mean,critic_historical_fit_samples=len(hist),critic_seconds_per_historical_three_seed_fit_mean=float(np.mean(hist)),critic_scale_for_larger_label_tensor=199104/129792,B_training_raw_caps=train,B_evaluation_max_caps=evaluation,historical_type_seconds={t:dict(n=len(x),mean=float(np.mean(x)),p90=float(np.quantile(x,.9))) for t,x in times.items()},caveats=['B cap before exact cache/local early reuse; actual work may be smaller','Wall estimate includes generic compilation/IO/analysis allowance; not deadline','A then B labels then critic then fresh evaluation; resource competition can extend waiting'])
    write(HERE/'TOTAL_RUNTIME_ESTIMATE.json',record);(HERE/'OPENING_TOTAL_RUNTIME_SUPPLEMENT.md').write_text('# 全轮耗时补充记录\n\n按跨类型3既有真实rollout与critic拟合耗时估计，18 CPU / 2 GPU连续可用时，全部任务保守规划约 %.1f–%.1f 小时。A约 %.1f 小时、B训练标签约 %.1f 小时、B新状态评估约 %.1f 小时（均为理想18 CPU值），critic约 %.1f 小时（2 GPU，按标签张量扩大缩放）。\n\nB部分以复用前上限估计；精确缓存和提前结束复用可减少工作，资源等待会延长。此补充只估计工程耗时，不修改任何候选、抽样、模型、结论规则或预算；原开工收据保持不变。类型耗时与样本数见 TOTAL_RUNTIME_ESTIMATE.json。\n' % (*range_hours,A['ideal_wall_hours_at18'],Btrain,Beval,GPU_mean))
    seal('OPENING_TOTAL_RUNTIME_SUPPLEMENT.receipt.json',[HERE/n for n in ('TOTAL_RUNTIME_ESTIMATE.json','OPENING_TOTAL_RUNTIME_SUPPLEMENT.md','total_runtime_estimate.py','OPENING_RECORD.receipt.json')],before_B_training_rollouts=True,no_scientific_protocol_change=True)
    print(json.dumps({k:v for k,v in record.items() if k.endswith('hours') or 'estimated_total' in k}),flush=True)
if __name__=='__main__':main()
