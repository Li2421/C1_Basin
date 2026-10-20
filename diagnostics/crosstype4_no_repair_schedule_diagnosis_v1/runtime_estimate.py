"""Budget estimate from historical actual execution time, no timing rollouts."""
from common import *
from records import actual
from collections import Counter

def main():
    todo=read(HERE/'A_EXECUTION_CASES.json');counts=Counter(c['family'] for c in todo);rows=read(HERE/'A_CASES.json');samples={t:[] for t in counts};seen=set()
    with rollout_db.connect(True) as db:
        for c in rows:
            t=c['family']
            if len(samples[t])>=40 or c['rollout_uid'] in seen:continue
            seen.add(c['rollout_uid']);r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            if r is None:continue
            e=actual(dict(r),verify=False)
            if e is not None and e['record'].get('elapsed_seconds') is not None:samples[t].append(float(e['record']['elapsed_seconds']))
    estimates={}
    for t,n in counts.items():
        x=samples[t];assert x,t;estimates[t]=dict(actual_execution_cap=n,historical_samples=len(x),mean_seconds=float(np.mean(x)),median_seconds=float(np.median(x)),p90_seconds=float(np.quantile(x,.9)),source='historical actual records, elapsed loop time; compilation/IO overhead additional')
    cpu=sum(v['actual_execution_cap']*v['mean_seconds'] for v in estimates.values());p90=sum(v['actual_execution_cap']*v['p90_seconds'] for v in estimates.values());write(HERE/'A_RUNTIME_ESTIMATE.json',dict(utc=utc(),types=estimates,CPU_hours_at_historical_mean=cpu/3600,ideal_wall_hours_at18=cpu/18/3600,planning_wall_hours_range=[cpu/18/3600*1.15,p90/18/3600*1.5],not_completion_guarantee=True,slower_MPC_and_planning_explicit=True))
    (HERE/'OPENING_RECORD.md').write_text('# 跨类型4开工记录\n\n全部为开发证据。已完成历史复用核查与开始审计；16条复现、8条执行等价性、32条提前结束验证均通过，后者覆盖全部19类型。A对象/抽样/种子/结论规则见 OPENING_CONFIG.receipt.json 和 PREREGISTRATION.md；B候选与34动作抽样见 CANDIDATES.receipt.json。\n\nA共41,962个唯一物理请求，DB已有13,718个；缺28,244个，已知提前结束等价复用22个，真实补跑上限28,222个。运行中新增的真实完整早期η轨迹若满足已验证规则，可进一步减少补跑；推断标签不写DB。\n\n按历史实际执行时间，18 CPU理想耗时 %.2f 小时，预留编译和I/O后的规划范围 %.2f–%.2f 小时。此为A部分估计；B训练标签原始增量69,312条，精确扣除缓存/等价后再封存；B新评估最多28,416请求，须训练冻结后再提交。当前不能把A估计当作全部任务完成时间。\n\n资源按当前prompt全用户合计18 CPU/2 GPU；每次提交检查squeue。B闭环标签等待A诊断完成，后续19折训练和新状态评估最后执行。所有工程修正记录于ITERATION_LOG.md；不做Git写操作。\n' % (cpu/18/3600,cpu/18/3600*1.15,p90/18/3600*1.5))
    seal('OPENING_RECORD.receipt.json',[HERE/n for n in ('OPENING_RECORD.md','A_RUNTIME_ESTIMATE.json','A_BUDGET.json','OPENING_CONFIG.receipt.json','CANDIDATES.receipt.json','EXECUTION_REPRODUCTION.receipt.json','EXECUTION_EQUIVALENCE.receipt.json','EARLY_TERMINATION_EQUIVALENCE.receipt.json','A_EXECUTION_FROZEN_v2.receipt.json','runtime_estimate.py')],before_first_A_submission=True)
    print(json.dumps(read(HERE/'A_RUNTIME_ESTIMATE.json'),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
