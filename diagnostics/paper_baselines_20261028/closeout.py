"""Close completed experiments; no rollouts. Directory becomes read-only afterwards."""
import json,csv,hashlib,subprocess
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def read(p):return json.load(open(p))
def rows(p):return list(map(json.loads,open(p)))
def fmt(x):return '—'if x is None else f'{x:.3f}'
def table(h,rr):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |'for r in rr])
def metrics(rr):
    ok=[r for r in rr if r['success']];n=len(rr);mean=lambda v:float(np.mean(v))if v else None;ci=binomtest(len(ok),n).proportion_ci(method='wilson')
    return dict(n=n,success=len(ok),success_mean_time=mean([r['completion_time']for r in ok]),truncated_mean_time=mean([r['truncated_time']for r in rr]),collision=sum(r['collision']for r in rr),timeout=sum(r['timeout']for r in rr),unknown=sum(r['numerical_failure']for r in rr),solver_nonoptimal_events=sum(r.get('solver_failure_count',0)for r in rr),solver_nonoptimal_episodes=sum(r.get('solver_failure_count',0)>0 for r in rr),mean_rollout_seconds=mean([r['elapsed_seconds']for r in rr]),wilson95=[ci.low,ci.high])
def main():
    b=read(HERE/'basis/summary.json');pr=rows(HERE/'paper/candidates.jsonl');pc=read(HERE/'paper/holdout_planned.json')['exact_configs'];ids={c['rollout_uid']for c in pc};hold=[r for r in pr if r['rollout_uid']in ids];assert len(hold)==48
    papers={m:metrics([r for r in hold if r['controller']==m])for m in sorted({r['controller']for r in hold})}
    legacy=rows(HERE/'legacy/candidates.jsonl');old=HERE.parent/'b5_oracle_20261027'
    prior=[r for r in rows(old/'legacy_candidates.jsonl')if r['method']in ('safety','B3','FF_B3')]+rows(old/'legacy_prior_candidates.jsonl')
    leg={m:metrics([r for r in legacy+prior if r['method']==m])for m in sorted({r['method']for r in legacy+prior})}
    summary=dict(basis=b,paper_holdout=papers,paper_by_scene={m:{s:metrics([r for r in hold if r['controller']==m and r['scene']==s])for s in sorted({r['scene']for r in hold})}for m in papers},legacy=leg,paper_selected=read(HERE/'paper/FINAL_FROZEN.json')['selected'],cross_domain=read(HERE/'CROSS_DOMAIN_FROZEN.json')['selected'])
    (HERE/'FINAL_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    labels={'dac_author_si':'Zhang T-RO 2025：DAC 作者核心','mgr_author_unicycle_tracking':'Lee IROS 2025：MGR 完整执行链适配','chen_impc_ob_si_tracking':'Chen RA-L 2023：IMPC-OB 作者核心','wang2017_full_di_tracking':'Wang 2017：完整消解的双积分器适配','wang2017_consistent_perturbation_di_tracking':'Wang 2017：Consistent Perturbation 适配','zhang2025_risk_si':'Zhang ICRA 2025：风险自适应逐公式实现'}
    pt=table(['论文方法（适配版）','成功/8','成功条件时间 s','全队列截断时间 s','碰撞','超时','unknown','内部非optimal事件/局'],[[labels[m],str(v['success'])+'/8',fmt(v['success_mean_time']),fmt(v['truncated_mean_time']),v['collision'],v['timeout'],v['unknown'],str(v['solver_nonoptimal_events'])+'/'+str(v['solver_nonoptimal_episodes'])]for m,v in papers.items()])
    lt=table(['旧四场景配置','成功/16','成功条件时间 s','截断时间 s','unknown'],[[m,str(v['success'])+'/16',fmt(v['success_mean_time']),fmt(v['truncated_mean_time']),v['unknown']]for m,v in leg.items()])
    fixed=table(['修正族','固定成功/16','固定成功时间 s','纯族 Oracle@32','Oracle首次成功时间 s'],[[','.join(e['columns']),str(e['fixed']['success'])+'/16',fmt(e['fixed']['success_mean_time']),e['oracle'][-1]['success'],fmt(e['oracle'][-1]['success_mean_time'])]for e in b['families']])
    text=f'''# C1 修正基筛选与论文基线：收尾报告

2026-10-28；分支 `research/paper-baselines-20261028`；前序提交 `daacc10`。本目录在收尾提交后冻结，只读复用；后续17类场景实验全部转入 `diagnostics/scene_types_basis_dimension_v1/`。

**没有得到可以直接替换生产原三列族的稳定固定策略。** 开发集选择的二维“朝目标＋局部净空梯度”固定策略在新测试为10/16，而历史冻结强三列为11/16；旧四checkpoint回测更降为TT 0/16、FF 1/16。把外部开发集和旧TT/FF的成功见证一起进行全18列支持集枚举后，最小已观察覆盖族为四维 **[G,P,R,C2]（朝目标、横向放大、互斥分离、局部净空梯度）**。它保留全部43个已知可覆盖开发案例的见证；这是有限记录的最小支持集，尚无新独立确认，不能称连续空间最优。新一轮将重新冻结场景和评价标准。

## 1. 完成范围与真实执行

统一固定配置筛选60组×22案例=1320条：888条精确复用，432条新运行。留出6族×32候选×16案例=3072条全运行，补强B3/安全参照32条。论文接口pilot共44条、开发参数筛选468条、独立测试48条；旧四checkpoint的新二维TT/FF测试32条。合计新运行4128条，另有精确复用888条。全部CPU，无训练，无GPU。旧生产实现、安全solver、checkpoint、动力学、成功判据均未修改，见 [FROZEN_INVARIANTS.json](FROZEN_INVARIANTS.json)。

源场景只有24状态，所以原计划的22–25索引在任何该批rollout前修正为未使用的22/23；future seed5，四场景×2物理初态×2原nominal=16例，仅8物理状态块。均有原始manifest、preflight、逐候选结果、轨迹、共享数据库journal与merger记录；所有已提交批次完成。timeout不称deadlock。

## 2. 修正族结果

{fixed}

历史冻结强三列：11/16，成功平均9.600s；安全过滤：10/16，成功平均10.190s。表中第一行原三列是共同中点的默认固定参数，不是强三列。四维[G,R,Y3,C2]固定12/16是测试观察，不能据此重新冻结并宣布已验证优胜。

Oracle定义为固定顺序前K项至少一个成功；K项全部实际运行，成功时间取顺序中第一个成功候选。纯族和含B3回退的联合池分别保存。联合@32为前16个B3＋前16个对应族，不能冒充纯一维/二维；预算相同的B3@32也保留。Oracle不是在线策略或连续参数最优。

逐族、逐场景、逐nominal、配对成功时间、全队列截断时间、失败/成本和描述性Wilson区间见 [basis/summary.json](basis/summary.json)，预算曲线见 [basis/budget_curves.csv](basis/budget_curves.csv)。与强B3配对的成功率差以8个物理状态块bootstrap，所有区间跨零；没有统计优势结论。

## 3. 旧四checkpoint转移

{lt}

只复用原Toy、Ring、Double-Bottleneck、Four-Way冻结运行时、真实checkpoint与seed16/17。新二维参数完全来自外部开发集；没有在旧四测试上调参。原G/P/R计算、TT投影和FF积分未改，运行中继续执行逐位零扩展断言。局部净空梯度使用该运行时的全部真实联合约束几何；Ring保留原已有48边形安全接缝。一个Four-Way TT案例触发原solver的 `identical_nonlinear_polish_failed`，单列unknown，没有换solver补救。

旧四反例说明：二维族在新分布式流水线中的Oracle覆盖不等于跨checkpoint部署能力。全域支持集搜索读取所有合格成功记录，排除时序策略、RHR、不同recipe和不等价BLEND。枚举262144个18列子集，得到四维[G,P,R,C2]覆盖外部开发22、旧TT16、旧FF5个已解决案例，见 [CROSS_DOMAIN_FROZEN.json](CROSS_DOMAIN_FROZEN.json)。未覆盖案例未被删掉，只是不属于“已有成功见证”目标集；不证明更低维不可能。

## 4. 论文基线

未对应论文的历史RHR-A/B、persistent/tuned RHR、Merry heuristic、partial-MGR已经退出本轮正式方法注册与比较；历史数据不删除。仅保留用户指定五篇论文，Wang的扰动和完整消解分列。

DAC、MGR、IMPC-OB优先调用作者完整核心。Wang为原双积分器barrier与三类消解逐公式实现，ICRA方法包含风险、旋转CLF及辅助CBF。作者commit、依赖版本见 [SOURCE_PINS.json](SOURCE_PINS.json)，逐方法适配与状态见 [PAPER_REGISTRY.json](PAPER_REGISTRY.json)。参数只在18个不同物理开发初态上选择，之后冻结，测试无重调。

{pt}

分母8是共同物理状态；这些论文策略不使用那两个原nominal，不复制成16次独立试验。我们的每个nominal也各8例，分组数据在summary。这里比较的是明确标注的共同环境适配版，**不冒充原论文实验的逐位复现或定理验证**。

共同环境保持原速度上限、碰撞与成功判据、horizon。论文方法使用自己的控制/安全核心，未套原B的安全solver来改变机制；DAC原生SI；MGR执行作者最终v/omega与heading-first端点，使用原SI步进承载该端点；IMPC-OB执行作者第一步预测位移，保留内部双积分器预测；Wang积分加速度到速度并施加共同速度约束。这些执行限制和有限步长不能继承原连续时间保证。

DAC有230次qpSWIFT非optimal返回（ExitFlag2，最大迭代次数），发生于7/8轨迹，包括作者内部锥判断QP；已保留原返回处理。未定位每次事件属于控制QP还是内部判断，因此这些环境成功标签不是所有内部求解都获证的声明。Chen有1次内部非optimal返回。Wang7/8和风险自适应3/8是数值/求解unknown，不能当普通任务失败排名；Wang默认pilot的3个失败状态用保存轨迹重算，去除共同速度约束后原加速度行仍被判PrimalInfeasible，见 [paper/failure_diagnosis.json](paper/failure_diagnosis.json)。它既不是Basis不可达证明，也不能归咎于额外速度球。

Wang全消解与纯扰动在这批独立结果相同，不能据此称两算法相同：有限轨迹没有建立Type1独立收益。ICRA公式未提供全部可直接复用的实验常数，本轮使用公开记录的预设参数网格，不能声称作者原参数。其正sigmoid乘不等式在精确算术下并不会彻底关闭约束，数值尺度值得进一步核验。

局部验证包括：风险辅助CBF解析导数与差分相对误差6.16e-10；Wang三种分类构型；MGR旋转矩形最近点/碰撞旋转等价误差6.29e-16。不能用这些局部检查替代完整闭环验证。IMPC-OB的Python3.9 GJK二进制以二维凸包GEOS距离替代，OMPL1.7保留ABITstar；作者壁钟终止保留，固定seed不保证位级复演，规划路径另存。没有引入另一规划器冒充。

## 5. 机制结论与局限

原三列族保留了旧checkpoint中的主要可用能力，固定删除P/R并不安全。前轮真实失败状态审计仍有效：局部净空梯度在6个十字口停滞恢复案例为6/6，而活跃约束切向投影为0/6（满秩切空间退化）；后者本轮新初态Oracle却16/16，说明预防和恢复不同。T1/P、T2/T3及角色列的互补、投影抵消和退化数据继续引用 [前轮报告](../basis_discovery_20261027/report.md)，未伪装成本轮重测。

五维没有不可替代的统一优势；当前有限库的全域覆盖保留四维[G,P,R,C2]即可，但没有冻结四维固定参数跨全域有效的证据。搜索、参数盒、时序和表达能力仍不能只凭Oracle失败完全分开；尚无“安全允许但整个连续候选族不可能”的数学证书。参考zip仍未提供，未声称运行其run_checks.py。原墙体CBF/物理间距不一致没有在旧轮修复，碰撞仍计失败。

## 6. 交付与冻结

`FINAL_SUMMARY.json` 为机器可读总表；`basis/`、`paper/`、`legacy/` 保留逐候选JSONL和原始轨迹；大文件不以Git提交冒充备份。所有代码/小型冻结配置/报告纳入本轮收尾提交。源码变更只在本目录。`COMPLETE.json` 表示已提交实验与交付文件收尾，不表示上述未验证科学假设成立。

后续工作全部写入新目录；本目录只读，不再追加实验。未解决的适配/统计/最小性问题列入 [GAPS.md](GAPS.md)。
'''
    (HERE/'report.md').write_text(text)
    registry=[]
    urls={'dac_author_si':'https://github.com/Parker-Zhang/Shaping-CLF-MCBF','mgr_author_unicycle_tracking':'https://github.com/Aiden-wjLee/merry-go-round','chen_impc_ob_si_tracking':'https://github.com/PKU-MACDLab/IMPC-OB','wang2017_full_di_tracking':'https://liwanggt.github.io/files/A2_Safe_Swarm.pdf','wang2017_consistent_perturbation_di_tracking':'https://liwanggt.github.io/files/A2_Safe_Swarm.pdf','zhang2025_risk_si':'https://arxiv.org/abs/2503.09621'}
    for m in papers:registry.append(dict(id=m,name=labels[m],source=urls[m],status='tested_full_mechanism_port_not_native_paper_reproduction',selected_options=summary['paper_selected'][m],formal_inclusion='Common-environment adapted comparison only; solver unknown remains separate.',details=next(r['controller_stats']['port']for r in hold if r['controller']==m)))
    (HERE/'PAPER_REGISTRY.json').write_text(json.dumps(dict(allowed=registry,retired=['RHR_A','RHR_B','persistent_RHR','tuned_RHR','Merry_heuristic','partial_MGR'],historical_evidence_retained=True),indent=2,ensure_ascii=False)+'\n')
    (HERE/'GAPS.md').write_text('''# 未解决事项（本轮关闭，不在旧目录继续实验）

- 四维跨域支持集为已有成功见证的最小覆盖；未新开独立确认，不是连续最小性证明。
- 所有论文方法均有明确共同环境适配；没有原论文原始实验的完整独立复现或保证移植。
- DAC内部非optimal事件尚未逐调用区分控制QP与锥判定QP；环境成功不等于所有子求解获证。
- Wang和ICRA有求解unknown；保留原始记录，不能作为方法能力不可能的结论。
- IMPC-OB壁钟规划存在同seed非位级复演风险，已保存路径；依赖与几何后端适配有记录。
- 留出只有8个物理状态块；不能据此声称统计稳健或Q16/B15认证。
- 参考C1_B5_validation_v0_20261027.zip缺失，未运行其run_checks.py。
- 旧轮墙体几何不一致保留；新17类场景按新规范在配置层一致化，不能回写旧结果。
- 早期失败pilot的逐次源码未单独版本封存，原manifest/hash/异常记录保留；正式开发及独立测试的源码与冻结hash已验证，早期pilot不进入性能结论。
''')
    (HERE/'COMPLETE.json').write_text(json.dumps(dict(status='closed_executed_scope',new_rollouts=4128,exact_reused=888,planned_batches_complete=True,scientific_limits='GAPS.md',next_directory='diagnostics/scene_types_basis_dimension_v1',read_only_after_closeout_commit=True),indent=2)+'\n')
    # Authoritative DB identity and labels are checked without importing old harness.
    import sys
    sys.path.insert(0,str(ROOT));from shared_rollout_db.src import rollout_db as db
    records=rows(HERE/'basis/candidates.jsonl')+pr+legacy
    with db.connect(True)as con:
        for r in records:
            x=con.execute('SELECT success,conflict_quarantined FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone();assert x and x['success']==r['success']and not x['conflict_quarantined'],r['rollout_uid']
    manifest={}
    for p in sorted(HERE.rglob('*')):
        if not p.is_file()or any(x in p.relative_to(HERE).parts for x in ('vendor','deps','deps_ompl17','__pycache__','traces','legacy_traces')):continue
        if p.name=='DELIVERY_MANIFEST.json':continue
        manifest[str(p.relative_to(HERE))]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (HERE/'DELIVERY_MANIFEST.json').write_text(json.dumps(dict(files=manifest,database_rows_checked=len(records),trace_counts={name:len(list((HERE/name/path).glob('*.npz')))for name,path in [('basis','traces'),('paper','traces'),('legacy','legacy_traces')]},parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()),indent=2)+'\n')
    print('Closed',len(records),'verified rows; report',HERE/'report.md')
if __name__=='__main__':main()
