"""Independent raw-record reproduction and readable frozen-analysis delivery."""
from analyze import *
from collections import Counter

def main():
    r=read(HERE/'results.json');z=np.load(R10/'LABELS.npz');names=z['controllers'].tolist();arms=z['arms'].tolist();raw_outcomes=Counter();cases=read(R10/'main_CASES.json');raw_hashes=[];step_rows=[]
    exception_names=list(r['exception_failure_summaries']);detail_names=list(dict.fromkeys(exception_names+['base_u_v3_macflow']))
    grouped={}
    for c in cases:
        p=R10/'main/raw'/f"{c['rollout_uid']}.json";raw=read(p);assert raw['case']==c;i=names.index(c['base_controller_name']);a=arms.index(c['arm']);s=c['state_index'];seed=c['seed'];assert raw['success']==z['success'][i,a,s,seed];assert raw['numerical_failure']==z['unknown'][i,a,s,seed];raw_outcomes[raw['outcome']]+=1;raw_hashes.append(dict(rollout_uid=c['rollout_uid'],sha256=sha(p)))
        if names[i] in detail_names:
            o=raw['outcome'];steps=raw['error']['step'] if raw['error'] else raw['summary'].get('first_success_step') or raw['episode_length'];grouped.setdefault((names[i],arms[a],o),[]).append(steps)
            if o!='success':step_rows.append(dict(controller=names[i],arm=arms[a],state_index=s,seed=seed,rollout_uid=c['rollout_uid'],outcome=o,occurrence_step=steps,error_type=raw['error'].get('type') if raw['error'] else None))
    original=read(R10/'ENDPOINT_OPENING.receipt.json');original_hashes={x['rollout_uid']:x['sha256'] for x in original['raw_inventory']};assert all(original_hashes[x['rollout_uid']]==x['sha256'] for x in raw_hashes)
    write('RAW_REPRODUCTION_AUDIT.json',dict(status='PASS',records=len(cases),outcomes=raw_outcomes,label_success_and_unknown_exact=True,all_raw_hashes_match_frozen_endpoint_receipt=True,endpoint_receipt_sha256=sha(R10/'ENDPOINT_OPENING.receipt.json')))
    summary=[]
    for (n,arm,o),steps in grouped.items():summary.append(dict(controller=n,arm=arm,outcome=o,count=len(steps),step_min=min(steps),step_median=float(np.median(steps)),step_max=max(steps),step_mean=float(np.mean(steps))))
    csvwrite('EXCEPTION_OUTCOME_STEPS.csv',summary);csvwrite('EXCEPTION_FAILURE_RECORDS.csv',step_rows)
    wide=[]
    for n in names:
        row={'controller':n}
        for key in ['E_S500','L_S500','N']:
            m=r['effects'][n][key];row[key+'_pp']=100*m['point'];row[key+'_CI95_lower_pp']=100*m['CI95_point'][0];row[key+'_CI95_upper_pp']=100*m['CI95_point'][1]
        wide.append(row)
    csvwrite('E_L_N_MAIN.csv',wide)
    sensitivity={}
    for schedule in ['S300','S400','S500']:
        e=np.array([r['effects'][n]['E_'+schedule]['point'] for n in names]);l=np.array([r['effects'][n]['L_'+schedule]['point'] for n in names]);nval=np.array([r['effects'][n]['N']['point'] for n in names])
        sensitivity[schedule]=dict(mean_E=float(e.mean()),mean_L=float(l.mean()),E_positive=int((e>0).sum()),L_positive=int((l>0).sum()),negative_N_with_E_ge10pp=int(((nval<0)&(e>=.1)).sum()),both_ge10pp=int(((e>=.1)&(l>=.1)).sum()),negative_E_controllers=[names[i] for i in range(14) if e[i]<0])
    r['sensitivity_summary']=sensitivity;write('results.json',r)
    p1=r['P1'];p2=r['P2'];p3=r['P3'];p4=r['P4'];fixed=p3['fixed'];expanded=p3['expanded']
    fmt=lambda x:f'{x*100:.2f}'
    lines=["H_compress **SUPPORTED（DEVELOPMENT）**：P1、P2 均满足预注册阈值。分解形式下，现有 R5 描述对前期效应 E 展现较强的留一预测能力（Spearman 0.842，符号准确率13/14），是可学习性的局部正面证据；后期效应 L 的预测能力尚未确立，不能据此宣布完整 controller–Q 关系已可学习。两个描述的1-NN均未降低动作选择遗憾。",'',"所有解释与结论均为 **DEVELOPMENT**。本次只分析已有结果，未新增 rollout、未写共享 DB，未改 round10 或历史资产。用户明确允许以数值完成启动本次纯分析；round10 的 NUMERICAL_COMPLETE_VIDEO_INCOMPLETE 状态保留。本分析完成不代表 round10 视频验收完成。",'',"## 冻结定义、证据与复用",'',"设计者 addendum 的 SHA256 为 `2ffa9e43d2dd94a1726fa5f5e0bb0b52d59b93af8bb88cc2214926770baacd9e`，receipt UTC 为2026-10-13T09:18:52+00:00，早于 round10 结果开放2026-10-13T09:59:37.263707+00:00。执行代理此前参与 round10，已接触过其结果；自己的预注册如实披露这点，不能将本执行登记称为代理盲态预注册。定义与预测来自独立设计者的冻结文件，本次未按结果改变。",'',"复用 round10 LABELS、18688条原始记录、64新 state、seed0–3、14真实 Flow、73个可用controller–arm组合，unknown131条全部保留。独立逐条核对 success、unknown 和原始记录哈希，与冻结 endpoint 收据完全一致。Freg仅三controller可用，其他位置不进入oracle。复用 round7 FEATURES：R3用共同96-state均值（沿用stage1），R5统一删105–109列保留155维；没有新短探测、Flow训练或checkpoint推理。新增证据仅为这些冻结结果的分析和源controller上的ridge拟合。",'',"E=S500−F10，L=S500−F13，N=F13−F10。各臂点成功率按unknown=0取保守下界，并同时报告unknown=1上界。两个下界成功率之差仍只是规定的点估计，并非该差值的数学下界；差值界另按A_lower−B_upper、A_upper−B_lower计算。E−L=N是恒等式，不是检验结果。",'',"CI为64个state成簇bootstrap5000次，每簇保留4个seed，各controller/arm共用抽样，seed202610131001。下表是点估计的percentile95% CI；完整unknown界、CI包络、S300/S400敏感性在E_L_N.csv与results.json。所有数值单位为pp。",'',"## 事实：P1、P2与逐controller效应",'',f"P1通过：N<0的{p1['N_negative']}个controller中，{p1['negative_with_E_ge10pp']}个E≥10pp，超过至少一半要求；全部14个中{p1['both_E_L_ge10pp']}个同时E、L≥10pp，超过至少5个要求。P2通过：E>0为{p2['E_positive']}/14，L>0为{p2['L_positive']}/14；N符号为负12、正2、零0。unknown的所有允许赋值不改变E、L正号数量（分别11与14）。",'',"|controller|E [95% CI]|L [95% CI]|N [95% CI]|","|---|---:|---:|---:|"]
    for n in names:
        vals=[]
        for k in ['E_S500','L_S500','N']:
            x=r['effects'][n][k];vals.append(f"{fmt(x['point'])} [{fmt(x['CI95_point'][0])}, {fmt(x['CI95_point'][1])}]")
        lines.append('|'+n.replace('_macflow','')+'|'+'|'.join(vals)+'|')
    lines += ['', '敏感性分析（DEVELOPMENT；不替代S500主判定）：', '', '|切换臂|平均E pp|平均L pp|E>0|L>0|N<0中E≥10pp|E、L均≥10pp|', '|---|---:|---:|---:|---:|---:|---:|']
    for schedule,s in sensitivity.items():lines.append(f"|{schedule}|{fmt(s['mean_E'])}|{fmt(s['mean_L'])}|{s['E_positive']}/14|{s['L_positive']}/14|{s['negative_N_with_E_ge10pp']}/12|{s['both_ge10pp']}/14|")
    lines += ['',"解释（DEVELOPMENT）：多数controller的两个阶段效应都为正且较大，而固定η净偏好的符号来自二者差值，符合预注册的H_compress映射。此结论支持该14-controller测试床上的机制解释；没有因果识别到‘所有过往零样本失败的唯一根因’，也不扩展到其他场景或任意controller。S500形成的有限差分亦不能严格拆成完全独立、无交互的局部时间段因果效应。",'',"## 事实：P3扩展动作空间",'',f"固定集每折源均值选择F10，平均遗憾{fmt(fixed['mean_regret'])}pp、最大{fmt(fixed['max_regret'])}pp。扩展集每折选择S500，平均遗憾{fmt(expanded['mean_regret'])}pp、最大{fmt(expanded['max_regret'])}pp；平均遗憾仅为固定集的{expanded['mean_regret']/fixed['mean_regret']*100:.2f}%，满足≤1/3条件，但最大遗憾超过5pp。因此 **P3未通过（DEVELOPMENT）**，不得因平均改善而改判。",'',f"unknown兼容的平均遗憾范围为固定[{fmt(fixed['mean_unknown_bounds'][0])},{fmt(fixed['mean_unknown_bounds'][1])}]pp、扩展[{fmt(expanded['mean_unknown_bounds'][0])},{fmt(expanded['mean_unknown_bounds'][1])}]pp；扩展最大遗憾范围[{fmt(expanded['max_unknown_bounds'][0])},{fmt(expanded['max_unknown_bounds'][1])}]pp。这些是允许赋值的边界，不是采样CI。",'',"最大的点遗憾来自base_u_v3：S500成功211/256，Freg成功226/256，差15/256=5.859375pp；两臂unknown分别5、4，使该controller遗憾界为3.91–7.42pp。base_u_v9_local的S500与F10无unknown，差13/256=5.078125pp，仍确定超过5pp；所以即便v3的unknown全部有利，P3的最大遗憾门也不能通过。其余非零遗憾为v7_local 2.734375pp、v10_local与v8_local各0.390625pp。",'',"解释（DEVELOPMENT）：扩展动作空间大幅改善平均零样本选择，但预注册的统一最坏情况保证未成立。",'',"## 事实：P4分解关系与描述选动作",'',"源13controller拟合等权z-score与PCA2，目标仅投影；ridge α=1、不惩罚截距、无调参。每折只用源标签选动作；目标标签仅用于评估。符号按负/零/正精确相等计分。",'',"|描述|效应|Spearman|符号准确率|源均值符号准确率|RMSE pp|源均值RMSE pp|","|---|---|---:|---:|---:|---:|---:|"]
    for rep,v in p4.items():
        for label,m in v['ridge_metrics'].items():lines.append(f"|{rep}|{label}|{m['Spearman']:.3f}|{m['sign_accuracy']*100:.2f}%|{m['source_mean_sign_accuracy']*100:.2f}%|{fmt(m['rmse'])}|{fmt(m['source_mean_rmse'])}|")
    lines += ['',"R5的E预测明显强于其N预测：ρ分别0.842与0.301，E的符号13/14高于源均值11/14，RMSE由24.88pp降至8.59pp。这是与设计者预期相反的重要局部正面发现。L的ρ仅0.209（R3为0.090）；L全14个为正，100%符号准确率与无描述源均值相同，不能用它证明描述学到了L的大小。N的符号12/14亦未优于源均值。源均值留一Spearman为−1是移除当前目标后均值随目标真值机械反向变化的性质，不是独立预测模型效果；该参考主要用于RMSE与符号基线。",'',"P4没有预注册方向性预测，也没有定义‘可预测’显著性阈值。因此报告这些观察，不补造正式统计门：E显示可学习信号，L尚缺充分证据，完整分解controller–Q可学习性尚未确立。样本仅14个controller，未进行新的独立controller验证；不能把L的弱相关解释成数学上不可学习。",'',"两种描述的1-NN平均遗憾均2.65pp、最大24.22pp，劣于源平均选择的1.03pp与5.86pp；没有进一步降低遗憾。最近源的动作选择在若干fold上失败，完整邻居、选择与regret见results.json和FOLD_PREPROCESSING.json。",'',"解释（DEVELOPMENT）：下一轮可将E的结构化预测作为候选，但不能声称完整critic或基于描述的动作选择已验证成功；L与选择最坏情况仍需解决。",'',"## 事实：例外与发生步数",'',"不满足E、L同时≥10pp的四个例外为v10_local、v7_local、v8_local、v9_local；前三者主要表现为F10/F13已经高成功或E很小，v9表现为晚切换后E为负。下表给出这些controller及P3关键例外v3各臂实际终止类型和发生步数。SUCCESS列用首个成功步，TIMEOUT列用实际终止步；这不是死锁检测时间。若有数值unknown则取记录中的error.step，不将其当失败或死锁。每条失败的state、seed、UID与精确步数见EXCEPTION_FAILURE_RECORDS.csv；逐arm完整步数数组保留在results.json中的例外摘要（v3详情见CSV）。",'',"|controller|臂|实际类型|数量|步数 min / median / max|","|---|---|---|---:|---|"]
    for x in summary:lines.append(f"|{x['controller'].replace('_macflow','')}|{x['arm']}|{x['outcome']}|{x['count']}|{x['step_min']} / {x['step_median']:g} / {x['step_max']}|")
    lines += ['',"这些记录没有合格的DEADLOCK事件证据。TIMEOUT只能称超时，不能冒充deadlock；success或运动渲染也不新增任何安全证据。",'',"## 产物与复现",'',"代码：analyze.py、verify_and_report.py。数值：results.json、E_L_N_MAIN.csv、E_L_N.csv、ARM_RATES.csv、RIDGE_PREDICTIONS.csv、P3_*_folds.csv、FOLD_PREPROCESSING.json。图：E_L_scatter.png/pdf，点编号见SCATTER_CONTROLLER_INDEX.csv。审核：ANALYSIS_AUDIT.json、RAW_REPRODUCTION_AUDIT.json、HISTORICAL_ASSET_INDEX.json及预注册/授权收据。",'',"复现：使用指定venv Python以-B运行analyze.py，再运行verify_and_report.py。脚本限制文件写入本目录，并将BLAS线程限制为1，无GPU、rollout或DB依赖。"]
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (HERE/'HANDOFF.md').write_text('''# Phase decomposition handoff

COMPLETE (pure analysis; all conclusions DEVELOPMENT). Round10 numerical-completion gate explicitly authorized by user; round10 video status remains unchanged.

H_compress SUPPORTED: P1 9/12 eta10-preferring controllers have E>=10pp; 10/14 have both E,L>=10pp. P2 E>0 11/14, L>0 14/14; N negative12, positive2.

P3 FAIL: expanded source-average selection mean regret1.0324pp vs fixed32.8962pp, but max5.8594pp>5. Unknown cannot rescue the whole gate because v9's5.0781pp deficit has no unknown.

P4: R5-PCA2 predicts E with Spearman0.8418, sign13/14, RMSE8.59pp; L Spearman0.2088 and its sign100% only equals uniform-positive source-mean reference. Partial positive evidence for learning E; full decomposed controller–Q learnability not established. R3/R5 1NN mean2.6507pp, max24.2188pp, worse than source-average menu selection. No post-hoc predictability cutoff was invented.

Read REPORT.md and results.json. Exact failure/termination steps in EXCEPTION_FAILURE_RECORDS.csv and EXCEPTION_OUTCOME_STEPS.csv; timeout is not deadlock. No new rollout or DB write. All18688 raw labels/hashes independently matched frozen evidence. Input hashes unchanged.

Reuse commands: specified venv Python -B analyze.py, then -B verify_and_report.py (from this directory or with absolute paths). Models/descriptors only source13 controllers per fold. R3 common96-state means; R5 frozen155D main scheme. No new checkpoint training or controller inference.

Next research candidate: structured E prediction, with independent-controller validation and a better L model; do not claim that descriptor-based action selection improved, or that the universal <=5pp guarantee passed.
''')
    artifacts=['analyze.py','verify_and_report.py','results.json','REPORT.md','HANDOFF.md','E_L_N.csv','E_L_N_MAIN.csv','E_L_scatter.png','E_L_scatter.pdf','ANALYSIS_AUDIT.json','RAW_REPRODUCTION_AUDIT.json']
    write('COMPLETE.receipt.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='COMPLETE',scope='pure frozen-evidence analysis only',evidence_level='DEVELOPMENT',prereg_sha256=sha(HERE/'PREREGISTRATION.md'),authorization_sha256=sha(HERE/'START_GATE_AUTHORIZATION.receipt.json'),artifacts={p:sha(HERE/p) for p in artifacts},rollouts_added=0,shared_DB_writes=0,round10_status_unchanged=True))
    print(json.dumps(dict(status='COMPLETE',raw_records_verified=len(cases),raw_outcomes=dict(raw_outcomes),exception_controllers=detail_names,report_lines=len(lines))))
if __name__=='__main__':main()
