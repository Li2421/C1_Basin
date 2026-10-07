"""Complete interpretation after locking static conclusions and fresh replication."""
import itertools
from collections import Counter
from datetime import datetime, timezone
import numpy as np
from diagnostics.astra_true_q_audit.audit import OUT, OLD, ROOT, read, write, sha, counts, arrays, corr

def main():
    fresh=read(OUT/'fresh_replication.json'); assert fresh['all_three_replicate']
    verify=read(OUT/'verification.json'); assert verify['traces_checked']==624
    control=read(OUT/'control_geometry_audit.json')
    control.update(final_verdict='TRUE_Q_GEOMETRY_EXISTS',fresh_comparisons=fresh['comparisons'],
        inference_scope='Conditional on three deliberately selected distinct frozen states and fixed phi endpoints.',
        confidence_correction='All three exact paired p values pass Bonferroni .05/3.',
        old_interval_issue='Old paired Wald/FD intervals can collapse to zero width at unanimity; replaced with exact binomial and conservative paired-arm difference bounds.',
        old_report_read_after_static=True)
    write('control_geometry_audit.json',control)
    q=read(OLD/'full_horizon_q_map.json'); timeout=read(OUT/'timeout_substitution_audit.json')
    allpairs=[]
    for s in q['states']:
        for a,b in itertools.combinations(s['policies'],2):
            if a['Q_D']==b['Q_D']:continue
            low,high=(a,b) if a['Q_D']<b['Q_D'] else (b,a)
            ds=low['Q_success']-high['Q_success']; dt=low['Q_timeout']-high['Q_timeout']
            kind='SUCCESS_INCREASE' if ds>0 else 'TIMEOUT_SUBSTITUTION' if dt>0 else 'OTHER'
            allpairs.append({'state_id':s['state_id'],'low_policy':low['probe_name'],'high_policy':high['probe_name'],
                'low_Q':[low[k] for k in ['Q_D','Q_success','Q_timeout','Q_collision']],
                'high_Q':[high[k] for k in ['Q_D','Q_success','Q_timeout','Q_collision']],
                'delta_Q_S':ds,'delta_Q_T':dt,'classification':kind})
    fresh_lookup={(r['state_id'],r['flow_seed'],r['arm']):r for r in fresh['records']}
    fresh_changes=[]
    for r in fresh['records']:
        if r['arm']!='phi_minus':continue
        high=fresh_lookup[(r['state_id'],r['flow_seed'],'phi_plus')]
        label='TRUE_ESCAPE' if r['outcome']=='success' else 'TIMEOUT_SUBSTITUTION' if r['outcome']=='timeout' else 'DELAYED_DEADLOCK' if r['outcome']=='deadlock' and r['steps']>high['steps'] else 'NO_EFFECT'
        fresh_changes.append({'state_id':r['state_id'],'seed':r['flow_seed'],'classification':label,'low_steps':r['steps'],'high_steps':high['steps']})
    timeout.update(fresh_paired_changes=fresh_changes,fresh_change_counts=dict(Counter(r['classification'] for r in fresh_changes)),
        all_unequal_Q_policy_pairs=allpairs,all_pair_categories=dict(Counter(p['classification'] for p in allpairs)),
        all_pair_caveat='Overlapping descriptive pairs, not independent observations or independently significant comparisons.',
        fresh_comparisons=fresh['comparisons'],fresh_low_arm_outcomes=dict(Counter(r['outcome'] for r in fresh['records'] if r['arm']=='phi_minus')),
        task_liveness_status='MIXED',
        status_scope='Whole historical map: success-versus-deadlock contrasts on baseline-success states, timeout-only recovery on baseline-deadlock states.',
        decisive_recovery_status='TIMEOUT_SUBSTITUTION_ONLY',
        genuine_escape_from_p0_deadlock_states_found=False,
        interpretation='Deadlock-control geometry replicates. Successful recovery from p0-deadlock states remains unobserved. Blanket absence of success-related control effects across the entire map would be false.')
    write('timeout_substitution_audit.json',timeout)
    multi=read(OUT/'multimodality_audit.json')
    multi['success_contrasts_not_multimodality']='22 descriptive unequal-Q pairs improve success on baseline-success states. No separated successful regions with an adverse intermediate region were demonstrated.'
    write('multimodality_audit.json',multi)
    feature=read(OUT/'feature_claim_audit.json'); rows=feature['rows']
    counter=[]; concordant=discordant=0
    for a,b in itertools.combinations(rows,2):
        dq=a['probabilities']['deadlock']-b['probabilities']['deadlock']; dx=a['displacement_censored_100']-b['displacement_censored_100']
        if dq==0 or dx==0:continue
        if dq*dx<0:concordant+=1
        else:
            discordant+=1
            if a['state_id']==b['state_id']:
                counter.append({'state_id':a['state_id'],'policy_a':a['policy'],'policy_b':b['policy'],
                    'Q_a':a['probabilities'],'Q_b':b['probabilities'],
                    'displacement_a':a['displacement_censored_100'],'displacement_b':b['displacement_censored_100']})
    # Direct trajectory conditional means from raw, independent of previous feature summary.
    grouped={e:[] for e in ['deadlock','success','timeout','collision']}; fixed20=[]
    base=OLD/'raw/q_map'
    for r in read(base/'manifest.json')['records']:
        d=arrays(base,r); disp=float(np.linalg.norm(d['positions_after'][min(100,len(d['event']))-1]-d['positions_before'][0]))
        grouped[r['outcome']].append(disp)
        if len(d['event'])>=20:fixed20.append((float(np.linalg.norm(d['positions_after'][19]-d['positions_before'][0])),int(r['outcome']=='deadlock')))
    feature.update(identifiable_within_state_counterexamples=counter[:12],
        descriptive_pair_concordance={'concordant':concordant,'discordant':discordant,'fraction':concordant/(concordant+discordant)},
        raw_conditional_displacement={e:{'n':len(v),'mean':float(np.mean(v)) if v else None} for e,v in grouped.items()},
        genuinely_fixed20_trace_spearman=corr([x[0] for x in fixed20],[x[1] for x in fixed20]),
        sufficient_structure='Not established; pooled rank association survives descriptive checking, but confounded by stopping time and selected state mixture.')
    write('feature_claim_audit.json',feature)
    nonsmooth=read(OUT/'nonsmoothness_audit.json')
    nonsmooth.update(final_support={'basin_like_outcome_regimes':'SUPPORTED at sampled resolution',
        'mathematical_nonsmoothness':'NOT_ESTABLISHED','multimodality':'NOT_ESTABLISHED'},
        fresh_evidence='Opposite fixed policies lead to mostly timeout versus universally deadlock in 32 fresh paired seeds on each of 3 states.',
        classification_counts=dict(Counter(s['classification'] for s in nonsmooth['slices'])),
        alternative_not_falsified='A steep but smooth Q_D transition between the sampled phi points.')
    write('nonsmoothness_audit.json',nonsmooth)
    table=[]
    for c in fresh['comparisons']:
        lo=c['low']; hi=c['high']; interval=lo['exact_95_intervals']['deadlock']
        table.append(f"| {c['state_id']} | {lo['counts']['deadlock']}/32 | [{interval[0]:.3f}, {interval[1]:.3f}] | {hi['counts']['deadlock']}/32 | {lo['counts']['success']} | {lo['counts']['timeout']} | {c['paired_exact_p']:.3g} |")
    report='''# C1 全时域 Q_D 独立科学审计

**CASE A — CONTROL GEOMETRY REPLICATES AND IS BASIN-LIKE**

**MULTIMODALITY STATUS: NOT_SUPPORTED**  
**TASK-LIVENESS STATUS: MIXED**（整个历史参数图）；三个决定性失败状态的恢复子集为 **TIMEOUT_SUBSTITUTION_ONLY**。

这里的 basin-like 指已采样参数点之间显著不同的终止结果区域。数据没有证明概率函数不连续，也没有排除陡峭但光滑的过渡。MIXED 不表示发现了原本失败状态的成功恢复：它承认旧图中原本成功的状态也存在由坏策略造成死锁、换回好策略成功的对照。

## 独立性与审计顺序

先核对 manifest、原始 NPZ、Q 计数和方向记录，写入 `blind_static_conclusion.json` 后才读取旧叙述报告。旧结论此前已出现在会话上下文中，因此这属于分析流程隔离，不能声称审计者完全盲法。原实验的文件全部保留。

432 条历史轨迹共 112,561 个物理步（包括缓存尾段）用冻结环境逐步重放，终止标签、位置和确定性状态反馈 residual 均一致。10 个起点的位置、速度、41 样本 monitor 历史和 candidate_since 与源轨迹精确一致。新实验采用独立编写的 continuation 调度，调用原冻结 Flow、两次硬投影与环境。没有使用旧风险函数或 certificate。

旧数据的 360 条主映射和 72 条方向验证计数可重现。旧方向验证确为 0/8、1/8、0/8 对 8/8，种子 61001–61008 与映射种子分离。

需要收紧的统计解释：

- 主映射中的 seed19073 同时生成了按未来 outcome/offset 选择的源轨迹，其尾段受选择条件影响；不等价于给定起点后的新独立随机抽样。JSON 同时提供去除该种子的三样本切片。
- 10 个起点只有 8 个源 pair ID；三个复现起点来自 231、228、227，确实不同，但均按大 FD 范数筛选，不是代表性随机状态样本。
- 三个状态的方向几乎相同，不能称为三个独立发现的控制方向。
- 跨状态汇总 p 值没有处理源状态依赖和筛选；审计主要依据逐状态新鲜复现。
- 旧 Wald 差值/FD 区间在全相同结果时会退化为零宽，不能据此断言概率精确为 0 或 1。本审计改用 exact 二项区间及保守的概率差区间。
- 8/6/4/2/1 s 来自不同轨迹，无法识别“沿同一轨迹接近 latch 导致方向增强”的时间因果趋势。

## A：真实 Q_D 控制几何

**TRUE_Q_GEOMETRY_EXISTS**，限定于所选状态和参数对。复现计划在运行前锁定三个原方向验证状态及其完整精度参数，未调整 phi。每臂 32 个新种子 92022001–92022032，两个策略使用配对的逐步 Flow key。不同状态的 key 再按 pair ID 分开。没有旧种子复用。

低风险 phi 约为 `(-0.0883883476483184, 0, +0.0883883476483184)`，高风险为相反数。G_phi 每步重算，持续到 first event 或 H=850；Flow 每步随机。G_phi 本身输出确定性四维 residual。

| 起点 | 低风险死锁数 | 低风险 Q_D exact 95% CI | 高风险死锁数 | 低风险成功数 | 低风险超时数 | paired exact p |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

高风险每臂 Q_D 的 exact 95% 区间均为 [0.891,1]。高减低的保守 95% 区间分别约为 [0.744,1]、[0.598,0.985]、[0.744,1]。三个 paired exact p 均通过三重比较 Bonferroni .05/3 门槛。无需追加至 64 种子。

GPU 复现共 192 条 continuation、54,357 个物理步，计算约 67 s。Flow 在 Slurm GPU 上运行，硬投影和环境使用原 CPU 实现。没有矢量化改写监测器或投影求解器。

![Fresh Q_D](figures/fresh_q_d.png)

## B：非光滑与盆地证据

旧的 7/10 标签可由相邻粗网格差值 ≥0.75 这个规则重现，但该规则本身不是连续性检验。

goal 切片为 [-.25,-.125,0,.125,.25]。D1 的估计是 [0,0,1,1,1]；D2/D4/D6 为 [0,.25,1,1,1]；D8 为 [0,.75,1,1,1]。S8g/S4g 为 [.75,0,0,0,0]。这些说明已采样点存在宽平台和强烈的结果区域差异。

审计分类：7 个 `BASIN_BOUNDARY_EVIDENCE`，2 个 `FLAT_OR_SATURATED`，1 个 `INSUFFICIENT_DATA`。最后一个状态不能仅凭稀疏弱变化就叫 piecewise smooth。

三处强对照已用新鲜数据复现，因此 basin-like 的操作性描述成立；“数学非光滑/不连续”最多合理猜测。两个 delta 的 FD 是割线，尚未证明导数存在、稳定或不存在。凸硬投影 active-set 改变也不自动意味着投影值或 Q_D 不连续。

![Raw slices](figures/phi_slices.png)

## C：多模态

严格分类为 **NO_MULTIMODAL_EVIDENCE**。没有在相同起点下观测两个分离的高成功区域及其中间的低成功区域，也没有证明低死锁区域之间不连通。相反方向、多个 timeout 点、不同状态的成功，都不能替代这一证据。

旧图的跨状态平均“basin”散点，以及把第三参数不同的点投影到相同二维坐标，都不足以识别成功盆地拓扑。本审计不复用该图作为多模态证据。可选二维网格未运行，避免为审计另做至少 648 条搜索式 continuation。

## D：低死锁是否意味着完成任务

三个新复现状态的低风险臂合计：93 timeout、3 deadlock、0 success、0 collision。所有 Q_D 减少都转入 Q_T；其余 3 个配对结果均为 DELAYED_DEADLOCK，低风险臂在 440/516/542 步死锁，对应高风险臂在 40/104/40 步死锁。每臂 0/32 success 的 exact 95% 上界仍为 0.109，因此这是未观测到成功，不能证明成功概率严格为零。

旧 p0 死锁基线的 160 个策略比较精确重现：0 TRUE_ESCAPE、47 TIMEOUT_SUBSTITUTION、22 DELAYED_DEADLOCK、91 NO_EFFECT。它们有重复基线，不是 160 个独立状态。

但是，若审计整个图的所有同状态不同 phi 比较，会发现 22 个描述性 pair 随 Q_D 下降伴随 Q_S 上升，另有 101 个伴随 timeout 上升。前者全部来自 p0 本来能成功的状态。例如 S8g_pair229 与 S4g_pair230 的 goal_m2 为 3/4 deadlock、1/4 timeout，p0 则为 4/4 success。它们是避免策略诱发失败的证据，不能称作三个失败起点的恢复，也未经过本轮新种子确认。

因此全图 task-liveness 标签为 MIXED；对决定性失败状态，应明确写：**deadlock-control geometry exists, but task-liveness geometry has not yet been found.**

![Outcome comparison](figures/outcomes.png)

## 投影与部署语义

旧 84 对轨迹的 0 alias 可复现；同时重新计算 first/middle/last 步的两次投影、active-set signature 与随机 key。联合旧/新 624 条轨迹共检查 1,872 个步，投影动作与存档一致，新 192 条的事件和 G_phi 也逐步重放一致。

投影总体为 **PARTIALLY_COMPRESSES**：执行差异通常比请求差异小，但关键策略差异保留下来。新 96 对也没有完全塌缩。该结果排除了“投影消灭了这些对照”，不证明投影创造或支配了因果几何。长度不同的轨迹只在共同非终止区间比较，不能把后续缺失状态补成相同。

## 位移特征

旧 100 步位移的 pooled Spearman -0.832911 可重现，失败起点内的 rho 约 -0.73 至 -0.84；在成功起点 S4g 则仅 -0.137。可识别的同状态排序反例保存在 `feature_claim_audit.json`。

关键限制：旧实现实际上使用 min(100, termination) 计算位移。20/40/80 步便死锁的轨迹得到更短运动窗口，把最终结果的信息带进了所谓“100 步早期特征”。只看四个样本全部存活到 100 步的 47 个 state-policy 点，rho 降至 -0.389；这项变化还混有样本组成变化，不能把全部差异归因于截尾。

固定 20 步、所有轨迹均能到达的观测窗口仍有 trace-level rho=-0.745，说明关联并非完全由提前终止造成；但这仍是跨状态混合关联。旧“persistent displacement”实为起点到终点的净位移，未测量路径中是否持续偏移。

一个明确反例是 S4g_pair230：goal_m1 的平均位移约 0.0878，Q_D=0；goal_m2 的位移更大、约 0.1946，Q_D 却升至 0.75。timeout 组的平均位移约 0.3313，success 组约 0.5329，deadlock 组约 0.0485。低死锁与大位移的关系不能自动识别任务成功。

所以“关联可重现、单独不充分”成立，“已验证强早期控制排序变量”不成立。位移可能由远离目标的运动产生，timeout 同样能有大位移。本审计没有把它定义为新 R_risk。

## 对旧结论的修正

保留真实死锁控制几何及经验性 basin-like 描述；撤回任何已证明不连续、多模态成功恢复、时间因果窗口趋势的解读。原始事件数据可用，主要问题在推断范围、置信区间和特征观测窗口，而非闭环执行语义。

结论依据已采样状态及本轮 192 条复现，不推广到整个连续状态/参数域。完整参数、种子、源码与数据 SHA256 见 manifest.json。
'''
    (OUT/'astra_independent_audit.md').write_text(report)
    write('decision.json',{'primary':'CASE A — CONTROL GEOMETRY REPLICATES AND IS BASIN-LIKE',
        'MULTIMODALITY_STATUS':'NOT_SUPPORTED','strict_multimodality':'NO_MULTIMODAL_EVIDENCE',
        'TASK_LIVENESS_STATUS':'MIXED','decisive_deadlock_starting_states':'TIMEOUT_SUBSTITUTION_ONLY',
        'mathematical_discontinuity':'NOT_ESTABLISHED','completed_at_utc':datetime.now(timezone.utc).isoformat()})
    print(read(OUT/'decision.json'))

if __name__=='__main__':main()
