"""Assemble a self-contained Round19b numerical and delivery report."""
from __future__ import annotations
import datetime,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/name).read_text())
def sha(name):return hashlib.sha256((HERE/name).read_bytes()).hexdigest()
def write(name,x):(HERE/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def pct(x):return f'{100*x:.2f}%'
def main():
    gates={name:read(name+'_REPORT.json') for name in ('G0','G1A','G1B','G2')}
    fixed=read('FIXED_RESULTS.json');sched=read('SCHEDULE_RESULTS.json');video=read('VIDEO_GATE.json')
    probe=read('R5_FREEZE.receipt.json');gap=read('PROBE_EVIDENCE_AUDIT.json')
    assert all(gates[n]['status']=='PASS' for n in ('G0','G1A','G2'))
    assert fixed['headroom_gate_passed'] and sched['status']=='COMPLETE'
    assert probe['status']=='FROZEN_BEFORE_ANY_NEW_SUCCESS_LABEL'
    answers=dict(
        Four_eta_regime='SUPPORTED' if fixed['headroom_gate_passed'] else 'REJECTED',
        H_compress_Four=sched['H_compress_Four'],E_learnability=sched['E_learning']['status'])
    result=dict(status='NUMERICAL_COMPLETE_VIDEO_INCOMPLETE' if not video['complete'] else 'COMPLETE',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),equivalence_gates={n:gates[n]['status'] for n in gates},
        gate_counts=dict(G0='8/8',G1a='32/32',G1b=f"{sum(x['passed'] for x in gates['G1B']['results'])}/4",G2='25/25'),
        hypotheses=answers,stage1=dict(probe_states=16,short_probes=1200,unknown=len(probe['unknown_probes']),
            feature_shape=probe['shape'],features_sha256=probe['features_sha256'],preflight_gap=gap['workflow_gap']),
        fixed=dict(rollouts=fixed['total_rollouts'],unknown=fixed['unknown'],majority_arm=fixed['majority_arm'],
            condition_a_qualifying=fixed['condition_a_qualifying'],condition_b_qualifying=fixed['condition_b_qualifying'],
            headroom_gate_passed=fixed['headroom_gate_passed']),
        schedule=dict(rollouts=sched['schedule_rollouts'],unknown=sched['schedule_unknown'],
            fixed_mean_regret_pp=sched['fixed_mean_regret_pp'],expanded_mean_regret_pp=sched['expanded_mean_regret_pp'],
            fixed_max_regret_pp=sched['fixed_max_regret_pp'],expanded_max_regret_pp=sched['expanded_max_regret_pp'],
            both_E_L_ge10pp_controllers=sched['both_E_L_ge10pp_controllers'],E_learning=sched['E_learning']),
        video=dict(complete=video['complete'],exported=len(video['videos']),missing_roles=video['missing_roles'],
                   official_exit_code=video['official_exit_code']),
        artifacts={name:str(HERE/name) for name in ('PREREGISTRATION.md','G0_REPORT.json','G1A_REPORT.json','G1B_REPORT.json',
            'G2_REPORT.json','R5_FREEZE.receipt.json','FIXED_RESULTS.json','SCHEDULE_RESULTS.json','FIXED_CENSUS.csv',
            'E_L_N.csv','VIDEO_GATE.json','PROBE_EVIDENCE_AUDIT.json')})
    write('results.json',result)
    qualifiers=[]
    for name in fixed['condition_a_qualifying']:
        r=next(x for x in fixed['table'] if x['controller']==name)
        best=r['best_arm'];major=fixed['majority_arm'];adv=100*(r['success_rates'][fixed['arms'].index(best)]-r['success_rates'][fixed['arms'].index(major)])
        qualifiers.append(f"| `{name}` | {best} | {pct(r['success_rates'][fixed['arms'].index(best)])} | {adv:.2f} pp |")
    em=sched['E_learning'];ratio_text=(f"{em['RMSE_ratio']:.3f}" if em.get('RMSE_ratio') is not None else 'undefined')
    chosen_index=sched['all_arms'][3:].index('S15to10_786')
    chosen_success=sum(row[chosen_index] for row in sched['schedule_success_counts'])
    chosen_known=sum(row[chosen_index] for row in sched['schedule_known_counts'])
    chosen_unknown=sum(row[chosen_index] for row in sched['schedule_unknown_counts'])
    learn_line=(
        f"所选时间表 `{em['selected_schedule']}`；留一 Spearman {em['Spearman']}, RMSE {em['RMSE_pp']:.2f} pp，"
        f"源均值基线 {em['baseline_RMSE_pp']:.2f} pp，RMSE 比 {ratio_text}。"
        if em['selected_schedule'] else f"{em['reason']}。")
    g1bad=[r for r in gates['G1B']['results'] if not r['passed']]
    first=(f"等价门：G0 **PASS 8/8**、G1a **PASS 32/32**、G1b **{gates['G1B']['status']} "
           f"{4-len(g1bad)}/4**（条件性、非阻断）、G2 **PASS 25/25**。"
           f"Four 跨 controller 的固定 η regime 差异 **{answers['Four_eta_regime']}**：多数最佳臂 {fixed['majority_arm']}，"
           f"{len(fixed['condition_a_qualifying'])} 个 controller 的非多数最佳臂领先至少 10pp。"
           f"阶段压缩 H_compress_Four **{answers['H_compress_Four']}**；由冻结探测学习 E **{answers['E_learnability']}**。"
           f"数值研究已完成；正式视频交付 **{'COMPLETE' if video['complete'] else 'INCOMPLETE'}**。")
    text=f'''# Round19b Four controller axis v2

{first}

## 事实与门控

补充预注册 UTC `{read('PREREGISTRATION.receipt.json')['utc']}`，SHA256 `{sha('PREREGISTRATION.md')}`。v1 因历史首动作字段缺失而在 Stage 0 停止，本轮只修改 G1 与目录。G1a 事前冻结 32 条历史干预记录，v12 与 v10-wide 各 success 8、timeout 8；outcome、episode_length、两项 float64 最小距离全部逐位一致。G0 用新适配层复现 E8 固定 8 例；G2 的 25 个候选结构、配置、FP32 和同 key 前向均合格。G1b 的 {len(g1bad)} 条分叉从首步动作开始，见 `G1B_REPORT.json`；这项按补充预注册不阻断后续实验，不能称为全通过。

16 个新 probe state 和 32 个新 census state 经共享库可恢复物理初态去重，无碰撞。1,200 次 200 步无标签探测产生 25×240 R5，7 次不完整；列均值插补、population zscore 与 PCA-2 在任何新成功标签前冻结，收据 SHA256 `{sha('R5_FREEZE.receipt.json')}`。探测只记录进展、投影比例、最小两机器人距离、平均速度、低速步数；不读取终止标签。

固定臂 25 controller × 32 state × 3 臂 × 2 seed = **{fixed['total_rollouts']:,}** 条；unknown **{fixed['unknown']}**，成功率一律为成功数 / 已知种子数。多数最佳臂为 **{fixed['majority_arm']}**；条件(a)达到 **{len(fixed['condition_a_qualifying'])} ≥ 3**，条件(b)为 **{len(fixed['condition_b_qualifying'])}**，故按预注册进入时间表阶段。

| 达到(a)的 controller | 最佳臂 | 该臂成功率 | 相对多数臂优势 |
|---|---|---:|---:|
{chr(10).join(qualifiers)}

时间表 25 × 32 × 5 × 2 = **{sched['schedule_rollouts']:,}** 条；unknown **{sched['schedule_unknown']}**。所有 E=S−F_late、L=S−F_early、N=F_early−F_late 见 `E_L_N.csv`。`S15to10_786` 在 25 个 controller 上的已知种子合计 **{chosen_success}/{chosen_known}** 成功，另有 {chosen_unknown} 个 unknown；不能把这 {chosen_unknown} 个算作成功。同 run 的 best/final 一起留出，固定动作集的平均/最大 oracle 遗憾为 **{sched['fixed_mean_regret_pp']:.2f}/{sched['fixed_max_regret_pp']:.2f} pp**，扩展动作集为 **{sched['expanded_mean_regret_pp']:.2f}/{sched['expanded_max_regret_pp']:.2f} pp**。同时满足 E、L≥10pp 的 controller 有 **{len(sched['both_E_L_ge10pp_controllers'])}/25**；预注册 H_compress_Four 判定 **{answers['H_compress_Four']}**。

E 学习：{learn_line}按同 run 留一、仅用冻结 R5 PCA-2、源折标准化与 ridge α=1，预注册判定 **{answers['E_learnability']}**；逐折系数、预测、真值见 `SCHEDULE_RESULTS.json`。

## 证据链与限制

固定臂和时间表均先序列化请求，运行官方 `shared_rollout_db.plan` preflight（分别 4,800 与 8,000 全部 genuinely_missing、ambiguous=0），worker 只追加本目录 shard journals，单进程官方 merger 使用本目录的精确 seed 身份适配器，然后运行官方 postflight。数值 unknown 保留、未重跑。新执行身份由 checkpoint SHA、归一化器所用适配层及时间表定义确定；旧 intervention 标签只用于 G1a 验证，不代替新 census 标签。

探测前完成 state 去重，但**没有在提交短探测前留存官方 probe preflight**。这违反了共享库工作流的收据要求，已在 `PROBE_EVIDENCE_AUDIT.json` 明确登记；不能事后伪称事前 preflight，也不重复有效探测。所有探测输出为本目录的无标签源证据，共享 DB 中这 16 个 probe state 的成功行数为 0。

视频：3 个来自不同真实完整安全成功轨迹的 MP4 已导出，正式 `safety_success` 通过素材校验。统一 `new_benchmark_common.batch_videos` 入口已运行，返回 {video['official_exit_code']}；Four 无已记录的正式死锁判据，`deadlock_show` 与 `deadlock_resolution` 缺失，manifest `complete=false`。timeout 未标为 deadlock；因此大 batch 视频交付仍未完成。

解释：Four 的固定 η 差异已由此队列证明；阶段压缩与 E 可学习性按各自预注册数值门单独判定。结论仅针对同环境 fingerprint 的 25 个 best/final checkpoint、这些新 development state 与指定臂/时间表；不同环境版本的 9 个 run 未纳入。

![固定臂成功率](figures/fixed_rates.png)

![冻结 R5 PCA](figures/r5_pca.png)

![时间表 E 与 L](figures/schedule_effects.png)
'''
    (HERE/'REPORT.md').write_text(text)
    handoff=f'''# Round19b handoff

工作根目录 `/home/zhihan/research/Basin_C1`；Python `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。所有本轮文件在 `diagnostics/four_controller_axis_v2/`；仅按官方生命周期写入共享 rollout DB，没有 git commit。

结论：Four η regime {answers['Four_eta_regime']}；H_compress_Four {answers['H_compress_Four']}；E learning {answers['E_learnability']}。G0/G1a/G2 PASS，G1b {gates['G1B']['status']}（非阻断）。数值完成，视频缺两个正式死锁角色；probe 官方事前 preflight 收据缺失。

核心文件：`PREREGISTRATION.md`、`G1A_SELECTION.json`、`G0_REPORT.json`、`G1A_REPORT.json`、`G1B_REPORT.json`、`G2_REPORT.json`、`FRESH_STATES.json`、`R5_FROZEN.npz`、`R5_FREEZE.receipt.json`、`FIXED_PROTOCOL.json`、`SCHEDULE_PROTOCOL.json`、`FIXED_RESULTS.json`、`SCHEDULE_RESULTS.json`、`results.json`、`REPORT.md`、`videos/manifest.json`。

Slurm arrays: probe 21020 (25 tasks, max18 concurrent CPU), fixed 21045 (18 CPU), schedule 21063 (18 CPU); phases sequential. Exact records: fixed 4,800, schedule 8,000; unknown fixed {fixed['unknown']}, schedule {sched['schedule_unknown']}。视频入口失败的缺口不能用 timeout 补齐。

只读复算：用上面的 Python 运行 `diagnostics/four_controller_axis_v2/analyze_fixed.py`、`analyze_schedule.py`、`plot_results.py`、`finalize.py`。前两者会核对 DB 与原始 trace SHA；不要重新提交 rollout。若需正式视频 COMPLETE，须另有事前定义并记录的 Four 死锁判据和同初态基线/成功干预，不得改旧 outcome。
'''
    (HERE/'HANDOFF.md').write_text(handoff)
    write('DELIVERY.receipt.json',dict(utc=result['utc'],status=result['status'],files={name:sha(name) for name in
          ('REPORT.md','HANDOFF.md','results.json','FIXED_CENSUS.csv','E_L_N.csv','videos/manifest.json')},
          numerical_rollouts=fixed['total_rollouts']+sched['schedule_rollouts'],short_probes=1200))
    print(json.dumps(dict(status=result['status'],hypotheses=answers,video_complete=video['complete'])),flush=True)
if __name__=='__main__':main()
