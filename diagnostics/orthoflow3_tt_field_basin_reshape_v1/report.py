"""Generate a compact report and self-contained HTML from computed evidence."""
from pathlib import Path
import base64
import csv
import hashlib
import html
import json

HERE=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text())
def pct(x):return f'{100*x:.2f}%'
def pp(x):return f'{100*x:.2f}'
def ci(x):return '['+', '.join(pp(v) for v in x)+']'


def main():
    rows=read(HERE/'summary.json');decision=read(HERE/'decision.json');audit=read(HERE/'cache_audit.json')
    get=lambda cohort,contrast,scene:next(r for r in rows if (r['cohort'],r['contrast'],r['scene'])==(cohort,contrast,scene))
    a=get('primary','TT->FF','all');toy=get('primary','TT->FF','toy_give_way');ring=get('primary','TT->FF','ring_exchange')
    rep=get('factorial','TT->FF','all');corr=get('factorial','TT->TF','all')
    fieldcorr=get('factorial','FT->FF','all')['assignment_bounds']
    qs=read(HERE/'paired_seed_test_summary.json');stability=read(HERE/'paired_seed_stability.json')
    zero=read(HERE/'zero_correction_control.json')
    assert all(r['seed_outcome_disagreements']==0 for r in zero)
    table=['场景 | states×eta | TT basin | Field basin | change rate | rescue | break | Δvolume (pp), 95% CI | paired state sign p']
    for name,r in [('Toy',toy),('Ring',ring),('合计',a)]:
        table.append(f"{name} | {r['states']}×16 | {r['TT_count']}/{r['cells']} ({pct(r['TT_fraction'])}) | "
          f"{r['Field_count']}/{r['cells']} ({pct(r['Field_fraction'])}) | {r['change_count']}/{r['cells']} ({pct(r['change_rate'])}) | "
          f"{r['rescue']} | {r['break_count']} | +{pp(r['volume_delta'])} {ci(r['volume_delta_state_bootstrap95'])} | {r['state_sign_p']:.6g}")
    statement=(
        'Using 32 outcome-blind true-initial states from Toy GiveWay and Ring Exchange, we compared TT and Field on the same '
        '16 prespecified Sobol interventions per state and 16 matched continuation seeds, defining empirical robust membership as at least 15 successes. '
        f"Field changed {a['change_count']}/{a['cells']} memberships ({pct(a['change_rate'])}) across {a['states_changed']}/{a['states']} states, "
        f"with {a['rescue']} rescues and {a['break_count']} breaks; mean robust basin fraction increased from {pct(a['TT_fraction'])} to {pct(a['Field_fraction'])} "
        f"(paired difference +{pp(a['volume_delta'])} percentage points, state-cluster bootstrap 95% CI {ci(a['volume_delta_state_bootstrap95'])}; "
        f"two-sided state sign-test p={a['state_sign_p']:.3g}), with {a['volume_increase_states']} state-level increases, {a['volume_decrease_states']} decreases and {a['volume_tie_states']} ties. "
        f"A separate cached factorial cohort corroborated correction relocation with terminal safety retained ({corr['TT_count']}/{corr['cells']} to "
        f"{corr['Field_count']}/{corr['cells']} robust memberships; {corr['rescue']} rescues, {corr['break_count']} breaks). "
        'These retrospective paired results support systematic reshaping and net empirical expansion under the specified intervention measure, '
        'without establishing set inclusion, continuous basin volume, topology, or universal enlargement across state and proposal distributions.'
    )
    (HERE/'paper_evidence.txt').write_text(statement+'\n')
    caption=(
        'Paired robust-success membership for one Toy and one Ring TEST state selected by a fixed hash rule independent of outcomes. '
        'TT and Field use identical 3D Sobol samples, whose IDs agree across panels; the horizontal and vertical axes are eta1 and eta2, '
        'and marker area encodes eta3. Purple/green in the first two columns denote empirical B15 success for TT/Field; gray denotes non-B15. '
        'The final column distinguishes both non-B15, TT-only, Field-only and both B15. This is a projection of measured points, '
        'not a fixed slice, interpolation, or estimated continuous boundary. The full 3D view and all-state atlas are supplied separately.'
    )
    (HERE/'figure_caption.txt').write_text(caption+'\n')
    text=f'''TT / Field：robust success basin 配对实验闭环

Scientific decision
{decision['claim']}
严格限定：在本次预先冻结的共享 Sobol 离散采样测度和 Toy/Ring true-t0 状态分布上，支持“系统性重塑 + 净经验扩张”。存在 break，因此不支持集合包含；不宣称连续体积、拓扑或任意 proposal 分布上的统一扩张。此轮是既有冻结实验的回顾分析，不包装成新的前瞻性验证。

Experimental contract
TT：原生 Flow 生成终端动作 a_ref，经同一硬安全集合 U(x) 的投影得到 a_safe，再执行 P_U(a_safe + g_eta)。
Field=FF：相同原生噪声和 checkpoint，把同一物理 g_eta 加入 Flow 向量场，同时在每个伪时间步投影整个场；10 步积分后只做安全断言。eta 在整段物理 episode 固定，basis 每个物理步按该分支当前状态重算，g_eta 在该次伪时间积分内恒定。
主队列：全部 32 个原 TEST states（16 Toy + 16 Ring），true-t0、互不重叠的 source families；Toy 为冻结 WIDE-IC uniform draws，Ring 为冻结 native development draws，无 outcome rejection。未用 generator 或 critic 选 eta。
eta：每 state 相同 16 个 scrambled Sobol 点，seed=202610042，域 [0.5,1.25]×[-0.5,0.5]×[0,0.75]。重新生成后 float64 完全一致。量度为这 16 点的等权经验测度；box 均匀体积仅是该设计的采样目标，未给出连续积分误差界。
Seeds：future indices 0..15；root=2026100403；按 state alias 的 SHA256 前8位、future index、physical step 依次 fold_in。每个 state/eta/seed 在两分支完全对应。Toy/Ring horizon 分别 850/700、dt=0.05；各场景内 checkpoint、精度、物理环境、初态、目标、安全约束和终止条件一致。
Robust criterion：至少 15/16 collision-free terminal successes，即经验 B15。success 需 native collision_free_success、termination=success 且无 numerical error。数值未决保留 unknown；若 s>=15 则 B15，若 s+unknown<15 则 non-B15，否则 unresolved。这不是总体 p>=15/16 的置信保证。
Rollouts：主比较复用 32×16×16×2={audit['primary_requested']:,} 条记录；另复用 16-state 四路对照 {audit['factorial_primary_requested']:,} 条和零修正对照 {audit['zero_control_requested']:,} 条，总计 {audit['total_archived_records']:,} 条；新增 0。主队列 {audit['primary_numerical_attempts']} 次数值未决，但所有 TT/FF B15 分类均已确定；没有把数值错误填为普通失败。数据库只读，{audit['source_files_verified']:,} 个原文件 hash 验证，96 次跨分支存档噪声轨迹配对核对通过。

Main quantitative table
{chr(10).join(table)}

Paired contingency table（主队列，单位 state–eta）
                    Field non-B15       Field B15
TT non-B15          {a['both_failure']} both fail         {a['rescue']} rescue
TT B15              {a['break_count']} break              {a['both_success']} both succeed
ΔV = (rescue − break)/512；membership change = (rescue + break)/512。

Per-state evidence and statistics
29/32 states（90.63%，Wilson 95% CI {ci(a['states_changed_wilson95'])}%）有 membership changes；Toy 16/16，Ring 13/16。
Volume increase/decrease/tie：Toy 15/1/0；Ring 12/1/3；总体 27/2/3。ΔV 的 min/Q1/median/Q3/max 为 {', '.join(pp(v) for v in a['volume_delta_quantiles'])} pp。
主效应 95% CI 来自 50,000 次 state cluster bootstrap，pooled 估计按 scene 分层并保持 16:16 权重；固定 eta bank 和原 seed panel。方向检验为双侧 exact paired state sign test，零差异单独列出。512 个 eta cells 和16个 seeds 不作为独立 state 样本。补充 whole-state sign-flip p={a['state_signflip_p']:.6g}，需差分符号可交换/对称假设。
对 seed 层成功/失败另作每 cell 的 matched McNemar/binomial 检验，经全部512项 Holm 校正，有 {qs['familywise_significant_cells']} 个显著组合，涉及 {qs['significant_states']} 个 states（{qs['positive_cells']} 正向、{qs['negative_cells']} 负向）。仅配对有效 slots，作为条件于 solver-valid 的补充证据。
阈值敏感性：B14 净增加 14.65–14.84 pp；B16 净增加 13.48–13.87 pp；保守数值 assignment 下 state CI 下界仍为正。2,000 次同 state 内配对 seed 重采样净增加范围为 {ci(stability['volume_delta_percentile95_conservative'])} pp（中央95%）；这是已有 seeds 的稳定性分析，不是独立重复实验或总体概率认证。

Attribution and existing evidence
另一个 outcome-blind 冻结队列（8 Toy + 8 Ring，同16 Sobol）复核 TT→FF：49→100/256，62 rescue、11 break，15/16 states 增加、1 tie。
原先的“49→88”实际是 TT→TF：保持 terminal safety，单独把 correction 移入 Flow。73/256 memberships 改变，56 rescue、17 break，ΔV=+15.23 pp，state-bootstrap95% CI {ci(corr['volume_delta_state_bootstrap95'])} pp，sign p={corr['state_sign_p']:.6g}。这排除了“全部收益仅由安全位置变化造成”的解释。
FT→FF（保持 field safety）也为正：61–62→100，ΔV=14.84–15.23 pp，最保守 state CI 下界 {pp(fieldcorr['volume_delta_state_bootstrap95'][0][0])} pp，最大 sign p={fieldcorr['state_sign_p'][1]:.6g}。FT 有强制 terminal safety closure，四路设计并非完美正交。
eta=0 的 TT/TF、FT/FF 各256组 seed 对照结果逐 seed 完全相同。旧10/20/40积分分辨率检查无收益方向反转。TT 包含原生 terminal decoding；这些控制降低其解释空间，但不声称消除了所有实现机制差异。
之前的 state-conditioned basin 和更换未来 checkpoint 导致 16/16→0/16，分别支持 state dependence 和 controller dependence；它们不是 TT/Field relocation 的直接证据。较早 generator-proposal pilot 的 Toy robust count 40→26 必须保留，因此不能把当前等权 Sobol 结果推广到任意 eta proposal 分布。TT-failure-screened family_v1 不进入本轮主统计。完整审计见 evidence_audit.txt。

Basin visualization
figures/paired_basin_map.png：同样的 eta 点显示 TT、Field 和四类配对变化；两代表状态按固定 hash 选出。
figures/paired_basin_map_3d.png：同一组点的完整三维视图。
figures/all_states_atlas.png：全部32 states，无遗漏。
figures/per_state_volume.png：每个 state 的 TT/Field basin fractions。
每幅图同时提供 PDF/SVG。二维图明确是 eta1/eta2 projection，面积编码 eta3，不插值、不画未测边界。对应 ID、坐标、counts、membership 见 cell_membership.csv / figure_state_index.csv。

Paper-ready evidence statement（4 sentences）
{statement}

Reproduce and inspect
运行 bash diagnostics/orthoflow3_tt_field_basin_reshape_v1/reproduce.sh 即可从本地完整 seed snapshot 重算统计、测试、图和报告，无 rollout、无模型加载。
加 --refresh-from-cache 可从只读原数据库重新执行 provenance audit/export。analysis_protocol.json 保留分析定义；source_hashes.json、snapshot_integrity.json 保留原始和导出文件校验值。
代码：analyze.py、plot.py、report.py、verify.py、test_analysis.py、reproduce.sh。
结果：summary.json、main_quantitative_table.csv、per_state.csv、cell_membership.csv、paired_seed_tests.csv、threshold_sensitivity.json、paired_seed_stability.json、zero_correction_control.json、decision.json。
来源：paired_snapshot.npz、seed_outcomes.csv.gz、inputs/、cache_audit.json、planned_reuse.json、source_hashes.json。
全部产物绝对路径和 SHA256：artifact_inventory.csv；源数据及执行文件路径：source_hashes.json。report.html 内嵌图，可以直接查看。
'''
    (HERE/'final_report.txt').write_text(text)
    images=[]
    for stem in ('paired_basin_map','per_state_volume','paired_basin_map_3d','all_states_atlas'):
        content=base64.b64encode((HERE/'figures'/(stem+'.png')).read_bytes()).decode()
        images.append(f'<figure><img src="data:image/png;base64,{content}" alt="{stem}"><figcaption>{html.escape(caption if stem=="paired_basin_map" else stem.replace("_"," "))}</figcaption></figure>')
    doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>TT–Field paired basin evidence</title><style>body{max-width:1150px;margin:36px auto;padding:0 24px;font:16px/1.6 system-ui;color:#18212b}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 ui-monospace,monospace}img{width:100%;height:auto}figure{margin:35px 0}figcaption{color:#4c5864;font-size:13px}</style><body><pre>'+html.escape(text)+'</pre>'+''.join(images)+'</body></html>'
    (HERE/'report.html').write_text(doc)
    # All generated files except the inventory itself (no recursive self hash).
    inventory=[]
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_inventory.csv':
            inventory.append(dict(relative_path=str(p.relative_to(HERE)),absolute_path=str(p),bytes=p.stat().st_size,
                                  sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    with (HERE/'artifact_inventory.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,list(inventory[0]));w.writeheader();w.writerows(inventory)
    print(json.dumps(dict(report=str(HERE/'report.html'),artifacts=len(inventory),new_rollouts=0)))


if __name__=='__main__':main()
