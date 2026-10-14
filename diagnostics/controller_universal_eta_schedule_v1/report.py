from core import *

def main():
    prereg();gate();r=read(HERE/'NUMERICAL_RESULTS.json');v=read(HERE/'videos/manifest.json');post=read(HERE/'main_POSTFLIGHT.json');assert post['status']=='PASS';p=read(HERE/'protocol.json');g=r['group_metrics']['S500'];audit=dict(utc=utc(),original_preregistration_sha256=sha(HERE/'PREREGISTRATION.md'),active_preregistration_sha256=sha(HERE/'PREREGISTRATION_V2.md'),state_audit_sha256=sha(HERE/'STATE_DEDUP_AUDIT_V2.json'),consistency_gate32=read(HERE/'CONSISTENCY_GATE.json')['passed'],postflight=post,inherited_assets=read(HERE/'INHERITED_ASSET_AUDIT.json')['passed'],fresh_outcomes_not_read_before_registered_opening=True,endpoint_opening=read(HERE/'ENDPOINT_OPENING.receipt.json'),video_complete=v['complete'],missing_video_roles=v['missing_roles']);r.update(status='COMPLETE' if v['complete'] else 'NUMERICAL_COMPLETE_VIDEO_INCOMPLETE',video_manifest=v,audit=audit);write(HERE/'results.json',r);write(HERE/'FINAL_AUDIT.json',audit)
    text=f"""# Round10 universal Ring eta schedule

**H_schedule 未获普适支持；按预注册映射标为{r['H_schedule']}，G_universal={'通过' if r['G_universal'] else '未通过'}，held-out组={g['held_out']['status']}。** 全14个controller平均增益{100*g['all']['mean_gain']:.2f}pp，95%未知稳健区间{[round(100*x,2) for x in g['all']['hierarchical_CI_unknown_envelope']]}pp；held-out10个平均增益{100*g['held_out']['mean_gain']:.2f}pp，区间{[round(100*x,2) for x in g['held_out']['hierarchical_CI_unknown_envelope']]}pp。普适点门的明确缺口是v9_local：S500为234/256=91.41%，F10为247/256=96.48%，低5.08pp、无unknown，超过预定5pp阈值。Held-out的v3低于Freg点率5.86pp，但unknown可行差值为−7.42至−3.91pp，跨越5pp界，所以held-out未通过点门且未知稳健判定INCONCLUSIVE。均值收益成立，逐controller普适不劣未建立；不能把controller regime差异全部归因于固定η的动作空间限制。前瞻终点在首次读取新state结果前已预注册，机制解释为DEVELOPMENT。数值完成，正式视频仍缺{v['missing_roles']}；状态{r['status']}。

18,688主请求+32一致性门；主有效{post['valid']}，unknown {post['unknown']}，missing/quarantined均0，18,688份真实轨迹hash已核验。unknown保留、不重跑、不当作失败，区间保留所有请求分母；timeout保持timeout。本轮只检验锁定的η13→η10，主500步，次300/400步，没有看到结果后换主时间表。

预注册原种子2026160000..0063与round4 loco_confirmation64状态全部重叠，原文件和失败审计完整保留；没有用这些状态作前瞻终点。在任何新rollout/终点读取之前，PREREGISTRATION_V2重新锁定2026190000..0063并留UTC/SHA，时间表、臂、阈值全部不变。修订64个状态的全DB UID/content和可恢复物理坐标均无重叠。旧DB缺物理坐标的记录在审计中单列，不能把它们假称已完成绝对坐标核验。新结果在全cohort结束、ENDPOINT_OPENING.receipt写出后才由merger/分析打开，WATCH_HISTORY只有文件数量/资源，没有中期结局分析。

| 组 | N controller | S500−oracle平均pp | 跨state/controller bootstrap95%未知稳健pp | >5pp亏损controller | E3 vsF10 | 点门 | 未知稳健门 |
|---|---|---|---|---|---|---|---|
"""
    for name,o in g.items():text+=f"| {name} | {o['controller_count']} | {100*o['mean_gain']:.2f} | {[100*x for x in o['hierarchical_CI_unknown_envelope']]} | {o['deficit_controllers_gt5pp']} | {o['point_E3']} | {o['point_gate_pass']} | {o['status']} |\n"
    text+='\nHeld-out门按要求只强制整体正收益和逐controller5pp点门，另报告E3；all/dev门同时要求E3=0。控制器等权，state64成簇、controller层独立重抽，10000次，oracle在每次bootstrap中重新取max。逐controller CI只重抽state。点率为known-success下界；结论采用unknown可行范围检查，不猜填unknown。max oracle仅限F10/F13及三项预注册Freg，不是完整K16最优η。\n\n'
    text+='G_universal的FAIL不依赖v3的unknown：v9的必要点门已经失败。H_schedule保留INCONCLUSIVE是遵守预注册的结论映射：dev=FAIL、held-out=未知可翻转的混合情形，不事后重写成“两组都不通过”。这不表示本批数据的普适门可能通过，也不把点估计低5.08pp误写成总体真实损失必然超过5pp的95%证明；v9的配对CI为−8.59至−1.95pp，跨越−5pp边界。\n\n'
    text+='''| controller | 组 | oracle固定臂 | F10 | F13 | Freg | S500 | S500−oracle pp / 95%未知稳健pp |
|---|---|---|---|---|---|---|---|
'''
    def fmt(o):return '—' if o is None else f"{100*o['success_lower']:.2f}–{100*o['success_upper']:.2f}%"
    for n,o in r['controllers'].items():
        a=o['arms'];s=o['comparisons']['S500'];text+=f"| {n} | {o['group']} | {o['oracle_point_arm']} | {fmt(a.get('F10'))} | {fmt(a.get('F13'))} | {fmt(a.get('Freg'))} | {fmt(a['S500'])} | {100*s['gain_vs_oracle_point']:.2f} / {[100*x for x in s['CI_unknown_envelope']]} |\n"
    text+='\n![14 Flow arm comparison](ARM_COMPARISON_all.png)\n\n![held-out Flow comparison](ARM_COMPARISON_held_out.png)\n\nS300/S400为次要臂，完整同样门指标保存在results.group_metrics，不能替代失败的S500主终点。\n\n'
    for arm in ('S300','S400'):
        o=r['group_metrics'][arm];text+=f"- {arm}：all={o['all']['status']}，held-out={o['held_out']['status']}，all平均增益{100*o['all']['mean_gain']:.2f}pp；held-out平均{100*o['held_out']['mean_gain']:.2f}pp。\n"
    corr=r['secondary_regime_association'];text+=f"\n时间表增益与F10−F13的Pearson={corr['Pearson']}，CI={corr['Pearson_CI']}；Spearman={corr['Spearman']}，CI={corr['Spearman_CI']}。这是预定14点相关描述，不能证明regime由某个失败阶段造成；unknown点下界和小controller样本的限制保留。\n\n"
    text+='''| controller / arm | 结局计数 | success/timeout等真实发生步数摘要 |
|---|---|---|
'''
    for key,o in r['outcome_summaries'].items():
        summaries={name:dict(n=len(values),min=min(values),median=float(np.median(values)),max=max(values)) for name,values in o['occurrence_steps'].items() if values};text+=f"| {key} | {o['outcomes']} | {summaries} |\n"
    text+='\n逐seed的完整step数组、配对F10/F13结局转移见NUMERICAL_RESULTS/results.outcome_summaries。数值异常用原error step，成功按simulator first_success_step，timeout按原真实终止步；没有新增死锁判定。\n\n'
    if r['H_schedule']=='SUPPORTED':text+='解释（DEVELOPMENT）：这支持固定全程η的动作空间约束是Ring上controller regime差异的一个来源；已冻结、无controller输入的S500在本轮新状态/held-out时间表组通过注册门。尚未检验其他地图、N、未纳入的Flow或所有η，不能把有限14个controller上的5pp点门变成任意controller的定理。后续Q(s,t,η)/中途选择属于下一轮候选，需另行验证。\n\n'
    else:text+='解释（DEVELOPMENT）：本轮没有建立S500对所有14个Flow的通用不劣；失败控制器和门余量见表。若dev通过而held-out失败，属于预注册的部分过拟合门模式；仅因均值CI宽而未通过，并不证明某个机制或所有时间表都不可行。不能用次要时刻救回主命题，也不能宣称固定η动作空间已解释全部controller依赖。\n\n'
    text+='视频适用：本轮是Ring大型评价，明确适用视频要求；没有使用纯Q豁免。videos.py在新目录中仅把统一入口场景范围缩到用户要求的ring_exchange，保留原roles/trace验证/renderer源码和完整帧数检查。已检索历史视频manifest，未发现有已记录真实deadlock criterion的兼容Ring案例；本次冻结runtime也无deadlock event。统一严格入口返回非零，VIDEO_GATE和videos/manifest.complete=false完整记录缺口。\n\n'
    for video in v['videos']:
        text+=f"- [{video['role']}]({Path(video['file']).relative_to(HERE)})：{video['frames']}帧，dt={video['dt']}，video SHA={video['sha256']}；formal_role={video['formal_role']}。\n"
    text+='\n两个timeout-to-success视频若存在，只是全程真实比较，清楚标TIMEOUT，不填deadlock_show/deadlock_resolution。独立safety_success保留simulator连续运动净余量、完整augmented初态（原true-t0 reset精确hash重建，零仿真步）、seed/config/checkpoint/trace/video来源。缺死锁案例意味着batch视频交付未完成；没有追加rollout来制造案例或更改阈值。\n\n'
    text+='执行：round9原run.py/core数值实现和FF控制链复用，run.py仅隔离IO并转换journal numpy。14纯固定profiles和已有dev schedule UIDs沿用，其他10 schedule按相同η hex/time/code fingerprints生成。先design/preflight，32gatePASS再提交主missing；worker仅journal，官方单进程merger以局部exact现代seed/timestep适配，postflight核验全部raw/DB/trace。只写本实验目录和明确授权的新DB/本experiment journals生命周期，不修改历史资产、不commit。资源用户任务CPU<=18/GPU<=3，17workers+1编排；ming任务未计入zhihan额度。\n'
    from shared_rollout_db.src.rollout_db import connect,transaction,canonical
    with connect() as db,transaction(db):
        old=db.execute('SELECT metadata_json FROM experiment WHERE experiment_uid=?',(EXP,)).fetchone();metadata=json.loads(old[0]);metadata.update(numerical_complete=True,batch_delivery_complete=bool(v['complete']),video_complete=bool(v['complete']),missing_video_roles=v['missing_roles'],H_schedule=r['H_schedule'],end_time_semantics='numerical dataset completion; video delivery separately gated');db.execute('UPDATE experiment SET metadata_json=? WHERE experiment_uid=?',(canonical(metadata),EXP))
    (HERE/'REPORT.md').write_text(text)
    (HERE/'HANDOFF.md').write_text(f"""# Round10 independent handoff

Status: {r['status']}. H_schedule={r['H_schedule']}, overallG={r['G_universal']}, held-out={g['held_out']['status']}. Read REPORT.md first, results.json second. Preregistered new-state endpoint is prospective; mechanism/causal interpretation DEVELOPMENT. Numeric18688+gate32 completed; unknown {post['unknown']} preserved; video gap {v['missing_roles']} is real and prevents complete batch marker.

Root {HERE}; Python {PY}; DB experiment {EXP}. Mainjob20112,17workers. Fixed Ring/FF/H700/dt.05/x64false; fixed profiles14; mainS500 eta13->eta10, secondaryS300/S400, F10/F13 plus Freg(v3/plus_w_v3 eta2,v8 eta4). Four dev names in protocol, other10 held. Q4 not B15. Oracle max of prespecified fixed menu, never fullK16.

Original prereg/receipt and overlapping FRESH_STATES/STATE_DEDUP_AUDIT remain unchanged. Active PREREGISTRATION_V2 with SHA receipt precedes all rollouts/endpoints and fixes unused seeds2026190000..0063. PROSPECTIVE_FRESH_STATES/STATE_DEDUP_AUDIT_V2 preserve new64 and allDB UID/content/full recoverable coordinate audit; missing legacy coordinate provenance explicitly unresolved. No result-dependent redraw or tuning. ENDPOINT_OPENING.receipt locks complete cohort before merger/label access; monitor reads filenames only.

core.py/run.py/protocol are frozen (hash asserted by round9 loop), do not edit during rollout. Imported original round9 run.py supplies every numerical action/step. design.py/register officialpreflight; submit.py own-user CPU allocation; gate.py exact32 checks round9+round4; merge.py official single entry plus exact modern seed adapter; postflight.py full integrity; asset_audit.py inherited SHA; analyze.py crossed state/controller bootstrap10000 with reselected oracle; videos.py official Ring scope + complete-frame MP4s + strict nonzero gap; report.py report; finalize.py serial lifecycle and preserves opening receipt on retry.

Full main/raw and main/traces; gate/raw/traces; original shared journals merged singleprocess. Numeric results LABELS.npz, STATISTICAL_RESULTS,NUMERICAL_RESULTS/results; videos/manifest hashes MP4s and traces/augmented initial states. Formal deadlock roles remain missing; never rename timeout. FINALIZE_STAGE/LOG show all steps, NUMERICAL_COMPLETE.receipt means numeric only, COMPLETE must not exist while video gate fails. Expected finalizer exit2 is honest video gap, not numeric failure; unexpected exceptions leave FINALIZE_ERROR.

Reanalysis only after registered endpoint receipt and PASS postflight: analyze.py then videos.py (returns2 for missing deadlocks) then report.py. No re-executing unknown, no new rollout unless separately authorized for a real evidence gap. No detector, physics, controller, timing or threshold edits. Scripts and raw are all inside this directory; DB/journal writes are authorized lifecycle exceptions. Do not git commit/delete historical artifacts.\n""")
    print(json.dumps(dict(status=r['status'],H_schedule=r['H_schedule'],held_out=g['held_out']['status'],video_complete=v['complete'])))
if __name__=='__main__':main()
