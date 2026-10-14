"""Final read-only evidence audit and report; no extra physical execution."""
from core import *
from collections import Counter
from shared_rollout_db.src.rollout_db import connect,transaction,canonical

def pp(x):return 'NA' if x is None else f'{100*x:.3f}pp'
def interval(x):return '['+', '.join(pp(v) for v in x)+']'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    prereg();labels_allowed();gate();r=read(HERE/'results.json');p=read(HERE/'protocol.json');video=read(HERE/'VIDEO_GATE.json');manifest=read(HERE/'videos/manifest.json');assert r['postflight']['requests']==12288
    old=read(HERE/'HISTORICAL_ASSET_INDEX.json');immutable=[]
    for q in old['assets']:
        path=Path(q['path']);path=path if path.is_absolute() else ROOT/path;assert sha(path)==q['sha256'],str(path);immutable.append(q)
    for path,h in old['core_checkpoint_component_hashes'].items():
        assert sha(path)==h,path;immutable.append(dict(path=path,sha256=h))
    incident=read(HERE/'INVALID_PRECISION_EVIDENCE.json')
    for q in incident['files']:assert sha(q['path'])==q['sha256'],q['path']
    receipt=read(HERE/'SELECTOR_FROZEN.receipt.json');rf=read(HERE/'R5_FROZEN.receipt.json');assert rf['utc']<receipt['utc']
    starts=[read(x) for x in sorted((HERE/'main').glob('START_AUTHORIZATION_*.json'))];assert len(starts)>=23 and all(x['utc']>receipt['utc'] and x['selector_receipt_sha256']==sha(HERE/'SELECTOR_FROZEN.receipt.json') and x['probe_receipt_sha256']==sha(HERE/'R5_FROZEN.receipt.json') for x in starts)
    excluded=read(PARENT/'SEMANTIC_CONTROLLER_NOVELTY_AUDIT.json')['already_evaluated_identical_best_policies'];assert len(p['controller_order'])==8 and not any(n+'__final' in p['controller_order'] for n in excluded)
    workers=[read(q) for q in sorted((HERE/'main').glob('worker_*.json'))];assert len(workers)==len(starts) and sum(x['new'] for x in workers)==12288
    assert all(x['complete'] for x in workers)
    earliest=min(q.stat().st_mtime for q in (HERE/'main/raw').glob('*.json'));assert datetime.fromtimestamp(earliest,timezone.utc).isoformat()>receipt['utc']
    with connect(True) as db:
        rows=[dict(x) for x in db.execute('SELECT outcome,count(*) n FROM rollout WHERE experiment_uid=? GROUP BY outcome',(EXP,))];assert sum(x['n'] for x in rows)==12320
        assert db.execute('SELECT count(*) FROM rollout WHERE experiment_uid=? AND conflict_quarantined!=0',(EXP,)).fetchone()[0]==0
        assert db.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?',(EXP+'_unlabeled',)).fetchone()[0]==0
    audit=dict(utc=utc(),passed=True,historical_assets_verified=len(immutable),invalid_precision_files_preserved=len(incident['files']),original_core_controller_physics_unchanged=True,source_fit_equivalence=read(HERE/'SOURCE_FIT_EQUIVALENCE_AUDIT.json')['passed'],eligible_final_count=8,excluded_count=6,main=12288,gate=32,DB_rollout_rows=12320,DB_outcomes=rows,corrected_probes=256,probe_success_rows=0,selector_freeze_before_every_worker_start=True,selector_freeze_before_earliest_final_raw=True,earliest_final_raw_utc=datetime.fromtimestamp(earliest,timezone.utc).isoformat(),invalid_target_precision_evidence_used=False)
    write(HERE/'FINAL_EVIDENCE_AUDIT.json',audit)
    sys.path.insert(0,str(R9/'vendor'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import csv
    with (HERE/'ARM_RATES.csv').open() as f:ars=list(csv.DictReader(f))
    fig,axs=plt.subplots(4,2,figsize=(13,15),sharey=True)
    for ax,n in zip(axs.flat,p['controller_order']):
        items=[x for x in ars if x['controller']==n];ys=np.array([float(x['success_lower']) for x in items])*100;low=np.array([float(x['CI95_lower']) for x in items])*100;high=np.array([float(x['CI95_upper']) for x in items])*100;ax.bar(p['arms'],ys,color=['#4979a7','#bf6574','#84a59d','#5b9f7b','#bb985a','#9176a8']);ax.vlines(np.arange(6),low,high,color='black');ax.plot(np.arange(6),low,'_',color='black');ax.plot(np.arange(6),high,'_',color='black');ax.set_title(n.replace('_macflow__final',''));ax.set_ylim(0,104);ax.set_ylabel('Success lower point (%)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Round12: eight distinct final policies, six arms\n95% state-cluster CI; unknowns retained separately');fig.tight_layout();fig.savefig(HERE/'six_arm_success.png',dpi=180);fig.savefig(HERE/'six_arm_success.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,scheme in zip(axs,['primary','secondary']):
        m=r['prediction_metrics'][scheme];x=np.array(m['actual_E'])*100;y=np.array(m['predictions'])*100;ax.scatter(x,y,color='#4979a7');span=[min(x.min(),y.min(),0)-2,max(x.max(),y.max())+2];ax.plot(span,span,'--',color='gray');ax.set(xlabel='Actual E (pp)',ylabel='Frozen Ehat (pp)',title=f"{scheme}: rho={m['Spearman']:.3f}, RMSE ratio={m['RMSE_ratio']:.3f}" if m['Spearman'] is not None and m['RMSE_ratio'] is not None else scheme)
        for i,n in enumerate(p['controller_order']):
            label=n.replace('base_u_','').replace('_macflow__final','').replace('plus_w_v4_train_targeted','w4-targeted').replace('plus_w_','w-').replace('_local','L')
            right=x[i]>40;offset=(-4,-12) if 'targeted' in label else (4,-12) if label=='v8L' else (-4,5) if right else (4,5)
            ax.annotate(label,(x[i],y[i]),fontsize=8,xytext=offset,textcoords='offset points',ha='right' if right else 'left')
        ax.set_xlim(span);ax.set_ylim(span)
        ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(HERE/'Ehat_vs_E.png',dpi=180);fig.savefig(HERE/'Ehat_vs_E.pdf');plt.close(fig)
    main=r['methods']['primary'];pm=r['prediction_metrics']['primary'];sec=r['methods']['secondary'];c4=r['C4'];models=read(HERE/'SELECTOR_MODELS.json');choices={x['controller']:x for x in models['choices']};dist={x['controller']:x for x in read(HERE/'SIBLING_FEATURE_DISTANCES.json')};compat={x['controller']:x for x in read(HERE/'COMPATIBILITY_AUDIT.json')['checks']};nov=read(PARENT/'SEMANTIC_CONTROLLER_NOVELTY_AUDIT.json');novmap={x['run']:x for x in nov['checks']}
    complete=bool(video['exit_code']==0 and manifest['complete']);r['status']='COMPLETE' if complete else 'NUMERICAL_COMPLETE_VIDEO_INCOMPLETE';r['video_gate']=video;r['confirmation_cohort_n']=8;r['excluded_parameter_identical_finals']=excluded;r['precision_incident']=dict(old_target_descriptors_and_choices_invalid=True,preserved_manifest_sha256=sha(HERE/'INVALID_PRECISION_EVIDENCE.json'),corrected_FP32=True,consistency_gate_passed=True);r['final_evidence_audit_sha256']=sha(HERE/'FINAL_EVIDENCE_AUDIT.json');r['pre_registration_addendum_sha256']=sha(PARENT/'PREREGISTRATION_ADDENDUM_N8_FP32.md');write(HERE/'results.json',r)
    text=f"""# Round12：8 个新参数 final 的前瞻确认

C1 **{r['C1']}**（字面点估计门 {'通过' if main['C1_point_pass'] else '未通过'}）；C3 **{r['C3']}**。C2 的 headroom 门 {'满足' if main['headroom_point_pass'] else '不满足'}：{main['headroom_count']}/8 个真实下界点 E<5pp（unknown 可取范围 {main['headroom_count_unknown_range']}），C2 **{r['C2']}**。这次只确认 8 个参数不同的 final；6 个参数完全相同的 final 排除。全部 12,288 条主 rollout 与 256 次修复后无标签探测已完成。批次状态 **{r['status']}**；视频严格门未通过时不宣称整批交付完成。

## 固定门槛与结果

主方案每个 final 排除同 run 的 best，使用 13 个历史 best；次方案 14 源仅作敏感性分析。155D R5、源侧均值填补/zscore/PCA2、ridge α=1、Ê<5pp 选 F10 否则 S500 均未改变。5000 次共享 state 成簇 bootstrap、64 state、每簇保留 4 seeds，不重采样 controller。

主 G_E 平均遗憾 {pp(main['mean_regret'])}，95% CI {interval(main['mean_regret_CI95'])}；unknown bounds {interval(main['mean_regret_unknown_bounds'])}。最大遗憾 {pp(main['max_regret'])}；C1b **{r['C1b']}**。相对 F10 平均收益 {pp(main['mean_gain_vs_F10'])}，95% CI {interval(main['gain_vs_F10_CI95'])}，unknown envelope {interval(main['gain_vs_F10_CI95_unknown_envelope'])}。C1 仍要求该下界>0、平均遗憾≤2pp、每 controller 损失≤10pp；n=8 不放宽。

C2 相对固定 S500 的配对改善 {pp(main['mean_gain_vs_source_average'])}，95% CI {interval(main['gain_vs_source_average_CI95'])}；{main['C2_reason']}。菜单内含 F10/F13/S300/S500，F2/F4 仅参加六臂 oracle。

C3 主 Spearman={pm['Spearman']}（门≥0.6），RMSE={pp(pm['RMSE'])}，13 源均值基线 RMSE={pp(pm['source_mean_baseline_RMSE'])}，比值={pm['RMSE_ratio']}（门≤0.7）。次方案 C1={sec['C1']}，C2={sec['C2']}，C3={r['prediction_metrics']['secondary']['C3']}；不替换主方案。n 从14降为8使 controller 关系估计和 CI 的精度降低，预期不确定性更宽；各实际 state CI 仍由数据决定。没有因样本数降低放宽任何门槛。

C4 **INCONCLUSIVE / 描述性**：E>0 {c4['E_positive']}/8，L>0 {c4['L_positive']}/8，E和L均≥10pp {c4['both_E_L_ge10pp']}/8。历史 best 参照分别11/14、14/14、10/14，必须比较分母/比例；无额外二元阈值。N=E−L 恒等式审计通过。

{r['conclusion_mapping']}

## 每个新 controller

"""
    rows=[]
    for x in main['choices']:
        n=x['controller'];e=r['effects'][n];rows.append([n,choices[n]['primary_action'],pp(choices[n]['primary_Ehat']),pp(e['E']['point']),pp(e['L']['point']),x['oracle_arm'],pp(x['regret']),pp(x['gain_vs_F10'])])
    text+=table(['controller','G_E','Ê','E','L','oracle','regret','gain vs F10'],rows)+'\n\n'
    text+='超出5pp遗憾的 controller：'+', '.join(main['regret_exceeds5pp_point'])+'。全部失败类型、发生动作步及原始 rollout UID 保存在 FAILURE_RECORDS.json；timeout 始终为 timeout，数值失败为 unknown，不当作已观测失败，也不重跑。\n\n'
    for x in main['choices']:
        if x['controller'] not in main['regret_exceeds5pp_point']:continue
        n=x['controller'];fails=read(HERE/'FAILURE_RECORDS.json').get(n,[]);counts=Counter((q['arm'],q['outcome'],q['error_type']) for q in fails);text+=f"{n}：六臂 oracle={x['oracle_arm']}，可描述为该臂偏好型；G_E={x['selected_arm']}，遗憾 {pp(x['regret'])}。事件汇总：{dict(counts)}。逐事件实际发生步请见 FAILURE_RECORDS.json，不将终止步推断为死锁发生步。\n\n"
    text+='## 兄弟差异与六个排除项\n\n'
    text+=table(['final','参数L2','相对L2','R5 raw L2','R5源标准化L2'],[[n,compat[n]['parameter_L2'],compat[n]['parameter_relative_L2'],dist[n]['R5_raw_L2'],dist[n]['R5_source_standardized_L2']] for n in p['controller_order']])+'\n\n'
    text+=table(['排除的run','best参数SHA256','final参数SHA256','排除原因'],[[n,novmap[n]['best_policy_parameter_sha256'],novmap[n]['final_policy_parameter_sha256'],'参数逐字节相同，已有best评估；不运行final'] for n in excluded])+'\n\n'
    text+='8 个纳入项与全部14历史best参数哈希均不同；未为六个排除 final 运行新的探测或主实验。原精度错误阶段对14 final的无标签探测作为事故证据保留，不能算作本轮有效确认。\n\n'
    text+='## 精度事故、拦截与冻结时序\n\n'
    text+='首次探测入口在导入前设置 FP32，但 poc/field_controller.py 导入时把 jax_enable_x64 改为 True。此前448个无标签探测、目标R5特征、Ê/G_E选择均无效，原始文件、protocol、收据完整保留，见 INVALID_PRECISION_EVIDENCE.json。初次一致性 worker 被 x64 断言在物理步之前拦截，未产生 final 成功标签。文件哈希不同的14 final也经参数审计纠正为8个真实新策略。\n\n'
    text+='用户批准后先冻结 N8/FP32 补充，然后在全部数值/controller导入之后显式恢复 FP32；检查网络参数与采样输出 float32，并在每次动作入口断言 x64False。模拟物理 numpy float64 和投影算法不变。8个零步兼容检查通过，256次重新探测统计独立重算通过；原固定源侧拟合参数逐值相同，仅修复目标特征与由其产生的预测/选择。重新冻结R5和选择之后运行历史best32例门，outcome、episode_length、first-action float64 bytes对Round9及Round4均精确相同；之后才运行8 final主实验。\n\n'
    receiptrows=[]
    for parent,n in [(PARENT,'PREREGISTRATION.receipt.json'),(PARENT,'PREREGISTRATION_ADDENDUM_N8_FP32.receipt.json'),(HERE,'R5_FROZEN.receipt.json'),(HERE,'SELECTOR_FROZEN.receipt.json'),(HERE,'ENDPOINT_OPENING.receipt.json')]:
        q=read(parent/n);receiptrows.append([str((parent/n).relative_to(PARENT)),q['utc'],sha(parent/n)])
    text+=table(['收据','UTC','收据SHA256'],receiptrows)+'\n\n'
    text+='每个 worker START_AUTHORIZATION 在新 selector 冻结之后；最早 final raw UTC='+audit['earliest_final_raw_utc']+'。主标签只在全部12288 raw到齐后打开，见 ENDPOINT_OPENING.receipt.json。原64状态逐字节复用并与原去重凭证相连，没有事后重抽。\n\n'
    text+=f"## 缓存、unknown 与视频\n\n主 preflight、32例 gate、单进程官方 merger/本地精确身份 adapter、主 POSTFLIGHT 均保存。主 unknown={r['unknown']}；每个 unknown保留原记录，不重跑、不零填成功标签；所有 trace/raw SHA及DB flags已核对，无quarantine。{len(immutable)}个历史资产和{len(incident['files'])}个错误精度证据文件hash保持不变；DB正确新实验共12320条（主12288+gate32），无标签probe成功表0行。\n\n"
    text+=f"统一视频入口已实际执行，退出码 {video['exit_code']}；完整视频 {len(manifest['videos'])} 个，缺失正式角色 {manifest['missing_roles']}。Ring原生runtime没有deadlock判据，不能把timeout冒充deadlock；两条timeout-to-success视频如存在，仅为完整诊断对照。safety素材验证来自实际连续运动碰撞检查。数值终点可报告，整批状态保持 {r['status']}，不生成完成标记。\n\n"
    text+='资源补充：用户授权无人使用时利用全部空闲CPU；计算17→23 worker，另留1CPU编排。RESOURCE_BOOST_PLAN/receipt记录明确非证据预留，仅转移尚未开始的尾部case；原在途case继续，不中止或重跑unknown。预留标记不含成功字段，原worker仅跳过，owner用相同冻结数值循环生成真实记录并原子替换；合并跳过经验证的操作标记，终点等待两组job完成。规模、物理、选择与门槛均未变。\n\n'
    text+='合并阶段曾因操作标记统计键未初始化报 KeyError，已按 MERGER_COUNTER_REPAIR.receipt.json 修复计数器，只重跑合并与后处理，仿真0条重跑。原失败日志保留；实际seed身份与标签解析规则未变。\n\n'
    text+='## 使用范围与文件\n\n这是Ring、同run兄弟checkpoint、全新64state上的前瞻数值确认，不是独立训练run或跨场景泛化；历史证据仍为DEVELOPMENT。未知结局的lower点及upper边界/CI全部报告，不能把lower点解释为所有unknown失败。图见 six_arm_success.png/pdf、Ehat_vs_E.png/pdf。机器结果 results.json；逐臂 ARM_RATES.csv；效应 E_L_N.csv；选择/遗憾 ACTIONS_REGRETS.csv；冻结模型 SELECTOR_MODELS.json；失败事件 FAILURE_RECORDS.json；视频 videos/manifest.json。\n'
    (HERE/'REPORT.md').write_text(text)
    (HERE/'HANDOFF.md').write_text('# Round12 FP32/N8 handoff\n\nActive directory: '+str(HERE)+'\n\n'+text.split('## 固定门槛')[0]+'\nRead REPORT.md and results.json for endpoints. Original parent phase is invalid for confirmation; do not resume old queued jobs or use its features/choices. Retained original prereg is superseded only by PREREGISTRATION_ADDENDUM_N8_FP32.md. Eligible list fixed in FINAL_BASE_PROFILES.json; six exclusions in semantic parameter audit. No training or additional rollout is needed. Unknowns must not be rerun. Reuse journals/DB only with exact semantic identities. All output writes restricted to parent experiment and explicit SQLite lifecycle.\n\n'+f'Numerical postflight PASS; strict video gate exit {video["exit_code"]}, missing {manifest["missing_roles"]}. No deadlock event exists in frozen Ring runtime. Do not fabricate roles or create COMPLETE marker. Source14 best, primary13 excluding brother, target8; original C1-C4/C1b/headroom thresholds unchanged. Corrected probe256, main12288, gate32. Reproduction: core.py guards prereg; probe.py/features receipt; models.py/selector receipt; design.py/cache; run.py binds immutable R9 loop and restores FP32; gate.py exact32bytes; merge.py official single merger; analyze.py frozen endpoints; videos.py unified Ring-scoped entry; report.py immutable/chronology audit.\n')
    with connect() as db,transaction(db):db.execute('UPDATE experiment SET end_time=CURRENT_TIMESTAMP,new_rollout_count=?,metadata_json=? WHERE experiment_uid=?',(12320,canonical(dict(status=r['status'],main12288=True,gate32=True,unknown=r['unknown'],video_complete=complete)),EXP))
    write(HERE/'DELIVERY.receipt.json',dict(utc=utc(),status=r['status'],results_sha256=sha(HERE/'results.json'),report_sha256=sha(HERE/'REPORT.md'),handoff_sha256=sha(HERE/'HANDOFF.md'),video_manifest_sha256=sha(HERE/'videos/manifest.json'),audit_sha256=sha(HERE/'FINAL_EVIDENCE_AUDIT.json'),batch_complete=complete))
    # Parent entrypoints point to corrected cohort and retain old reports as incident snapshots.
    for name in ['REPORT.md','HANDOFF.md','RUN_STATUS.json','results.json']:
        src=PARENT/name;snapshot=HERE/('INVALID_PARENT_'+name)
        if src.exists() and not snapshot.exists():snapshot.write_bytes(src.read_bytes())
    (PARENT/'REPORT.md').write_text(text+'\nActive evidence directory: fp32_n8/. All referenced numerical artifacts in that directory.\n')
    (PARENT/'HANDOFF.md').write_text((HERE/'HANDOFF.md').read_text())
    for name,obj in [('RUN_STATUS.json',dict(utc=utc(),status=r['status'],active_directory=str(HERE),main_requests=12288,final_success_labels_generated=True,valid_features=True,valid_selectors=True,original_precision_phase_invalid=True)),('results.json',r)]:
        (PARENT/name).write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=r['status'],C1=r['C1'],C2=r['C2'],C3=r['C3'],batch_complete=complete)))
if __name__=='__main__':main()
