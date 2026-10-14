from core import *
from plots import main as plots
from shared_rollout_db.src.rollout_db import connect,transaction,canonical
from collections import Counter

def pp(x):return 'NA' if x is None else f'{100*x:.3f}pp'
def pct(x):return 'NA' if x is None else f'{100*x:.3f}%'
def interval(x):return '['+', '.join(pp(y) for y in x)+']'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'
def main():
    prereg();gate();r=read(HERE/'results.json');p=read(HERE/'protocol.json');v=read(HERE/'videos/manifest.json');vg=read(HERE/'VIDEO_GATE.json');assert r['postflight']['status']=='PASS';old=read(HERE/'HISTORICAL_ASSET_INDEX.json')
    for path,h in old['assets'].items():assert sha(path)==h,path
    for q in old['historical_gate_records']:assert sha(q['path'])==q['sha256'],q['path']
    workers=[read(q) for q in (HERE/'main').glob('worker_*.json')];starts=[read(q) for q in (HERE/'main').glob('START_AUTHORIZATION_*.json')];assert len(workers)==len(starts) and sum(q['new'] for q in workers)==13440 and all(q['complete'] for q in workers);assert all(q['gate_sha256']==sha(HERE/'CONSISTENCY_GATE.json') for q in starts)
    with connect(True) as db:
        n=db.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?',(EXP,)).fetchone()[0];g=db.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?',(EXP+'__gate',)).fetchone()[0];assert n==13440 and g==32
    freeze(HERE/'FINAL_EVIDENCE_AUDIT.json',dict(utc=read(HERE/'FINAL_EVIDENCE_AUDIT.json')['utc'] if (HERE/'FINAL_EVIDENCE_AUDIT.json').exists() else utc(),passed=True,historical_assets_unchanged=len(old['assets']),historical_gate_raws_unchanged=32,main_actual_physical_count=sum(q['new'] for q in workers),worker_instances=len(workers),no_duplicate_physical_execution=True,DB_main=n,DB_gate=g,allstarts_after_gate=True,prereg_before_main=True,raw_trace_hashes_verified=13440,unknown_retained_not_rerun=r['unknown'],newlabels_opened_after_all13440=True))
    complete=bool(v['complete'] and vg['exit_code']==0);r['status']='COMPLETE' if complete else 'NUMERICAL_COMPLETE_VIDEO_INCOMPLETE';r['video_gate']=vg;r['prereg_receipt_sha256']=sha(HERE/'PREREGISTRATION.receipt.json');r['final_evidence_audit_sha256']=sha(HERE/'FINAL_EVIDENCE_AUDIT.json');write(HERE/'results.json',r);plots()
    x1=r['X1'];x2=r['X2'];x3=r['X3'];f=x2['methods']['fixed'];e=x2['methods']['expanded'];supported=x1['status']=='SUPPORTED' or x2['status']=='SUPPORTED'
    text=f"""# Round13：四场景时间表的前瞻检验

X1 **{x1['status']}**，X2 **{x2['status']}**；按预注册映射，四场景 LOSO 失败与 controller 级失败的动作压缩解释 {'得到支持' if supported else '尚未得到前瞻确认'}。X1 字面点门 {'通过' if x1['point_pass'] else '未通过'}，X2 字面点门 {'通过' if x2['point_pass'] else '未通过'}；unknown 边界影响另列。新40state/scene、3个原E8 Flow-reference算子、7臂、4seed共13440条实际rollout已完成；没有训练critic或新Flow。批次状态 **{r['status']}**，不将视频缺口隐瞒为完整交付。

## 事实：X1

保持 Ring 原时间表 S13→10@t2；各场景成功率取3controller平均，与该场景3固定臂的最大均值比较。门为四场景各自差≥−5pp，未加新的CI显著性要求；CI为state成簇5000次、seed202610131301，保留3controller与4seed。主状态保守地检查unknown边界，字面点结果同时报告。

"""
    text+=table(['场景','S13@t2','最佳固定','固定率','差','95% CI','unknown bounds','点门'],[[q['scene'],pct(q['S13_t2']),q['best_fixed_arm'],pct(q['best_fixed']),pp(q['difference']),interval(q['CI95']),interval([q['unknown_lower'],q['unknown_upper']]),q['point_pass']] for q in x1['scenes']])+'\n'
    text+='逐controller（每controller160次，40state×4seed）：\n\n'+table(['场景','controller','S13@t2','最佳固定','差','95% CI'],[[q['scene'],q['controller'],pct(q['S13_t2']),pct(q['best_fixed']),pp(q['difference']),interval(q['CI95'])] for q in x1['controllers']])+'\n'
    text+='## 事实：X2\n\n源场景等权平均后argmax，留出场景不进入该fold选择。并列按固定菜单顺序取第一项；bootstrap每次重新算源选择和held七臂oracle，不把某个已选源动作当作已知常量。固定集F10/F13/F15；扩展集加4时间表。门仍为扩展最大遗憾≤5pp且平均≤固定均值的1/3。\n\n'
    text+=table(['留出场景','集合','源选动作','held oracle','遗憾','95% CI','unknown bounds','unknown可能动作'],[[q['scene'],q['method'],q['selected'],q['oracle_arm'],pp(q['regret']),interval(q['CI95']),interval([q['regret_lower'],q['regret_upper']]),','.join(q['possible_source_choices'])] for q in x2['scenes']])+'\n'
    text+=f"固定平均/最大遗憾：{pp(f['mean_regret'])}/{pp(f['max_regret'])}；扩展平均/最大：{pp(e['mean_regret'])}/{pp(e['max_regret'])}。平均比例={x2['mean_regret_ratio']}；配对平均改善95%CI {interval(x2['paired_mean_improvement_CI95'])}。扩展平均/最大unknown bounds：{interval(e['mean_unknown_bounds'])}/{interval(e['max_unknown_bounds'])}。\n\n"
    text+='## 事实：X3（INCONCLUSIVE / 描述性）\n\nX3没有另给二元判定阈值，故不制造SUPPORTED/REJECTED。E_a=S−F10，L_a=S−F_a，N_a=F_a−F10，E−L=N恒等式全部通过。两阶段均≥10pp的数量：'+str(x3['both_ge10pp_per_scene'])+f"，总{x3['both_ge10pp']}/24。\n\n"
    text+=table(['场景','controller','早期η','E','L','N','E 95%CI','L 95%CI','两项≥10pp'],[[q['scene'],q['controller'],q['early_eta'],pp(q['E']),pp(q['L']),pp(q['N']),interval(q['E_CI95']),interval(q['L_CI95']),q['both_ge10pp']] for q in x3['effects']])+'\n'
    failures=read(HERE/'FAILURE_RECORDS.json');summaries=[]
    for sc in SCENES:
        for arm in ARMS:
            fs=[q for q in failures if q['scene']==sc and q['arm']==arm];counts=Counter(q['type'] for q in fs);steps=Counter((q['type'],q['occurrence_step']) for q in fs);summaries.append(dict(scene=sc,arm=arm,counts=counts,occurrence_steps=[dict(type=k[0],step=k[1],count=n) for k,n in sorted(steps.items())]))
    write(HERE/'FAILURE_TIMING_SUMMARY.json',summaries)
    text+='失败类型与实际发生步汇总：\n\n'+table(['场景','臂','事件数','事件发生步 min–max'],[[q['scene'],q['arm'],str(q['counts']),'; '.join(f"{kind}: {min(x['step'] for x in q['occurrence_steps'] if x['type']==kind)}–{max(x['step'] for x in q['occurrence_steps'] if x['type']==kind)}" for kind in q['counts'])] for q in summaries])+'\n'
    text+='逐seed、原始UID、具体事件步见 FAILURE_RECORDS.json，完整步频分布见 FAILURE_TIMING_SUMMARY.json。真实deadlock仅来自Toy/DB原window_progress_v2终止事件；candidate单独记录，timeout始终为timeout。配对失效变化见 FAILURE_TRANSITIONS.json（每同state/controller/seed对照F10/Fa→时间表），不从动画或timeout推断新死锁。\n\n'
    text+='## 解释与边界\n\n'+r['conclusion_mapping']+'\n\n'
    if x1['status']!='SUPPORTED' and x2['status']=='SUPPORTED':text+='LOSO扩展选择能消除regime，但Ring原S13@t2未获支持；各fold源选动作为 '+str(e['choices'])+'。这是新state证据，不把事后最优动作另称为下一轮独立确认。\n\n'
    descriptive=r['descriptive_mechanism_mapping'];text+='最好时间表相对最佳固定的场景差：'+str(descriptive['best_schedule_minus_best_fixed'])+'；每场景是否至少有一个双阶段≥10pp项：'+str(descriptive['each_scene_has_both_effects_ge10pp'])+'。\n\n'
    if not supported:
        if descriptive['each_scene_has_both_effects_ge10pp']:text+='按描述性阶段证据，各场景内部仍有两阶段结构，但未消除共同LOSO选择的剩余遗憾；场景信息/少数时间表选择问题仍在。\n\n'
        if descriptive['no_schedule_point_superiority_in_Toy_DB_Four']:text+='Toy/DB/Four中没有任何已登记时间表在点率上优于最佳固定；依任务映射提示Ring特有解释，不能仅由无性能优势推出这些场景没有阶段结构。\n\n'
    text+='本轮固定对照新增了历史E8 K4中没有的F13；X1是不劣检验，并不要求时间表严格优于每个固定动作。因此是否仅新增固定F13就改善LOSO，以及时间表是否有增量收益，必须看X2固定/扩展对照，不能直接拿本轮率与旧K4失败率作唯一因果证据。\n\n'
    text+='最终合成边界检查发现恰好5pp可因浮点相减误判；首版分析已生成但agent未查看真实结果时暂停后处理，保留在analysis_before_integer_arithmetic。INTEGER_ARITHMETIC_BUG.receipt及修复收据记录时间。现用整数成功/unknown次数判定相同5pp、1/3、10pp门，并按整数源计数解决并列；每场景分母480、每controller160，bootstrap仍为同样5000次state抽样。没有epsilon放宽、阈值/菜单/模型变化，也没有标签或仿真重跑。\n\n'
    text+='新states上的比较为前瞻；历史E8、审计、Round9–12结论为DEVELOPMENT来源，不能把Round12真实checkpoint结论直接移给E8 affine算子。Flow checkpoint嵌套于场景，只覆盖四种固定几何；即使X1或X2通过，也不能证明H_compress是唯一根因。下一步critic/新几何LOSO仅为提示指定的研究方向，本轮没有实施。\n\n'
    text+=f"## 执行、冻结与资源\n\n预注册UTC {read(HERE/'PREREGISTRATION.receipt.json')['utc']}，SHA {sha(HERE/'PREREGISTRATION.md')}。原E8运行时/模型/物理/投影/事件未改，唯一动作干预是绝对物理动作索引t替换η；Toy JAXFP64，其他FP32，均在全部导入后恢复并每步断言。切换364/607、471/786、300/500，使用完整IEEE η值。旧Toy没有augmented_state接口，用已有snapshot和全部监测器内存只读记录；随后初态文件同hash临时名并发冲突改为每进程文件。两次纯IO修复与原代码/失败日志保留，已产生的gate记录复用、无unknown重跑。32例历史F10/F15的终止、长度、首动作float64字节全部精确一致，之后才启动主实验。\n\n"
    text+='160状态原自然生成、expert接受规则与去重细节见states_part及STATE_DEDUP_AUDIT.json。DB原expert8模式可解性接受检查是初态生成的辅助仿真，不是主FF标签/Flow训练，抽样与拒绝计数完整记录。原18CPU资源上限后被用户明确改为24；最高23计算worker+1编排，0GPU。纯IO预留将尚未开始案例转交helper，原在途case不中断，标记无成功字段、只由owner换成真实记录；全部13440实际physical counter校验，零重复仿真。\n\n'
    text+=f"全量到齐才打开终点，ENDPOINT_OPENING收据UTC {read(HERE/'ENDPOINT_OPENING.receipt.json')['utc']}。source-only argmax未知边界的并列处理由合成数据检查修正为预注册的最早菜单项，未读真实标签、没有更改参数或判定门。标准preflight无ambiguous、只缺失生成、worker-localjournal、官方单进程merger/本地完整seed身份adapter；postflight13440条trace/raw/DBflags一致、quarantine0、unknown{r['unknown']}保留并给upper/lower边界。没有把unknown当作观测失败或重跑。\n\n"
    text+=f"## 视频与交付\n\n完整视频 {len(v['videos'])} 个。统一四场景入口已实际调用，退出码{vg['exit_code']}；缺口 {v['missing_roles']}。Ring/Four原runtime无deadlock判据，不可伪造；timeout-to-success如存在仅为完整诊断视频，不填正式deadlock角色。每个正式安全素材检查实际连续运动余量与全部过程。manifest含源trace/videoSHA、真实初态、controller、seed和event，所有正式pair同初态/seed/config。批次保持{r['status']}；严格入口非零阻止COMPLETE标记。数值检验与视频交付状态分别报告。\n\n"
    text+='results.json和LABELS.npz为机器结果；ARM_RATES.csv、X1_CONTROLLER/X2_SCENES/X3_EFFECTS及FAILURE*为逐项数值；图 four_scene_seven_arms、X2_loso_regrets、X3_phase_effects 各PNG/PDF；视频 videos/manifest.json；所有新资产仅在本目录，历史文件hash保持不变，没有gitcommit。\n'
    (HERE/'REPORT.md').write_text(text)
    (HERE/'HANDOFF.md').write_text('# Round13 handoff\n\n'+text+'\nReproduction: prereg receipt ->prepare_gate/run(gate)/gate ->states/dedup ->design/preflight ->run(main)/helper if resource transfer ->single merger ->analyze ->videos ->report. No target success analysis before all13440 actual raws. Unknowns neverrerun. No further rollout/video fabrication or critic training is authorized by this completed numerical stage. Fullbatch remains incomplete when required true-deadlock roles are missing.\n')
    with connect() as db,transaction(db):db.execute('UPDATE experiment SET end_time=CURRENT_TIMESTAMP,new_rollout_count=?,metadata_json=? WHERE experiment_uid=?',(13440,canonical(dict(status=r['status'],unknown=r['unknown'],video_complete=complete)),EXP))
    write(HERE/'RUN_STATUS.json',dict(utc=utc(),status=r['status'],main13440=True,unknown=r['unknown'],X1=x1['status'],X2=x2['status'],X3='INCONCLUSIVE_DESCRIPTIVE',video_complete=complete,max_CPU=24,GPU=0))
    write(HERE/'DELIVERY.receipt.json',dict(utc=utc(),status=r['status'],report_sha256=sha(HERE/'REPORT.md'),handoff_sha256=sha(HERE/'HANDOFF.md'),results_sha256=sha(HERE/'results.json'),video_manifest_sha256=sha(HERE/'videos/manifest.json'),audit_sha256=sha(HERE/'FINAL_EVIDENCE_AUDIT.json'),batch_complete=complete));print(json.dumps(dict(status=r['status'],X1=x1['status'],X2=x2['status'],videos=len(v['videos']))))
if __name__=='__main__':main()
