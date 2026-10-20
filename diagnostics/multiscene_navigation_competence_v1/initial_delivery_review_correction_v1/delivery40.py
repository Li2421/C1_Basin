"""Final delivery from frozen evidence; never drives controllers or selection."""
from common import *
from deliver_failed_expert_pilot import NAMES,protected_status
from design_generated import ARMS
import csv

METHOD_NAMES=dict(nominal_only='仅执行导航策略',safety_baseline='只加安全投影',F10='全程使用第十组修正参数',F13='全程使用第十三组修正参数',F15='全程使用第十五组修正参数',S13='第十三组至第十组修正参数按比例切换',S15='第十五组至第十组修正参数按比例切换',Sprogress='第十三组至第十组修正参数按进度切换',right_hand='通用持续右手规则',zero_shot_selector='场景无关零样本选择模型',source_average_best='源场景平均最佳臂',four_pilot_schedule_right_hand='四个预实验状态选择时间表或持续右手规则',complete_menu_oracle='完整菜单样本内最佳参照',best_fixed_oracle='最佳固定参数样本内参照')
for k in [1,2,4,8,16]:
    METHOD_NAMES[f'fewshot_empirical_{k}']=f'每臂{k}条已知预实验标签的经验选择'
    METHOD_NAMES[f'fewshot_bayesian_{k}']=f'每臂{k}条已知预实验标签的贝叶斯选择'
STANDARD_UNAVAILABLE={'第十六轮冻结的控制器级评价选择模型（A3）':'原环形交换表示与当前场景无关三十三维输入不兼容，没有发明适配器','第十六轮冻结的不看控制器的评价模型（A0）':'原场景表示不兼容','探测估计后规则选择（G_E）':'旧环形交换探测表示和规则没有本轮兼容适配','预算匹配的多机器人专家':'旧联合可解证明使用一万步预算；没有本轮评估预算的匹配专家绩效'}

def compare_integrity():
    begin=read(HERE/'INTEGRITY_RESUME_BEGIN.json');end=read(HERE/'INTEGRITY_RESUME_END.json');peers=set(begin['parallel_metadata_only'])|set(end['parallel_metadata_only']);before=begin['sources'];after=end['sources'];changes=[]
    for p in sorted(set(before)|set(after)):
        if before.get(p)!=after.get(p):changes.append(dict(path=p,before=before.get(p),after=after.get(p)))
    def peer(path):
        try:return Path(path).relative_to(D).parts[0] in peers
        except ValueError:return False
    protected=[r for r in changes if not peer(r['path'])];sb=protected_status(begin['Basin']['status'],peers);se=protected_status(end['Basin']['status'],peers)
    passed=not protected and sb==se and begin['Basin']['head']==end['Basin']['head'] and begin['Basin']['diff_sha256']==end['Basin']['diff_sha256'] and begin['MAC']==end['MAC']
    result=dict(utc=utc(),passed=passed,baseline='INTEGRITY_RESUME_BEGIN.json',end='INTEGRITY_RESUME_END.json',protected_source_changes=protected,protected_status_equal=sb==se,Basin_HEAD_equal=begin['Basin']['head']==end['Basin']['head'],Basin_diff_equal=begin['Basin']['diff_sha256']==end['Basin']['diff_sha256'],MAC_equal=begin['MAC']==end['MAC'],parallel_changes=[r for r in changes if peer(r['path'])],parallel_metadata_begin=begin['parallel_metadata_only'],parallel_metadata_end=end['parallel_metadata_only'],prohibited_metadata_begin=begin['prohibited_metadata_only'],prohibited_metadata_end=end['prohibited_metadata_only'],prior_readonly_boundary_deviation='Preserved; user-authorized recovery, content not used, no prohibited writes')
    write(HERE/'INTEGRITY_FINAL_COMPARISON.json',result);return result

def expert_for(family):
    if family in 'BF':return read(HERE/'EXPERT_GATE_RESULTS.json')['families'][family]
    p=HERE/f'FULL_{family}_EXPERT_GATE_RESULTS.json';return read(p)['families'][family] if p.exists() else None

def oracle_report_metrics(family,key,record):
    """Required ancillary metrics for the already-defined posthoc oracle."""
    z=np.load(HERE/'selector_labels40'/f'{family}.npz');Y=z['success'];U=z['unknown'];C=z['collision'];T=z['timeout'];rates=Y.mean((1,3));indices=np.arange(2,5 if key=='best_fixed_oracle' else 9);choice=indices[rates[:,indices].argmax(-1)];selected=rates[np.arange(len(rates)),choice];oracle=rates[:,2:].max(-1);upper=(Y+U).mean((1,3))[np.arange(len(rates)),choice];oracle_upper=(Y+U).mean((1,3))[:,2:].max(-1)
    out=dict(record)
    for name,array in [('collision_pct',C),('timeout_pct',T),('unknown_pct',U)]:out[name]=float(array[np.arange(len(rates)),:,choice,:].mean()*100)
    regret=oracle-selected
    out.update(mean_regret_pp=float(regret.mean()*100),max_regret_pp=float(regret.max()*100),regret_identified_pp=[float((oracle-upper).mean()*100),float((oracle_upper-selected).mean()*100)],harm_controllers=int(np.sum(selected<=rates[:,1]-.1+1e-12)),catastrophe_controllers=int(np.sum(selected<=rates[:,2]-.1+1e-12)),oracle_chosen_arms=[ARMS[int(i)] for i in choice],ancillary_metrics='Same frozen in-sample argmax and menu-order tie rule; reporting only')
    assert np.isclose(out['success_pct'],selected.mean()*100)
    return out

def make_tables(families,analyses):
    headers=['子项']+list(METHOD_NAMES.values())+list(STANDARD_UNAVAILABLE)+['证据等级','原因'];rows=[];metrics=[];oracle_records={}
    for family in 'ABCDEFG':
        value=analyses.get(family);cells=[]
        for key,label in METHOD_NAMES.items():
            r=value['methods'].get(key) if value else None
            if r and key in ['complete_menu_oracle','best_fixed_oracle']:
                r=oracle_report_metrics(family,key,r);oracle_records.setdefault(NAMES[family],{})[METHOD_NAMES[key]]=r
            cells.append(f"{r['success_pct']:.2f}% [{r['success95_pct'][0]:.2f}, {r['success95_pct'][1]:.2f}]" if r else '缺口：无适用评估')
            if r:metrics.append(dict(family=NAMES[family],method=label,**{k:r.get(k) for k in ['success_pct','success95_pct','collision_pct','timeout_pct','unknown_pct','mean_regret_pp','max_regret_pp','regret_identified_pp','harm_controllers','catastrophe_controllers','crossfit_success_pct','evidence']}))
        reason='' if value else families[family]['reason']
        rows.append([NAMES[family]]+cells+['缺口：不适用']*len(STANDARD_UNAVAILABLE)+['确认性证据，受源覆盖和冻结程序条件限制；参照为描述性' if value else '无新修正终点证据',reason])
    # Pool only methods available for every analyzed family; preserve pairing.
    if analyses:
        from corrective_analysis40 import choices_for,cluster_bootstrap
        common=set.intersection(*(set(a['methods']) for a in analyses.values()));weights={};ys=[]
        for f in analyses:
            z=np.load(HERE/'selector_labels40'/f'{f}.npz');ys.append(z['success']);w=choices_for(f,z['controller_names'].tolist())
            for key in common:
                if key in w:weights.setdefault(key,[]).append(w[key])
        bs=cluster_bootstrap(np.concatenate(ys),{k:np.concatenate(v) for k,v in weights.items()},read(HERE/'SELECTOR40_CONFIG_FROZEN.json')['bootstrap_seed']+400)
        cells=[]
        for key in METHOD_NAMES:
            if key in common:
                C=[len(a['controllers']) for a in analyses.values()];r=[a['methods'][key] for a in analyses.values()];point=np.average([x['success_pct'] for x in r],weights=C);bkey=key+'_in_sample' if key in ['complete_menu_oracle','best_fixed_oracle'] else key;ci=np.percentile(bs[bkey],[2.5,97.5])*100;cells.append(f'{point:.2f}% [{ci[0]:.2f}, {ci[1]:.2f}]')
            else:cells.append('缺口：各族覆盖不齐')
        rows.append(['前六族合并，可得族按实例等权']+cells+['缺口：不适用']*len(STANDARD_UNAVAILABLE)+['确认性证据的可得子集，参照为描述性',f'实际包含{len(analyses)}族；缺失族不补零'])
    else:rows.append(['前六族合并']+['缺口：未评估']*(len(METHOD_NAMES)+len(STANDARD_UNAVAILABLE))+['无修正终点证据','多机器人能力门未提供合格族'])
    rows.append(['随机障碍场锁定测试']+['缺口：未评估']*(len(METHOD_NAMES)+len(STANDARD_UNAVAILABLE))+['未打开锁定测试','原训练清单与原评估几何重叠，训练前阻断'])
    with (HERE/'AXIS_TABLE.csv').open('w') as out:w=csv.writer(out);w.writerow(headers);w.writerows(rows)
    (HERE/'AXIS_TABLE.md').write_text('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(row)+' |\n' for row in rows)+'\n\n成功率为观测下界；方括号为五千次实例后状态成簇重抽的百分之九十五置信区间。未知识别上界、碰撞、超时、遗憾及伤害数见附表。\n\n'+ '\n\n'.join(label+'：'+reason+'。' for label,reason in STANDARD_UNAVAILABLE.items())+'\n')
    p=HERE/'AXIS_METRICS.csv'
    if metrics:
        with p.open('w') as out:w=csv.DictWriter(out,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    else:p.write_text('子项,方法,状态\n全部,全部,缺口：未评估\n')
    write(HERE/'ORACLE_REPORT_METRICS40.json',oracle_records)

def main():
    assert (HERE/'FULL_NAVIGATION_PHASE_COMPLETE.json').exists()
    assert (HERE/'FULL_DESCRIPTIVE_EXECUTION_COMPLETE.json').exists()
    phase=read(HERE/'FULL_NAVIGATION_PHASE_COMPLETE.json')
    if phase['correction_stage_eligible']:assert (HERE/'CORRECTIVE_STUDY40_COMPLETE.json').exists() or (HERE/'CORRECTIVE_STUDY40_BLOCKED.json').exists()
    integrity=compare_integrity();gates=read(HERE/'ALL_ROUND40_NAVIGATION_GATES.json')['families'];families={};gaps=[];expert_lines=[];nav_lines=[];five=[];analyses={}
    for f in 'ABCDEFG':
        gate=gates[f];expert=expert_for(f);p=HERE/f'MULTIAGENT_CAPABILITY_{f}.json';multi=read(p) if p.exists() else None;ap=HERE/f'CORRECTIVE_ANALYSIS40_{f}.json';analysis=read(ap) if ap.exists() else None
        if analysis:analyses[f]=analysis
        status=read(HERE/f'FULL_{f}_STATUS.json') if (HERE/f'FULL_{f}_STATUS.json').exists() else None
        reason=(status or {}).get('reason') or (multi or {}).get('reason') or ('多机器人能力门留下少于两个实例' if multi and not multi['family_passed'] else '修正评估未进入' if not analysis else '')
        name=NAMES[f];qualified=sum(r['passed'] for r in gate['instances']);families[f]=dict(name=name,expert_gate=expert,navigation=gate,qualified_navigation_instances=qualified,multiagent=multi,correction=analysis,reason=reason)
        if not analysis:gaps.append(dict(item=name+'的修正选择结果',reason=reason or '未进入修正评估',required_resolution='满足原导航与多机器人能力门；不得替换不合格实例或据结果调参' if f!='G' else (status or {})['required_resolution']))
        if expert:
            expert_lines.append(f"{name}：专家成功{expert['counts']['success']}/{expert['episodes']}，撞墙{expert['counts']['collision']}，超时{expert['counts']['timeout']}，未知{expert['counts']['unknown']}；专家门{'通过' if expert['passed'] else '未通过'}。最低墙体扫掠余量分位数（米）：{json.dumps(expert['minimum_wall_clearance_quantiles'],ensure_ascii=False)}。失败位置图：{Path(expert['failure_plot']).relative_to(HERE)}。")
        else:expert_lines.append(name+'：专家门未运行；'+reason+'。')
        for r in gate['instances']:nav_lines.append(f"{name}的{'确定性行为克隆策略' if r['kind']=='deterministic' else '混合密度行为克隆策略'}，初始化种子{r['seed']}：新导航留出成功{r['success']}/16，撞墙{r['collision']}，超时{r['timeout']}，未知{r['unknown']}，{'通过' if r['passed'] else '未通过'}。最终检查点：{Path(r['checkpoint']).relative_to(HERE)}。")
        if not gate['instances']:nav_lines.append(name+'：导航门未运行；'+reason+'。')
        catalog=read(HERE/'ROUND39_CATALOG_FROZEN.json')['catalog'][f];ev=[s for s in catalog if s['role']=='evaluation'];five.append(dict(family=name,geometry_valid=all(s['geometry_valid'] for s in ev),single_robot_path=all(s['single_robot_paths_exist'] for s in ev),joint_solvable=all(s['joint_feasibility']=='certified' for s in ev),joint_reference_budget=10000,controller_success=analysis['methods']['safety_baseline']['success_pct'] if analysis else None,deadlock_repair_success_uplift_pp=analysis['descriptive']['best_correction_minus_safety_pp'] if analysis else None,timeout_not_deadlock=True,performance_budget_matched_expert=False))
    gaps.extend([dict(item='原双瓶颈与原四路交叉历史训练源',reason='指定旧标签在流积分内部加修正，本轮在第一次安全投影之后加修正，动作位置不兼容',required_resolution='找到已有同位置修正、同菜单与可恢复状态的标签；本轮不补跑旧场景'),dict(item='零个新训练族的规模曲线',reason='只有一个兼容历史源场景，原留一源场景验证无法成立',required_resolution='补齐兼容历史源场景；不改验证规则'),dict(item='原流场完整行为探测',reason='严格历史名单中一个原流场身份没有完整兼容的十六状态、两组参数、两百步轨迹',required_resolution='找到已有兼容完整轨迹；缺失特征仅在训练折插补')])
    gaps.append(dict(item='原环形交换完整修正菜单的历史覆盖',reason='兼容旧标签只覆盖三组全程固定参数及第十三组至第十组按比例切换；缺少其余三臂。旧围绕场景中心的持续右手规则与本轮朝目标方向顺时针旋转的规则不兼容，已排除',required_resolution='找到已有同执行语义的缺失臂标签；本轮不补跑旧场景。源平均对照的各臂覆盖明细保存在冻结选择文件中'))
    af=read(HERE/'AF_ANALYSIS40_RESULTS.json') if (HERE/'AF_ANALYSIS40_RESULTS.json').exists() else None;scale=read(HERE/'SCALE_CURVE40_RESULTS.json') if (HERE/'SCALE_CURVE40_RESULTS.json').exists() else None
    if scale and scale['missing_fixed_combinations']:
        missing=scale['missing_fixed_combinations'];by_m={str(m):sum(r[1]==m for r in missing) for m in [0,1,3,5]}
        gaps.append(dict(item='预注册训练族数量规模曲线',reason=f'合格目标族中缺失{len(missing)}个固定条件；按训练族数计为'+json.dumps(by_m,ensure_ascii=False)+'。受阻源族不替换；不同可得规模点的目标构成不同',required_resolution='补齐兼容源证据与合格源族，并在独立预注册研究中检验原固定组合；本轮不据评估结果更换组合'))
    counts='；'.join(f'{NAMES[f]}（族 {f}）{families[f]["qualified_navigation_instances"]}个通过' if families[f]['navigation']['instances'] else f'{NAMES[f]}（族 {f}）未运行' for f in 'ABCDEFG')
    first=('恢复后的完整性审计通过' if integrity['passed'] else '恢复后的完整性审计未通过，差异已列出')+'；此前只读源码边界偏差和用户授权恢复记录原样保留。串联瓶颈走廊与房间与门试点通过。导航能力门：'+counts+'。'
    if af:
        first+=f"多场景终点一{'通过' if af['endpoint1']['passed'] else '未完整通过'}，终点二{'通过' if af['endpoint2']['passed'] else '未通过'}（{af['endpoint2']['count']}族遗憾不超过五个百分点），终点三{'通过' if af['endpoint3']['passed'] else '因固定组合缺口无法完整判定'}；随机障碍场锁定测试因训练几何隔离冲突保持未打开，终点四未评估。"
        if 'available_family_paired_difference_pp' in af['endpoint1']:
            first+=f"可得族合并的模型遗憾减源平均最佳臂遗憾为{af['endpoint1']['available_family_paired_difference_pp']:.2f}个百分点，配对百分之九十五置信区间{np.asarray(af['endpoint1']['paired95_pp']).round(2).tolist()}；规模曲线可得组合斜率为{af['endpoint3']['slope']}个百分点每训练族，区间{af['endpoint3']['slope95']}。"
        for f,a in analyses.items():
            first+=f"{NAMES[f]}最佳修正相对只加安全投影提高{a['descriptive']['best_correction_minus_safety_pp']:.2f}个百分点；比较时间表与持续右手规则时，偏好{a['descriptive']['preferred_arm_family']}。"
        first+='串联瓶颈走廊与房间与门的多机器人能力门分别只保留一个和零个实例，修正选择结果记为缺口。'
    else:first+='修正选择终点、规模曲线尚无可得结果；随机障碍场锁定测试未打开。'
    result=dict(status='COMPLETE_WITH_GAPS',integrity=integrity,pilot_passed=True,families=families,AF_endpoints=af,scaling_curve=scale,locked_test=dict(status='GAP',evaluations=0,labels_opened=0,reason=families['G']['reason']),gaps=gaps,unknown_retries=0,timeout_not_deadlock=True,report_first_paragraph=first)
    write(HERE/'results.json',result);write(HERE/'GAPS.json',gaps);write(HERE/'FIVE_LAYER_EVIDENCE.json',five)
    previous=D/'multiscene_generalization_v1';oldnav=read(previous/'ALL_VALID_NAVIGATION_GATES.json')['families'];olddata=read(previous/'DATASET_FROZEN.json')['summary'];comparison=[]
    for f in 'ABCDEFG':
        oldbest=max(oldnav[f]['instances'],key=lambda r:r['success']);newbest=max(gates[f]['instances'],key=lambda r:r['success']) if gates[f]['instances'] else None;expert=expert_for(f);outcomes=olddata[f]['outcomes'];comparison.append(dict(family=NAMES[f],old_expert_collision_pct=outcomes.get('collision',0)/sum(outcomes.values())*100,new_expert_collision_pct=expert['collision_rate']*100 if expert else None,old_best_success_pct=oldbest['success']/16*100,old_best_collision_pct=oldbest['collision']/16*100,old_best_timeout_pct=oldbest['timeout']/16*100,new_best_success_pct=newbest['success']/16*100 if newbest else None,new_best_collision_pct=newbest['collision']/16*100 if newbest else None,new_best_timeout_pct=newbest['timeout']/16*100 if newbest else None))
    write(HERE/'NAVIGATION_COMPARISON40.json',comparison)
    with (HERE/'NAVIGATION_COMPARISON40.csv').open('w') as out:w=csv.DictWriter(out,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    compheader=['族','旧专家撞墙率','新专家撞墙率','旧最佳成功率','旧最佳撞墙率','旧最佳超时率','新最佳成功率','新最佳撞墙率','新最佳超时率'];comptext='| '+' | '.join(compheader)+' |\n| '+' | '.join(['---']*len(compheader))+' |\n'
    for r in comparison:comptext+='| '+r['family']+' | '+' | '.join('未运行' if v is None else f'{v:.2f}%' for k,v in r.items() if k!='family')+' |\n'
    (HERE/'EXPERT_GATE.md').write_text('\n\n'.join(expert_lines)+'\n');(HERE/'NAVIGATION_GATE.md').write_text('\n\n'.join(nav_lines)+'\n\n'+comptext+'\n旧评估几何上的本轮结果只作描述，不用于检查点或训练停止选择；详见各族策略诊断及训练轮次清单。旧、新导航门使用不同几何，专家样本规模也不同，以上为描述性对照，不能称为配对因果改善。\n')
    make_tables(families,analyses)
    fewshot_rows=[]
    for f,a in analyses.items():
        for method,curve in a.get('fewshot_curves',{}).items():
            fewshot_rows.append(dict(family=NAMES[f],method='经验选择' if method.startswith('empirical') else '贝叶斯选择',known_labels_per_arm=curve['known_labels_per_arm'],total_known_label_budget=curve['total_known_label_budget'],mean_regret_pp=curve['mean_regret_pp'],mean_p90_regret_pp=curve['mean_p90_regret_pp'],all_controllers_available=curve['all_controllers_available']))
    if fewshot_rows:
        with (HERE/'FEWSHOT_CURVE40.csv').open('w') as out:w=csv.DictWriter(out,fieldnames=list(fewshot_rows[0]));w.writeheader();w.writerows(fewshot_rows)
    else:(HERE/'FEWSHOT_CURVE40.csv').write_text('族,方法,状态\n全部,全部,缺口：没有适用修正评估\n')
    paragraphs=[first,'本轮只在规定目录新增代码和输出。源码与物理只读复用，生成多机器人环境完全继承原步进；授权的单机器人环境仅用于示范和导航门，原等价性测试按提示复用，没有重新运行。所有策略固定使用第三轮检查点，没有按导航留出选择轮次。共享数据库只经追加身份登记、预检、缺失执行、默认记录、单进程官方合并和后检。','原评估状态按提示原样沿用。原训练与原修正评估几何在多个族有重叠，已单独审计；新导航留出与所有训练几何分离。随机障碍场受到评估几何不得训练的额外明确约束，冲突一块原训练地图，因此在新训练前阻断，修正评估零次。不能把旧原评估地图描述为未见导航地图。','严格历史训练源使用一万二千条旧环形交换标签，其中七十二条未知保留。继承的宽审计额外包含三类不在提示训练源名单内的控制器，旧宽审计和旧派生数据保留，选择训练只读严格子集。原双瓶颈与原四路交叉旧执行链不兼容，未补跑旧场景。','全局算法、特征、固定族组合和超参数早于修正评估采集冻结。前六族结果按族独立打包，每个目标的模型只读其他源族数据；对应选择收据早于该目标分析。报告区分采集、源拟合、冻结和目标分析，不把冻结时间写为早于已发生的采集。','死锁修复用相对只加安全投影的成功率提高衡量。超时不直接称为死锁，停滞候选只作诊断。未知保持识别上下界，不重新运行。样本内最佳参照与按状态奇偶交叉拟合最佳参照分别报告。']
    for f,a in analyses.items():
        winners=a['descriptive']['winning_arm_by_controller'].values();composition='、'.join(f'{METHOD_NAMES[arm]}：{list(winners).count(arm)}个实例' for arm in ARMS if arm in winners)
        paragraphs.append(f"{NAMES[f]}：最佳修正相对只加安全投影提高{a['descriptive']['best_correction_minus_safety_pp']:.2f}个百分点；比较时间表与持续右手规则时，偏好{a['descriptive']['preferred_arm_family']}，持续右手规则减最佳时间表为{a['descriptive']['right_hand_minus_best_schedule_pp']:.2f}个百分点，百分之九十五配对置信区间为{np.asarray(a['descriptive']['right_hand_minus_best_schedule95_pp']).round(2).tolist()}。完整菜单样本内最佳臂构成：{composition}。最佳臂为事后描述性参照，不用作未来选择。")
    if scale:
        coverage='；'.join(f'训练{m}个新族：{len([r for r in scale["rows"] if r["m"]==m])}个可得固定组合，目标族为'+('、'.join(NAMES[f] for f in sorted({r['family'] for r in scale['rows'] if r['m']==m})) or '无') for m in [0,1,3,5])
        paragraphs.append('规模曲线可得均值（训练族数：遗憾百分点）：'+json.dumps(scale['mean_regret_by_training_family_count'],ensure_ascii=False)+'；可得组合斜率与置信区间：'+json.dumps(dict(slope=scale['slope_pp_per_family'],interval=scale['slope95']),ensure_ascii=False)+'。覆盖：'+coverage+'。不同训练族数的目标构成不同，部分曲线不支持训练族数量的因果趋势结论。零族及其他固定组合缺口不补零，不视为完整规模终点通过。')
    paragraphs.append('缺口及补齐条件：'+'；'.join(g['item']+'：'+g['reason'] for g in gaps)+'。')
    paragraphs.append('实际冻结的训练与新导航留出几何图册见 [几何图册](GEOMETRY_FIGURES40.md)；图像只展示已有清单，不产生轨迹。保存轨迹的哈希与官方记录核查见 EVIDENCE_REVIEW40.json。')
    if scale:paragraphs.append('独立曲线文件：[训练族数量与遗憾](figures/training_families_regret.svg)、[少量标签与遗憾](figures/fewshot_labels_regret.svg)；数值来源与哈希见 FIGURES40_PROVENANCE.json。')
    (HERE/'REPORT.md').write_text('\n\n'.join(paragraphs)+'\n')
    (HERE/'HANDOFF.md').write_text('Basin_C1 / OrthoFlow3 研究的实现交接\n\n'+first+'\n\n工作根目录 /home/zhihan/research/Basin_C1；本轮目录 diagnostics/multiscene_navigation_competence_v1/。全部新增代码、检查点、清单、预测和报告均在本轮目录。禁止目录按 decentralized_ 和 teacher_student_ 前缀剪枝；分布式流水线目录仅按共同协议只读例外。禁止与并行目录只记录元数据，原边界失败证据保留。\n\n复现十六条、旧几何等价性、试点门、所有已完成策略及未知记录均保留，不重复。继续前先读本轮提示、共同协议、恢复开工审计、全部预注册与冻结收据。源码和数据库规则不变，不安装包，不执行任何写入版本历史的命令。\n\n选择程序在 corrective_pipeline40.py，按源集合复用冻结模型；严格历史源指针在 HISTORICAL_SELECTOR_SOURCE40_CURRENT.json。目标族不进入其模型的拟合、验证或预处理；随机障碍场不得进入任何选择训练。冻结配置不能据当前结果改变。\n\n随机障碍场尚无新导航策略，原训练几何与锁定评估几何冲突在 ORIGINAL_TRAIN_CORRECTION_EVALUATION_GEOMETRY_OVERLAP_AUDIT.json 和 LOCKED_G_TRAIN_GEOMETRY_GAP.receipt.json。需要明确非重叠训练清单修订；不能替换原评估状态，也不能借用前六族示范训练该族策略。锁定测试零次评估。\n\n'+ '\n\n'.join(g['item']+'：'+g['reason']+'；补齐条件：'+g['required_resolution'] for g in gaps)+'\n\n可审阅结果在 REPORT.md、results.json、AXIS_TABLE.md、AXIS_METRICS.csv、导航与专家门报告、五层证据、各族每轮训练数据清单以及源集合模型收据。超时和停滞不直接称为死锁，修复用成功率相对只加安全投影的提高衡量。\n')
    artifacts=['REPORT.md','HANDOFF.md','results.json','GAPS.json','AXIS_TABLE.md','AXIS_TABLE.csv','AXIS_METRICS.csv','ORACLE_REPORT_METRICS40.json','EXPERT_GATE.md','NAVIGATION_GATE.md','NAVIGATION_COMPARISON40.csv','FEWSHOT_CURVE40.csv','FIVE_LAYER_EVIDENCE.json','INTEGRITY_FINAL_COMPARISON.json','GEOMETRY_FIGURES40.md','GEOMETRY_FIGURES40_PROVENANCE.json','EVIDENCE_REVIEW40_COMPLETE.receipt.json']
    if scale:artifacts+=['FIGURES40_PROVENANCE.json','figures/training_families_regret.svg','figures/fewshot_labels_regret.svg']
    seal('FINAL_DELIVERY40.receipt.json',[HERE/n for n in artifacts],complete_with_gaps=True,integrity_passed=integrity['passed'],G_evaluations=0)

if __name__=='__main__':main()
