"""Standard deliverables, generated from sealed results without retuning."""
from common import *
from labels import TYPE_NAMES,type_name,method_name
from families import family,UNSEEN
import csv

def dump_csv(name,rows):
    with (HERE/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(clean(rows))
def cell(r):
    if not r:return 'N/A'
    return f"{r['success_lower_pct']:.2f} [{r['success_CI95_pct'][0]:.2f}, {r['success_CI95_pct'][1]:.2f}]"
def gain(r):
    if 'gain_pp' not in r:return 'N/A'
    return f"{r['gain_pp']:+.2f}pp [{r['CI95_pp'][0]:+.2f}, {r['CI95_pp'][1]:+.2f}]"

def main():
    assert (HERE/'FAILURE_ATTRIBUTION.receipt.json').exists();audit=read(HERE/'INTEGRITY_COMPARISON.json');dev=read(HERE/'DEVELOPMENT_RESULTS.json');confirmed=read(HERE/'CONFIRMATORY_RESULTS.json');known=read(HERE/'KNOWN_NEW_STATE_RESULTS.json');frozen=read(HERE/'FROZEN_PIPELINE.json');gate=read(HERE/'UNSEEN_CAPABILITY.json');attribute=read(HERE/'FAILURE_ATTRIBUTION.json');primary=frozen['primary_method'];other='fixed500' if primary=='self_schedule' else 'self_schedule'
    columns=['pipeline',other,'no_schedule','no_controller','legacy_pipeline','all_S500','all_Sprog','all_safety_baseline','nominal_only','evaluated_state_crossfit_oracle','evaluated_state_in_sample_oracle','conservative'];table=[];metrics=[]
    def add(phase,group,report,evidence,reason=''):
        methods=report.get('type_equal',{}).get('methods',{}) if report else {};row=dict(phase=phase,group=type_name(group),evidence=evidence)
        for key in columns:row[method_name(key)]=cell(methods.get(key))
        row['说明']=reason;table.append(row)
        for key,r in methods.items():metrics.append(dict(phase=phase,group=type_name(group),method=method_name(key),evidence=evidence,**r))
    base=read(HERE/'BASE_PROTOCOL.json');alltypes=sorted(set(base['types'].values()))
    for kind in [*alltypes,*UNSEEN]:add('训练状态留一类型',kind,dev['reports'].get(kind),'DEVELOPMENT','未见类型不提供训练标签' if kind in UNSEEN else '仅26个预标动作；不完整的历史参照为N/A；训练seed交叉拟合oracle不可用')
    for group,title in [('combined','训练类型合并'),('learned','学习型'),('optimization','优化型'),('search','搜索型')]:
        add('训练状态留一类型',title,dev['reports'].get(group),'DEVELOPMENT')
    add('训练状态留一类型','未见类型合并',None,'N/A','未见类型不参与critic训练')
    for kind in [*alltypes,*UNSEEN]:
        r=confirmed['reports'].get(kind) if kind in UNSEEN else known['reports'].get(kind);add('新评估状态',kind,r,'CONFIRMATORY' if kind in UNSEEN else 'DEVELOPMENT','原九类学习型不做新状态回归；仅保守强化学习按明确42实例清单保留' if not r and kind not in UNSEEN else '')
    add('新评估状态','未见类型合并',confirmed['reports'].get('combined'),'CONFIRMATORY','固定类型分层、类型等权')
    add('新评估状态','训练类型合并',known['reports'].get('combined'),'DEVELOPMENT','42个已有实例；不带时间表、top2–4和原始动作未评估')
    add('新评估状态','学习型',None,'N/A','按新版指定表格：原九类学习型无新状态回归；保守强化学习6实例仍单类型报告')
    for group in ('optimization','search'):add('新评估状态',group,known['reports'].get(group),'DEVELOPMENT')
    dump_csv('AXIS_TABLE.csv',table)
    headers=list(table[0]);md=['# 双η时间表跨控制器类型检验\n','数值为成功率% [95% CI]。全称与身份：'+ '；'.join(type_name(k,True) for k in TYPE_NAMES)+'。\n','训练状态与已有类型新状态均为DEVELOPMENT；仅三个新类型新状态为CONFIRMATORY。所有未标动作保留缺口，不以失败代替。\n','| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    md += ['| '+' | '.join(str(r[k]) for k in headers)+' |' for r in table];(HERE/'AXIS_TABLE.md').write_text('\n'.join(md)+'\n');write(HERE/'AXIS_METRICS.json',metrics)
    if metrics:
        keys=list(dict.fromkeys(k for r in metrics for k in r));dump_csv('AXIS_METRICS.csv',[{k:json.dumps(r.get(k),ensure_ascii=False) if isinstance(r.get(k),(list,dict)) else r.get(k) for k in keys} for r in metrics])
    zoo=[]
    with (D/'crosstype2_mixed_zoo_selector_v1/ZOO_MANIFEST.csv').open() as f:old={r['controller']:r for r in csv.DictReader(f)}
    for n in base['controller_order']:
        t=base['types'][n];v=base['profiles'][n]['payload'];zoo.append(dict(controller=n,type=type_name(t),type_code=t,family=family(t),cohort='training',parameters=json.dumps(v.get('nominal_parameters'),ensure_ascii=False),checkpoint=v.get('flow_checkpoint_path'),capability_passed=True,safety_capability_rate=old[n].get('safety_success_rate','Historical authoritative manifest'),no_repair_needed=old[n].get('no_repair_needed','See historical manifest'),recipe=v.get('execution_semantics')))
    unseenbase=read(HERE/'EVALUATION_PROTOCOL_BASE.json');statuses={r['controller']:r for r in gate['rows']}
    for n in read(HERE/'OPENING_CONFIG.json')['unseen_order']:
        v=unseenbase['profiles'][n]['payload'];s=statuses[n];t=s['type'];zoo.append(dict(controller=n,type=type_name(t),type_code=t,family=family(t),cohort='unseen',parameters=json.dumps(v.get('nominal_parameters'),ensure_ascii=False),checkpoint=v.get('flow_checkpoint_path'),capability_passed=s['passed'],safety_capability_rate=s['safety_success_rate'],no_repair_needed=s['no_repair_needed'],recipe=v.get('execution_semantics')))
    dump_csv('ZOO_MANIFEST.csv',zoo)
    combined=confirmed['endpoints'].get('combined',{});e1=confirmed['endpoints'].get('endpoint1_needs_repair_combined',combined.get('endpoint1',{}));eligible_types=[t for t in UNSEEN if t in confirmed['endpoints'] and not confirmed['endpoints'][t]['endpoint1'].get('excluded_from_repair_effectiveness')];allpassed=bool(eligible_types) and all(confirmed['endpoints'][t]['endpoint1'].get('passed') for t in eligible_types) and e1.get('passed',False);noninferior=combined.get('endpoint2',{}).get('noninferior',False)
    answer='主线在需要修正的未见类型上均获得相对安全投影的明确提升，且不劣于较好的人工时间表。' if allpassed and noninferior else '主线在需要修正的未见类型上获得明确提升，但学出的双η尚不能证明完全替代较好的人工时间表。' if allpassed else '当前证据只支持部分未见类型的修正泛化，不能宣称普遍泛化。'
    report=[('完整性审计通过：保护源码和历史输入未变；用户主动修改的提示词已独立记录并保留两次修订快照。' if audit['passed'] else '完整性审计存在差异，详见INTEGRITY_COMPARISON.json。')+answer,'\n本轮保持跨类型1c的生成器、critic结构、24维机制特征、双η动作与源内选择规则。三个新类型整类不参与critic训练；已有16类型全部属于研究者已用过的开发范围。冻结前仅执行已获用户确认的盲化能力门与无成败机制探测。','\n主方法：'+method_name(primary)+'。']
    for j in range(1,7):
        r=e1 if j==1 else combined.get('endpoint'+str(j),{});report.append(f'\n终点{j}：{gain(r)}；判定与逐类型结果见CONFIRMATORY_RESULTS.json。')
    summary=confirmed['reports'].get('combined',{}).get('type_equal',{}).get('methods',{})
    if summary:report.append(f"\n未见类型等权成功率：主方法{cell(summary['pipeline'])}；只加安全投影{cell(summary['all_safety_baseline'])}。")
    eta=read(HERE/'SELECTED_ETA_DESCRIPTIONS.json')['rows']
    for t in UNSEEN:
        r=confirmed['reports'].get(t,{}).get('type_equal',{}).get('methods',{});a=[x for x in eta if x['type']==t]
        if not r:report.append('\n'+type_name(t,True)+'：能力或执行证据缺口，没有替换类型。');continue
        early=np.median([x['eta_early'] for x in a],0);late=np.median([x['eta_late'] for x in a],0);why=attribute['rows'].get(t,{}).get('reasons',[])
        report.append('\n'+type_name(t,True)+f"：只加安全投影{r['all_safety_baseline']['success_lower_pct']:.2f}%，主方法{r['pipeline']['success_lower_pct']:.2f}%；选中前期/后期η中位数分别为{early.tolist()}、{late.tolist()}。"+'；'.join(why)+'。')
    report += ['\n完整数值与限制：AXIS_TABLE.md、AXIS_METRICS.json、FAILURE_ATTRIBUTION.md、GAPS.md。终点7的分布、越界、遗憾与逐条转移见SELECTED_ETA_DESCRIPTIONS.json、FEATURE_OUT_OF_RANGE.json、CONTROLLER_STATE_CROSSFIT_REGRET.csv、NOMINAL_TO_SAFETY_TRANSITIONS.csv及figures/。','\n确认性置信区间使用5000次配对重抽，未见合并固定类型分层、类型等权，controller内再按state成簇，两个seed保留在同簇。每次重选较好人工对照和实际评估动作oracle。unknown保留上下界并抵消相同UID；timeout只报告超时。','\n冲突搜索实现为Sharon等提出的冲突树搜索的有限节点预算版本，底层采用既有连续圆盘时空栅格；不宣称全部Ring路径最优。[算法原文](https://doi.org/10.1016/j.artint.2014.11.006)。']
    (HERE/'REPORT.md').write_text('\n'.join(report)+'\n')
    gaps=['# 证据缺口与限制\n','旧16类型均为DEVELOPMENT；仅三新类型新状态为CONFIRMATORY。生成器仍沿用原先的预训练范围，不能说整个pipeline从未见过任何相关场景。','每训练pair只有26/288唯一动作和seed0，不具完整菜单oracle或训练seed交叉拟合证据。诊断只有8状态、26动作、seed0；样本内最佳是事后上界，不能证明未标262动作都无效。','历史5个机制特征向量不完整，NaN保留，仅在源内填补。抽查8个向量中7个为完整数值的逐位复现，另1个按不完整轨迹规则重现全NaN缺失状态。','训练人工参照只查旧记录；不完整覆盖明确N/A。原九类学习型不做新状态回归，学习型是否退步用相同训练状态/动作的留一类型结果与旧跨类型1c对照；明确42清单中的保守强化学习仍单类型保留。','梯度下降模型预测控制在能力门中全部只加安全投影已成功，按预设规则只检验不伤害，不计入修正有效性的类型判定。','新状态对DB全UID、内容及可恢复物理坐标去重；历史不可恢复物理记录的残余见NEW_STATE_DEDUP.json。','持续右手规则完全不进入候选、对照、特征或标签。共同协议的未实施专家、规则选择、旧A3/A0等标准项为N/A；没有为了填这些栏目增加未经要求的评估。','首4条耗时来自已计划能力门，不重复生成；用户提示词在开工后修订，修订范围及新增诊断抽样在任何新评估前封存。','不安装依赖，不修改旧代码、物理、安全、事件或成功标准，不做Git写操作。用户修改现有提示词在完整性审计中独立列为授权差异。']
    gaps += [json.dumps(g,ensure_ascii=False) for g in confirmed.get('gaps',[])];(HERE/'GAPS.md').write_text('\n\n'.join(gaps)+'\n')
    handoff=['# 跨类型3接手说明\n',f'完整输出目录：{HERE}。REPORT.md给结论，AXIS_TABLE.md/CSV与AXIS_METRICS.json给逐类型、族和两阶段统计。仅三个新类型新状态是确认性证据，事后归因为开发证据。','有效规范为PROMPT_REVISION_20261019_V2.md，原预注册与两次修订收据均保留；盲化能力/机制例外和保守额外动作授权分别在USER_DESIGN_DECISIONS与CONSERVATIVE_EVALUATION_AUTHORIZATION收据。','可直接复用：104个旧类型的精确profile，62实例旧动作索引与DB标签；冻结生成器、STATE_INPUTS/CANDIDATES；三个新控制器代码和Transformer checkpoints；24维机制特征和原轨迹索引；TRAINING_BUNDLE、critic_fits/、源内曲线及SELECTED_MODELS。','每份旧目录构造器可能写自己的缓存。下一轮复制到新目录、记录SHA并先复现，不直接运行本目录入口。new_execution.py只经既有公开cache供给新nominal，物理循环完全沿用复现过的mixed runtime。','16折留一类型、3族留一描述模型按精确source set缓存156组×2分支，每拟合3个seed。推理恢复h/c float32与C统计float64，highest矩阵乘法。动作编码为前期3、后期3、后期−前期3、切换步/700；与旧编码同维但不能互换。','新评估请求顺序优先未见类型，所有方法在首次新状态提交前选定并密封。保守动作去重后单列3744上限；事后诊断仅在CONFIRMATORY_RESULTS.receipt之后提交。','数据按KNOWN_EVAL_BUNDLE与UNSEEN_EVAL_BUNDLE分开，分别24与48状态，−2为缺失填充、−1为unknown；不可混淆。全部实际动作与两个seed明细见CONTROLLER_STATE_DETAILS.csv。','下一轮若继续用这三个已打开的类型，应标开发范围；不能重新宣称它们是研究者未见类型。新类型不得由能力不合格的替代类型顶替。','全部缺口见GAPS.md；保留大型数据、checkpoint和SQLite，不清理，不做Git提交。']
    (HERE/'HANDOFF.md').write_text('\n\n'.join(handoff)+'\n')
    write(HERE/'results.json',dict(utc=utc(),answer=answer,integrity=audit,evidence='CONFIRMATORY_TRULY_UNSEEN_TYPES_PLUS_DEVELOPMENT',primary_method=primary,confirmed_endpoints=confirmed['endpoints'],development_file='DEVELOPMENT_RESULTS.json',known_new_state_file='KNOWN_NEW_STATE_RESULTS.json',failure_attribution_file='FAILURE_ATTRIBUTION.json'))
    seal('DELIVERY.receipt.json',[HERE/n for n in ('REPORT.md','HANDOFF.md','GAPS.md','results.json','AXIS_TABLE.md','AXIS_TABLE.csv','AXIS_METRICS.json','AXIS_METRICS.csv','ZOO_MANIFEST.csv','report.py','CONFIRMATORY_RESULTS.receipt.json','FAILURE_ATTRIBUTION.receipt.json','INTEGRITY_COMPARISON.receipt.json','FIGURES.receipt.json')],no_post_confirmation_retuning=True)
    print(answer,flush=True)
if __name__=='__main__':main()
