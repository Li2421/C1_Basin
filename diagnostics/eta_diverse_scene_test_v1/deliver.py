"""Deliver the finite no-model test, retaining capability exclusions explicitly."""
from common import *
from design import NEW_NAMES,OLD_NAMES,CONTROL_IDS,CONTROL_NAMES,menu,describe_eta
from audit import database,integrity
import numpy as np

def pp(x):return f'{100*x:+.2f}'
def pct(x):return f'{100*x:.2f}%'
def interval(g):return f"{pp(g['gap'])}个百分点，95%区间[{pp(g['ci'][0])}, {pp(g['ci'][1])}]"
def eta(a):return describe_eta(a['key'],a['eta_a'])

def geometry_finish():
    geometry=read(HERE/'GEOMETRY_BASE_REPORTS.json');labels=read(HERE/'NEW_LABEL_TABLES.json');cap=read(HERE/'CAPABILITY_TABLE.json')
    for n,report in geometry.items():
        report['joint_solvable']='未认证';report['deadlock_resolution']='未认证'
        for si,state in enumerate(report['states']):
            state['joint_solvable']='未认证';state['deadlock_resolution']='未认证'
            if n in labels and labels[n]['complete']:
                y=np.asarray(labels[n]['y']);state['controller_success']=dict(joint_rollouts_tested=True,seed=0,
                    successful_candidates={ctl:int(y[c,si].sum())for c,ctl in enumerate(CONTROL_IDS)},candidates_per_controller=25)
            else:state['controller_success']=dict(joint_rollouts_tested=False,reason=cap['scene_status'][n]['status'],single_robot_capability=cap['per_scene'][n])
    write(HERE/'GEOMETRY_REPORTS.json',geometry)

def make_report():
    s=read(HERE/'STATISTICS.json');d=read(HERE/'TEST_DECISION.json');cap=read(HERE/'CAPABILITY_TABLE.json');cohort=read(HERE/'HISTORICAL_COHORT_CHECK.json')
    first=[d['status']+'。']
    for n,r in s['new_scenes'].items():
        if r['judgable']or'gaps'in r:first.append(f"{NEW_NAMES[n]}（{n}）：{r['status']}，新场景的缺口(3)为{interval(r['gaps']['3'])}，样本内最好η为{eta(r['in_sample_best_candidate'])}。")
        else:first.append(f"{NEW_NAMES[n]}（{n}）：{r['status']}，缺口(3)、区间和主标签最优η均无法计算。")
    for key,label in [('new_plus_base','4个新场景加对穿6机'),('new_plus_existing11','4个新场景加全部11个现有场景')]:
        g=s['coverage'][key]
        if g['full_requested_group_available']:first.append(f"{label}的覆盖缺口(3)为{pp(g['results']['3']['coverage_gap'])}个百分点。")
        else:first.append(f"{label}的完整组覆盖缺口(3)无法计算（缺少未通过能力门的场景主标签）；可用子集的覆盖缺口(3)为{pp(g['results']['3']['coverage_gap'])}个百分点，仅作附报。")
    first=''.join(first)+'\n\n'
    lines=[first,'本轮只检验预定4个单旋钮场景、3个快速控制器、25个固定候选；不训练模型、不做第二阶段，不追加rollout。全部新场景保留基础对穿6机的12个位置、目标和速度；大机器人只改半径至0.14，紧时限只改步数260→130；障碍按说明的Config默认字段与指定矩形构造。所有初始状态几何有效、每台机器人有单机器人路径，但联合可解与死锁解决均未认证。\n\n',
        '控制器为'+'、'.join(f'{CONTROL_NAMES[c]}（{c}）'for c in CONTROL_IDS)+'。能力门使用状态0–3、每台机器人单独运行、种子0与1，每控制器每场景48次；三者均≥95%才进入主测试。其余场景记为控制器不适用，不换控制器、不替换场景。\n\n',
        '“每个场景自己的最好 η”：200次随机对半，训练半选25候选中成功率最高者，计分半计分；并列按预定候选顺序。“现有场景最好的 k 个固定 η”只使用11个现有场景标签，从25候选穷举全部组合挑选。按用户明确的主口径，在新场景计分半直接取这k个候选的最高成功率，形成较强的覆盖基准；另报训练半从池中选再计分的口径。交叉使用损失的互异性要求两个方向均≥10个百分点。\n\n',
        '每场景3个控制器、状态均等权。unknown计失败，表中另报计成功时的成功率上界；不重跑unknown。新场景的缺口区间按状态重抽5000次，每次内部重新做200次对半划分；numpy.random.default_rng(2026102611)，冻结分割表后连续使用随机流。比较方法共享同一场景的计分状态，bootstrap保留3个控制器的同一状态一起重抽。池集合挑选使用等价整数分数避免浮点并列误差。\n\n',
        '样本内最佳候选和前3名仅作描述；缺口与交叉损失使用上述对半计分，不以样本内最优率代替。“自己的最好η”在不同划分中可能不同，选择频数保存在STATISTICS.json。样本内最好率不在[20%,95%]时标为本测试无法判断，不算不符合。\n\n',
        '| 新场景 | 控制器 | 能力门成功/48 | 成功率 | 通过 |\n|---|---|---:|---:|---|\n']
    for n,rows in cap['per_scene'].items():
        for ctl,r in rows.items():lines.append(f"| {NEW_NAMES[n]}（{n}） | {CONTROL_NAMES[ctl]}（{ctl}） | {r['success']}/48 | {pct(r['rate'])} | {'是'if r['passed']else'否'} |\n")
    lines.append('\n两个障碍场景的能力门失败只说明这些既定控制器在指定执行配置下不适用，不说明几何无效、联合不可解或发生死锁。几何与控制器证据分层见GEOMETRY_REPORTS.json。\n\n')
    lines.append('| 现有场景最好的 k 个固定 η | 候选（三数依次为朝目标、侧移、互斥） | 现有11场景等权覆盖率 | 穷举组合数 |\n|---|---|---:|---:|\n')
    for k,r in s['existing_best_sets'].items():lines.append(f"| k={k} | {'；'.join(eta(a)for a in r['candidates'])} | {pct(r['existing_scene_mean_oracle_success'])} | {r['combinations_checked']} |\n")
    lines.append('\n## 每个新场景\n\n')
    candidate_table=[];controller_table=['| 场景 | 控制器 | 样本内最好η（三个数） | 成功率 |\n|---|---|---|---:|\n']
    for n,r in s['new_scenes'].items():
        section=[f'### {NEW_NAMES[n]}（{n}）\n\n']
        if'gaps'not in r:
            section.append(f"{r['status']}；未生成25候选主标签。\n\n")
            for ctl in CONTROL_IDS:controller_table.append(f"| {NEW_NAMES[n]}（{n}） | {CONTROL_NAMES[ctl]}（{ctl}） | 未测：能力门未通过 | 不适用 |\n")
        else:
            section.append(f"样本内最好成功率{pct(r['in_sample_best_rate'])}；每个场景自己的最好 η对半计分为{pct(r['own_best_crossvalidated_rate'])}，unknown计成功上界{pct(r['own_best_unknown_success_upper'])}。{r['status']}。\n\n")
            section.append('样本内前3名：'+'；'.join(f"{eta(a['candidate'])}，{pct(a['rate'])}"for a in r['top3'])+'。\n\n')
            section.append('| k | 固定池候选在全部12状态上的成功率 | 计分半池内最高成功率 | 新场景的缺口(k)与95%区间 | 训练半池内选择的缺口（附报） |\n|---|---|---:|---|---:|\n')
            for k,g in r['gaps'].items():
                description='；'.join(f"{describe_eta(a['key'],a['eta'])}：{pct(a['success_rate'])}"for a in g['pool_candidate_full_state_rates'])
                section.append(f"| {k} | {description} | {pct(g['pool_oracle_scoring_rate'])} | {interval(g)} | {pp(g['train_selected_pool_gap'])}个百分点 |\n")
            for ctl,a in r['per_controller_best'].items():controller_table.append(f"| {NEW_NAMES[n]}（{n}） | {CONTROL_NAMES[ctl]}（{ctl}） | {eta(a['candidate'])} | {pct(a['rate'])} |\n")
        section.append('\n| 候选η（三个数） | 3控制器等权成功率 | unknown计成功上界 |\n|---|---:|---:|\n')
        for a in(r['candidate_rates']if'candidate_rates'in r else[dict(candidate=a,rate=None,unknown_success_upper=None)for a in menu()]):
            section.append(f"| {eta(a['candidate'])} | {pct(a['rate'])if a['rate']is not None else '未测：能力门未通过'} | {pct(a['unknown_success_upper'])if a['unknown_success_upper']is not None else '不适用'} |\n")
        section.append('\n');lines.extend(section);candidate_table.extend(section)
    lines.append('## 交叉使用损失\n\n行场景训练半选出的η用于列场景计分半；表内为相对列场景自己的最好η所损失的百分点。两个方向分别计算，负值表示借来的η更好。\n\n')
    cross_names=list(NEW_NAMES)+['xsc_swap_n6'];all_names={**OLD_NAMES,**NEW_NAMES}
    cross_table=['| 来源→目标 | '+' | '.join(all_names[n]for n in cross_names)+' |\n|---|'+'---:|'*len(cross_names)+'\n']
    for a in cross_names:
        values=[]
        for b in cross_names:
            if a==b:values.append('—')
            elif a in s['cross_use_losses']and b in s['cross_use_losses'][a]:values.append(pp(s['cross_use_losses'][a][b]['loss']))
            else:values.append('不适用')
        cross_table.append(f'| {all_names[a]}（{a}） | '+' | '.join(values)+' |\n')
    lines.extend(cross_table);lines.append('\n## 覆盖缺口\n\n固定池只在每场景训练半上选择，再在相应计分半取池内最高率。完整预定组缺场景时不填造完整组结果，下表清楚标记可用子集。\n\n')
    coverage_table=['| 预定组 | 实际计分场景数 | 完整组可计算 | k | 可用组自己的最好η | 可用组固定池覆盖率 | 可用组覆盖缺口(k)（百分点） |\n|---|---:|---|---:|---:|---:|---:|\n']
    for key,g in s['coverage'].items():
        label='4新场景+对穿6机'if key=='new_plus_base'else'4新场景+11现有场景'
        for k,r in g['results'].items():coverage_table.append(f"| {label} | {len(g['available_scenes'])} | {'是'if g['full_requested_group_available']else'否，仅可用子集'} | {k} | {pct(r['own_best_crossvalidated_rate'])} | {pct(r['pool_oracle_scoring_rate'])} | {pp(r['coverage_gap'])} |\n")
    lines.extend(coverage_table)
    lines.append('\n## 按控制器的最好η\n\n');lines.extend(controller_table)
    circle=cohort['lit_circle_five'];outside=next(a for a in actions()if a['key']=='sE27')
    lines.append(f"\n历史口径说明：圆形五机原合格控制器集合的最好候选包含在25候选中；仅限本轮3个控制器时，全49候选的最好为{eta(outside)}，成功率{pct(circle['three_controller_full49_best'])}，预定25候选最高为{pct(circle['three_controller_selected25_best'])}。本轮没有追加候选；所有结论条件于指定25候选，不宣称覆盖连续η空间或完整49候选。\n")
    lines.append('\n能力门、标签缺口与两组不可计算原因见GAPS.md。现有标签直接复用，不为旧场景生成rollout；数值失败保留不重跑，timeout不等于deadlock。保留最终检验的prospective目录未打开，仅审计os.stat元数据。\n')
    if d['meets_expansion_condition']:lines.append('\n满足本轮扩展条件的旋钮：'+'、'.join(NEW_NAMES[n]for n in d['new_need_scenes'])+'。仅报告该结论，由用户决定后续；本轮不追加。\n')
    else:lines.append('\n本轮未达到预登记的扩展条件，不据此否定因能力门或难度范围而无法判断的场景；本轮到此结束，不追加或更换场景。\n')
    (HERE/'REPORT.md').write_text(''.join(lines));(HERE/'CANDIDATE_RATES.md').write_text(''.join(candidate_table));(HERE/'CROSS_USE_LOSSES.md').write_text(''.join(cross_table));(HERE/'COVERAGE_GAPS.md').write_text(''.join(coverage_table));(HERE/'PER_CONTROLLER_BEST.md').write_text(''.join(controller_table))
    return first

def main():
    assert(HERE/'main_COMPLETE.json').exists();database();geometry_finish()
    d=read(HERE/'TEST_DECISION.json');cap=read(HERE/'CAPABILITY_TABLE.json');missing=[n for n,r in cap['scene_status'].items()if not r['test_applicable']]
    (HERE/'GAPS.md').write_text('# 证据缺口与范围\n\n'+''.join(f'- {NEW_NAMES[n]}（{n}）：{cap["scene_status"][n]["status"]}，未跑25候选联合主标签；几何与单机器人有路仍成立，联合可解及死锁解决未认证。\n'for n in missing)+
        '\n完整4新场景+对穿6机、4新场景+11现有场景的覆盖缺口因上述主标签缺失而无法计算，仅附报可用场景子集。未替换场景、未改控制器或参数、未增加候选、未追加第二阶段。\n\n本轮固定25候选对3控制器的结果不等于完整49候选或连续η范围。仅12状态、种子0的新场景小测试，bootstrap区间不包含重新抽取场景/候选族的不确定性。\n')
    first=make_report();integrity();db=read(HERE/'DB_POSTFLIGHT.json');audit=read(HERE/'INTEGRITY_COMPARE.json');cost=read(HERE/'ACTUAL_COST.json')
    assert db['passed']and audit['passed']
    with(HERE/'REPORT.md').open('a')as f:f.write(f"\n结束核对：{db['unique_requests']}个唯一新请求结果均有精确身份、元数据、η和轨迹哈希核对；没有未合并journal。能力门{db['stage_requests']['capability']}条，主标签{db['stage_requests']['main']}条。实测单CPU单局耗时合计{cost['single_cpu_episode_wall_hours_sum']:.3f}小时；主标签墙钟{cost['stage_wall']['main']['seconds']/60:.2f}分钟。两仓库与{audit['source_files']}个复用输入、{audit['frozen_files']}个冻结文件完整性通过；未读取保留新场景。无训练，无搜索/A* rollout，无追加。\n")
    (HERE/'STATUS.md').write_text(f'# 当前状态\n\n{utc()}：COMPLETE。能力门与主标签、统计、决策及结束审计完成。\n\n'+first)
    write(HERE/'DELIVERY_CODE_SHA256.json',dict(files={str(p):sha(p)for p in HERE.glob('*.py')}))
    files=['REPORT.md','CANDIDATE_RATES.md','CROSS_USE_LOSSES.md','COVERAGE_GAPS.md','PER_CONTROLLER_BEST.md','HISTORY_CHECK.md','SCENES.json','GEOMETRY_REPORTS.json','CAPABILITY_TABLE.json','TEST_DECISION.json','STATISTICS.json','STATUS.md','GAPS.md','ITERATION_LOG.md','DB_POSTFLIGHT.json','INTEGRITY_START.json','INTEGRITY_END.json','INTEGRITY_COMPARE.json','TOOLING_COPY.receipt.json','STATISTICAL_CONVENTIONS.json','EXISTING_BEST_SETS.json','HALF_SPLITS.json','DELIVERY_CODE_SHA256.json']
    result=dict(utc=utc(),status='COMPLETE',no_training=True,no_second_stage=True,no_additional_rollouts=True,decision=d['status'],meets_expansion_condition=d['meets_expansion_condition'],
        completed_new_requests=db['unique_requests'],control_inapplicable_scenes=missing,source_and_database_audits_passed=True,reports={n:sha(HERE/n)for n in files})
    write(HERE/'COMPLETE.json',result);write(HERE/'HANDOFF.json',result);write(HERE/'PROGRESS.json',dict(utc=utc(),stage='complete',remaining=0,running=0,pending=0,report='REPORT.md'))
    print(first,flush=True)
if __name__=='__main__':main()
