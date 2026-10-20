from core import *
import csv

TYPE_NAMES={'Flow':'环形交换场景流匹配策略','gaussian':'高斯行为克隆策略','deterministic':'确定性行为克隆策略','mixture':'混合密度行为克隆策略','diffusion':'扩散行为克隆策略','chunking':'动作分块行为克隆策略','recurrent':'循环网络行为克隆策略','cvae':'条件变分自编码器行为克隆策略','ring_ippo':'参数共享独立近端策略优化强化学习策略','distributed_shooting':'分布式模型预测控制','centralized_sampling':'集中式随机采样模型预测控制','cross_entropy':'交叉熵方法模型预测控制','prioritized_astar':'优先级时空A*规划','ring_ippo_conservative':'保守参数共享独立近端策略优化强化学习策略','distributed_cross_entropy':'分布式交叉熵方法模型预测控制','centralized_mppi':'集中式模型预测路径积分控制','implicit':'隐式能量行为克隆策略'}
GROUP_NAMES={'development':'开发类型合并','learned':'学习型合并','nonlearned':'非学习型开发类型合并','locked':'锁定类型合并','new_development':'非学习型与保守强化学习开发类型合并','locked_and_new_development':'锁定类型与新增开发类型合并'}
METHOD_NAMES={'all_safety_baseline':'一律只加安全投影','all_S500':'一律按第500步切换时间表','all_Sprog':'一律按进度切换时间表','all_RHR':'一律固定参数持续右手规则','all_RHR_b':'一律短名单持续右手规则一','all_RHR_c':'一律短名单持续右手规则二','pipeline':'生成器与看状态和控制器的critic','pipeline_conservative':'生成器与保守critic','source_fixed':'源类型多数票固定候选','source_mean_fixed':'源类型等权平均最优固定候选','old_pipeline':'跨类型前轮冻结pipeline','original_pipeline':'原生成器与原critic','no_controller':'不看控制器的critic','no_state':'不看状态的critic','few_2':'前两个训练状态选固定候选','few_4':'前四个训练状态选固定候选','few_8':'前八个训练状态选固定候选','state_crossfit_oracle':'按状态交叉拟合事后最佳','state_in_sample_oracle':'按状态样本内事后最佳','controller_crossfit_oracle':'按控制器交叉拟合事后最佳','controller_oracle':'按控制器样本内事后最佳','pipeline_single':'单分支完整critic开发版本','pipeline_scratch':'从头训练完整pipeline','pipeline_warm':'旧critic初始化完整pipeline','generator_only':'生成器均值按进度切换','all_nominal_only':'只执行控制器动作','all_F10':'全程旧默认参数','A3':'原冻结控制器特征选择模型','A0':'原冻结无控制器特征模型','G_E':'原冻结探测规则选择方法','expert':'特权专家参照'}


def main():
    records=[];rows=[];columns=['阶段','类型','权重','证据等级',*METHOD_NAMES.values()];legacy=read(HERE/'LEGACY_F10_REFERENCE.json')['types']
    for phase,file,evidence in [('训练状态留一类型','DEVELOPMENT_RESULTS.json','DEVELOPMENT'),('新评估状态','CONFIRMATORY_RESULTS.json','CONFIRMATORY')]:
        packet=read(HERE/file);report=packet['reports' if phase=='训练状态留一类型' else 'results']
        for group in [*TYPE_NAMES,'development','learned','nonlearned','locked']:
            for weighting in (('type_equal','controller_equal') if group in GROUP_NAMES and group in report else ('type_equal',)):
                summary=report.get(group,{}).get(weighting);name=TYPE_NAMES.get(group,GROUP_NAMES.get(group,group));row={'阶段':phase,'类型':name,'权重':'类型等权' if weighting=='type_equal' else '控制器等权','证据等级':evidence}
                for key,label in METHOD_NAMES.items():
                    value=summary['methods'].get(key) if summary else None
                    if phase=='训练状态留一类型' and key=='all_F10' and group in legacy:
                        value=legacy[group].get('reference_metrics')
                    if value is None:
                        if group=='implicit':reason='缺口：原合成门未通过'
                        elif summary is None:reason='本阶段不属于开发检验范围'
                        elif key=='all_F10':reason='菜单外；历史灾难数单列'
                        elif key in ('A3','A0','G_E'):reason='原固定η菜单不在本轮候选中'
                        elif key=='expert':reason='本轮未要求匹配专家标签'
                        elif 'crossfit' in key:reason='训练完整菜单仅seed0'
                        else:reason='本阶段未要求'
                        row[label]='N/A（'+reason+'）';continue
                    ci=value['success_CI95_pct'];row[label]=f"{value['success_lower_pct']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]";value=dict(value)
                    if phase=='训练状态留一类型' and group in legacy and key in legacy[group]['methods']:
                        count=legacy[group]['methods'][key];value['disasters_vs_F10']=count['possible'];value['disasters_vs_F10_certain']=count['certain']
                        value['disaster_unavailable_reason']=count.get('reason')
                    records.append(dict(phase=phase,type=name,type_code=group,weighting=weighting,method=label,method_key=key,evidence_level=evidence,**value))
                rows.append(row)
    with (HERE/'AXIS_TABLE.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    text=['# 混合黑盒动物园结果','','成功率为% [95%区间]。区间按配对分层bootstrap计算并覆盖unknown下/上界；训练状态用共同seed0，交叉拟合oracle仅新评估状态可识别。非学习型开发合并只含四个非学习型开发类型；锁定类型另列。','','| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    text.extend('| '+' | '.join(str(row[c]) for c in columns)+' |' for row in rows)
    text.extend(['','碰撞、超时、unknown、平均与最大遗憾、灾难数、伤害数与证据等级见AXIS_METRICS.json。前轮冻结pipeline直接用原九类型最终模型及原24维特征、13候选，没有持续右手规则候选。','不看状态的critic仍读取随状态改变的生成器候选，不能解释为整条pipeline完全没有状态信息。','oracle上下界条件于按unknown下界选定的臂；交叉拟合有限seed选择有噪声，可能得到负遗憾。','全程旧默认参数没有本轮新状态标签；训练仅以精确历史身份复用，缺失时不伪造灾难数。'])
    text.append('源类型固定方法、少量标签方法和按控制器oracle固定的是候选编号；若编号来自生成器，实际η仍随状态变化，不把它误称为完全不读取状态的常数η。')
    (HERE/'AXIS_TABLE.md').write_text('\n'.join(text)+'\n');write(HERE/'AXIS_METRICS.json',records)
    seal('AXIS_TABLE.receipt.json',[HERE/n for n in ('AXIS_TABLE.md','AXIS_TABLE.csv','AXIS_METRICS.json','deliver_tables.py','DEVELOPMENT_RESULTS.receipt.json','CONFIRMATORY_RESULTS.receipt.json','LEGACY_F10_REFERENCE.receipt.json')])
    print('Delivered',len(rows),'axis-table rows and',len(records),'metric records')


if __name__=='__main__':main()
