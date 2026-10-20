"""Complete per-type and equal-weight tables, including explicit evidence gaps."""
from common import *
import csv
TYPE_NAMES={'Flow':'环形交换场景流匹配策略','gaussian':'高斯行为克隆策略','deterministic':'确定性行为克隆策略','mixture':'混合密度行为克隆策略','diffusion':'扩散行为克隆策略','chunking':'动作分块行为克隆策略','recurrent':'循环网络行为克隆策略','cvae':'条件变分自编码器行为克隆策略','ring_ippo':'参数共享独立近端策略优化强化学习策略'}
METHODS={'pipeline':'冻结的主修正流程','self_schedule':'自学两个η与切换时刻的修正流程','fixed500':'学习两个η、固定第500步切换的修正流程','no_schedule':'不带时间表的修正流程','generator_mean':'整局固定生成器均值','all_S500':'一律前500步η13、之后η10','all_Sprog':'一律按进度从η13切到η10','legacy_pipeline':'跨类型1b冻结修正流程（后期η10）','no_controller':'不看控制器的critic','conservative':'保守修正流程','all_safety_baseline':'只加安全投影','nominal_only':'只执行控制器动作','all_types_reference':'全部九类型训练模型（参照）','all_F10':'全程旧默认参数η10（历史参照）','evaluated_state_crossfit_oracle':'已评估动作中的按状态交叉拟合事后最佳','evaluated_state_in_sample_oracle':'已评估动作中的按状态样本内事后最佳','A3':'原冻结按控制器选择器','A0':'原冻结不看控制器选择器','GE':'原冻结探测规则','full_menu_oracle':'全部320候选事后最佳','controller_fixed_oracle':'按控制器最佳固定候选','expert':'特权专家','schedule_generator_pipeline':'冻结生成器加专门时间表生成器的修正流程'}
def reason(method,phase):
    if method=='schedule_generator_pipeline':return '可选专门时间表生成器未启用；不把此版本冒充已评估'
    if method=='all_types_reference':return '本部分是留一类型开发，全部类型模型仅在新状态作参照'
    if method=='all_F10':return '没有完整同身份历史记录；新状态未追加此未要求的固定动作'
    if method=='evaluated_state_crossfit_oracle':return '训练纯生成器动作只有seed0，交叉seed参照不可识别'
    if method=='full_menu_oracle':return '本轮只评估稀疏动作，不能识别全部320候选上的oracle'
    if method=='controller_fixed_oracle':return '不同控制器/状态的稀疏菜单不同，没有同一候选跨状态完整覆盖'
    if method in ('A3','A0','GE'):return '原方法采用另一固定菜单，本轮未评估其完整所需动作；不外推标签'
    if method=='expert':return '没有本轮同状态同执行链的特权专家标签'
    return '本阶段无完整同身份标签；保留缺口，不补造'
def main():
    dev=read(HERE/'DEVELOPMENT_RESULTS.json');confirm=read(HERE/'CONFIRMATORY_RESULTS.json');rows=[];metrics=[]
    for phase,source in [('训练状态留一类型',dev['reports']),('全新评估状态',confirm['reports'])]:
        for kind in [*sorted(TYPE_NAMES),'combined']:
            for weighting in (['type_equal','controller_equal'] if kind=='combined' else ['type_equal']):
                report=source[kind][weighting];label=('类型等权合并' if weighting=='type_equal' else '控制器等权合并') if kind=='combined' else TYPE_NAMES[kind]+f'（{kind}）';evidence='DEVELOPMENT_HELD_TYPE_SPARSE_SEED0' if phase.startswith('训练') else 'CONFIRMATORY_HELD_TYPE_NEW_STATE';row={'阶段':phase,'类型':label,'证据等级':evidence}
                for key,title in METHODS.items():
                    if key in report['methods']:
                        value=report['methods'][key];ci=value['success_CI95_pct'];row[title]=f"{value['success_lower_pct']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]";metrics.append(dict(phase=phase,type=kind,weighting=weighting,method=key,method_name=title,evidence='SEEN_TYPES_REFERENCE' if key=='all_types_reference' else evidence,**value))
                    else:row[title]='N/A（'+reason(key,phase)+'）';metrics.append(dict(phase=phase,type=kind,weighting=weighting,method=key,method_name=title,evidence='GAP',reason=reason(key,phase)))
                rows.append(row)
    with (HERE/'AXIS_TABLE.csv').open('w') as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 双η与切换时刻学习结果','', '成功率单位%，区间为5000次配对bootstrap的95%区间，并包络unknown上下界。主统计类型→控制器→状态重抽，两个seed同簇。','', '| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']
    for row in rows:lines.append('| '+' | '.join(str(v).replace('|','/') for v in row.values())+' |')
    lines+=['','训练评分只在每个控制器/状态实际标出的26个纯生成器动作上；其外的人定时间表与旧修正流程只用于独立对照，不进入新critic训练。新状态的oracle仅是已评估的动作中的参照，不是完整320候选的最优值。两阶段oracle池不同，不能把其差异当作模型进步。','', '对角元没有实际切换，时间表选择比例按两个η是否相同判定；选中了非对角对也不保证episode在结束前执行过后期η，另见PIPELINE_SWITCH_EXECUTION。全部九类型训练模型见过每个类型，只作新状态参照。','', '碰撞率、超时率、unknown率、平均/最大控制器遗憾、伤害数、可识别的历史F10灾难数、非对角对比例及证据等级见AXIS_METRICS.json/CSV。负交叉拟合遗憾保留，有限seed参照不是严格数学下界；unknown界条件于按下界选定的oracle臂。']
    (HERE/'AXIS_TABLE.md').write_text('\n'.join(lines)+'\n');write(HERE/'AXIS_METRICS.json',dict(utc=utc(),metrics=metrics,development=dev,confirmation=confirm,oracle_not_full_menu=True))
    with (HERE/'AXIS_METRICS.csv').open('w') as f:
        keys=list(dict.fromkeys(k for r in metrics for k in r));writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in metrics)
    base=read(HERE/'BASE_PROTOCOL.json');zoo=[]
    for name in base['controller_order']:
        p=base['profiles'][name];v=p['payload'];kind=base['types'][name];zoo.append(dict(controller=name,type=kind,type_name=TYPE_NAMES[kind],family='learned',checkpoint=v.get('flow_checkpoint_path'),checkpoint_sha256=v.get('flow_checkpoint_sha256'),controller_uid=p['controller_uid'],placement='FF' if kind=='Flow' else 'TT',source='Cross-type1/1b qualified zoo; no retraining',parameters=json.dumps(v.get('nominal_parameters'),ensure_ascii=False)))
    with (HERE/'ZOO_MANIFEST.csv').open('w') as f:writer=csv.DictWriter(f,fieldnames=list(zoo[0]));writer.writeheader();writer.writerows(zoo)
    seal('AXIS_TABLE.receipt.json',[HERE/n for n in ('AXIS_TABLE.csv','AXIS_TABLE.md','AXIS_METRICS.json','AXIS_METRICS.csv','ZOO_MANIFEST.csv','deliver_tables.py')])
    print('Axis tables complete:',len(rows),'rows;',len(zoo),'unchanged qualified controllers',flush=True)
if __name__=='__main__':main()
