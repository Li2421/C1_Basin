"""Readable full-name axis tables plus machine-readable event/regret metrics."""
from core import *
import csv

TYPE_NAMES={'Flow':'环形交换场景流匹配策略','gaussian':'高斯行为克隆策略','deterministic':'确定性行为克隆策略',
 'mixture':'混合密度行为克隆策略','diffusion':'扩散行为克隆策略','chunking':'动作分块行为克隆策略',
 'recurrent':'循环网络行为克隆策略','cvae':'条件变分自编码器行为克隆策略','ring_ippo':'参数共享独立近端策略优化强化学习策略'}
METHOD_NAMES={'all_safety_baseline':'只加安全投影','all_F10':'全程旧默认参数（额外历史参照）','all_S500':'一律按第500步切换时间表','all_Sprog':'一律按进度切换时间表',
 'pipeline':'生成器与看state和controller的critic','pipeline_conservative':'生成器与保守critic',
 'pipeline_scratch':'从头训练的完整critic','pipeline_warm':'从旧critic初始化的完整critic',
 'frozen_controller_selector':'跨type1冻结的按controller选择器','original_pipeline':'原生成器与原critic',
 'generator_only':'生成器均值按进度切换','no_controller':'不看controller的critic','no_state':'不看state的critic',
 'state_crossfit_oracle':'按state交叉拟合事后最佳候选','state_in_sample_oracle':'按state样本内事后最佳候选',
 'controller_oracle':'按controller样本内事后最佳候选','controller_crossfit_oracle':'按controller交叉拟合事后最佳候选',
 'all_nominal_only':'只执行controller动作','all_types_reference':'全部九类型训练模型（参照）'}


def main():
    phases=[('训练state留一类型','DEVELOPMENT_RESULTS_V2B.json','DEVELOPMENT_HELD_TYPE_COMMON_SEED0'),
        ('新评估state','CONFIRMATORY_RESULTS.json','CONFIRMATORY')]
    methods=list(METHOD_NAMES);columns=['阶段','类型','证据等级',*[METHOD_NAMES[k] for k in methods]]
    rows=[];metrics=[];legacy=read(HERE/'LEGACY_F10_TRAINING_REFERENCE.json')['summaries']
    for phase,path,evidence in phases:
        report=read(HERE/path)['summaries']
        for group,r in report.items():
            for weighting,summary in r.items():
                name=TYPE_NAMES[group] if group!='combined' else ('类型等权合并' if weighting=='type_equal' else 'controller等权合并')
                row={'阶段':phase,'类型':name,'证据等级':evidence}
                for key,label in METHOD_NAMES.items():
                    m=summary['methods'].get(key)
                    if phase=='训练state留一类型' and key=='all_F10':m=legacy[group][weighting]
                    if m is None:
                        row[label]='N/A（训练生成候选只有seed0）' if 'crossfit' in key else 'N/A（本阶段未要求）'
                        continue
                    ci=m['success_CI95_pct'];row[label]=f"{m['success_lower_pct']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"
                    m=dict(m)
                    if 'evidence_level' in m:m['reference_evidence_level']=m.pop('evidence_level')
                    if phase=='训练state留一类型' and key in legacy[group][weighting]['disaster_count_by_method']:
                        count=legacy[group][weighting]['disaster_count_by_method'][key]
                        m.update(disasters_vs_F10=count['possible'],disasters_vs_F10_certain=count['certain'],disaster_unavailable_reason=None)
                    metrics.append(dict(phase=phase,type=name,weighting=weighting,method=label,method_key=key,evidence_level=evidence,**m))
                rows.append(row)
    with (HERE/'AXIS_TABLE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
    text=['# 跨类型生成器与critic结果','',
        '成功率单位为%，单元格为按unknown下界计的点估计 [配对bootstrap 95%区间]；区间同时覆盖unknown上界。训练阶段统一使用seed0；完整候选集的seed交叉拟合仅在新评估阶段可识别。',
        '', '| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    text.extend('| '+' | '.join(str(r[c]) for c in columns)+' |' for r in rows)
    text.extend(['','碰撞率、超时率、unknown率、平均与最大遗憾、伤害数和灾难数详见 AXIS_METRICS.json。',
        '全程旧默认参数（F10）未列入本轮十三候选；训练state复用其历史精确标签计算灾难数，新state的灾难数为N/A，未追加仿真。',
        '旧轮次按controller选择器直接使用冻结模型。round16的其他固定菜单方法不属于本轮候选空间，列为N/A。'])
    (HERE/'AXIS_TABLE.md').write_text('\n'.join(text)+'\n')
    write(HERE/'AXIS_METRICS.json',metrics)
    seal('AXIS_TABLE.receipt.json',[HERE/'AXIS_TABLE.csv',HERE/'AXIS_TABLE.md',HERE/'AXIS_METRICS.json',
        HERE/'deliver_tables.py',HERE/'DEVELOPMENT_RESULTS_V2B.receipt.json',HERE/'CONFIRMATORY_RESULTS.receipt.json',HERE/'LEGACY_F10_TRAINING_REFERENCE.receipt.json'])


if __name__=='__main__':main()
