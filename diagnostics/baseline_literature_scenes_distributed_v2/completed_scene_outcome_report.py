"""Readable completed-scope outcome tables, separate from the final REPORT."""
from experiment import *
from registry import verify_receipt


METHODS={'safety':'只加安全投影','fixed_eta10':'全程固定 η10 修正','fixed_eta13':'全程固定 η13 修正','human_progress':'人定双 η 广播进度切换','learned_scene':'场景内重新学习修正'}
LAYERS={'central':'集中式联合投影','distributed':'分布式逐机器人投影'}


def pct(row):
    lo,hi=row['ci95']
    return f"{100*row['success_rate']:.2f}% [{100*lo:.2f}, {100*hi:.2f}]"


def delta(row):
    lo,hi=row['ci95']
    return f"{100*row['delta']:+.2f} pp [{100*lo:+.2f}, {100*hi:+.2f}]"


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])


def main():
    if (HERE/'COMPLETED_SCENE_OUTCOME_REPORT.receipt.json').exists():
        verify_receipt('COMPLETED_SCENE_OUTCOME_REPORT.receipt.json');return
    verify_receipt('COMPLETED_SCENE_NUMBERS.receipt.json')
    results=read(HERE/'COMPLETED_SCENE_RESULTS.json')['results']
    full=[];learned=[];gains=[]
    for scene,entry in results.items():
        fixed=entry['fixed_full']
        for layer in LAYERS:
            rows={r['method']:r for r in fixed['statistics'] if r['type']=='type_equal' and r['layer']==layer}
            full.append([NAMES[scene],LAYERS[layer],rows['safety']['controllers'],*[pct(rows[m]) for m in list(METHODS)[:4]]])
        distributed={r['method']:r for r in fixed['statistics'] if r['type']=='type_equal' and r['layer']=='distributed'}
        comparisons={r['method']:r for r in fixed['layer_comparisons'] if r['type']=='type_equal'}
        for method in list(METHODS)[1:4]:
            gains.append([NAMES[scene],METHODS[method],delta(distributed[method]['delta_vs_safety']),delta(comparisons[method]['correction_gain_difference'])])
        evaluation=entry['learned_evaluation']
        for layer in LAYERS:
            rows={r['method']:r for r in evaluation.get('statistics',[]) if r['type']=='type_equal' and r['layer']==layer}
            if 'learned_scene' not in rows:continue
            best=max(list(METHODS)[1:4],key=lambda m:rows[m]['success_rate'])
            learned.append([NAMES[scene],LAYERS[layer],pct(rows['safety']),METHODS[best]+'：'+pct(rows[best]),pct(rows['learned_scene']),delta(rows['learned_scene']['delta_vs_best_aggregate_fixed'])])
    paras=['本轮开发证据的阶段性结果。四场景的只加安全投影与三种固定修正均已完成；窄通道、二维拥挤随机、门口的场景内重新学习评估已完成；圆形围场的重新学习仍在运行。本文件不表示第一阶段或完整 v2 已结束，最终交付仍以 REPORT.md、MIGRATION_NUMBERS.receipt.json、DB_POSTFLIGHT.json、INTEGRITY_END.json 和 V2_COMPLETE.json 为准。',
        '以下均为能力门合格实例、八类黑盒控制器类型等权率，包含 controller 级失败。类型→实例→state 成簇，两个 seed 同簇，5000次配对 bootstrap，数值为成功率下界 [95% CI]。这与此前 LIVE 中按局池化的成功率口径不同。逐类型数值见 COMPLETED_SCENE_RESULTS.csv，unknown 上界、伤害数和配对差的识别界见同名 JSON。',
        '全部主 state 的固定修正：\n\n'+table(['场景','安全层','合格实例','只加安全投影','全程固定 η10 修正','全程固定 η13 修正','人定双 η 广播进度切换'],full),
        '重新学习的独立评估 state：\n\n'+table(['场景','安全层','同范围基线','同范围最佳固定修正','场景内重新学习','重新学习−最佳固定修正'],learned),
        '重新学习比较只使用各自共同独立评估 state，不与上一表全部主 state 混比。每次 bootstrap 重新选择最佳聚合固定修正。集中式重新学习只复用已有窄通道与门口结果；圆形围场尚未出重新学习结果。窄通道预注册菜单的6144条训练标签均为失败，密封选择全部为动作0；这组数据没有观测到成功动作，不支持评价模型识别成功动作的能力。',
        '分布式修正的增益及与集中式的差异：\n\n'+table(['场景','做法','分布式修正−分布式基线','修正增益：分布式−集中式'],gains),
        '白盒朝目标参照单独见 WHITEBOX_REFERENCE.md，不混入本表。死锁分类、controller 级失败剔除分析和诊断发生率在本阶段性成功率报告中为 N/A，必要的保存轨迹诊断仍在运行；不把超时当作死锁。本文没有新增 rollout、训练或改变任何判定标准。']
    (HERE/'COMPLETED_SCENE_RESULTS.md').write_text('\n\n'.join(paras)+'\n')
    seal('COMPLETED_SCENE_OUTCOME_REPORT.receipt.json',[HERE/n for n in ['COMPLETED_SCENE_RESULTS.md','completed_scene_outcome_report.py','COMPLETED_SCENE_NUMBERS.receipt.json']],full_main_complete=False,new_rollouts=0,diagnosis_not_imputed=True)


if __name__=='__main__':main()
