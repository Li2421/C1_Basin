"""Primary matched comparisons and the requested15-action training ranking."""
from experiment import *
from registry import verify_receipt
import csv


def find(summary,method,typ='type_equal',layer='distributed'):
    return next((r for r in summary.get('statistics',[]) if (r['type'],r['layer'],r['method'])==(typ,layer,method)),None)


def primary_tables(results,save,cell,delta,table):
    chunks=[];flat=[];rankings=[];gains=[]
    labels={r['type_code']:r['type'] for r in read(HERE/'CONTROLLER_MANIFEST.json')}
    for scene in NAMES:
        prefix='TRAIN_SELECTED_FIXED_20261022_'+scene;verify_receipt(prefix+'.receipt.json');selection=read(HERE/(prefix+'.json'))
        summary=results[scene]['revised']['primary_evaluation'];rows=[]
        for typ in sorted({r['type'] for r in summary.get('statistics',[])}-{'type_equal'})+['type_equal']:
            base=find(summary,'safety',typ);fixed=find(summary,'train_selected_fixed',typ);learned=find(summary,'learned_scene',typ)
            diff=learned.get('delta_vs_train_selected_fixed') if learned else None
            rows.append([labels.get(typ,'类型等权合并'),selection['selected_menu_index'],cell(base),cell(fixed),cell(learned),delta(diff)])
            flat.append(dict(scene=scene,type=typ,state_scope='held_out_learning_evaluation',selected_menu_index=selection['selected_menu_index'],baseline=base,train_selected_fixed=fixed,learned=learned,learned_minus_train_selected_fixed=diff))
        chunks.append(NAMES[scene]+'；分布式修订语义；独立评估 state\n\n'+table(['controller 类型','训练选定菜单索引（0起）','只加安全投影','最佳固定动作','场景内学习','学习−最佳固定'],rows))
        reference=results[scene]['revised']['fixed_full'];central=[]
        for typ in sorted({r['type'] for r in reference.get('statistics',[])}-{'type_equal'})+['type_equal']:
            central.append([labels.get(typ,'类型等权合并'),*[cell(find(reference,m,typ,'central')) for m in ('safety','fixed_eta10','fixed_eta13','human_progress')]])
        chunks.append(NAMES[scene]+'；集中式修订语义；全部主 state（集中式无菜单标签，不选菜单最佳固定动作）\n\n'+table(['controller 类型','只加安全投影','全程η10','全程η13','人定双η'],central))
        ranking_rows=[[r['rank'],r['menu_index'],f"{100*r['type_equal_success_lower']:.6f}%",f"{100*r['type_equal_success_upper']:.6f}%",f"{r['successes']}/{r['requests']}",r['unknowns']] for r in selection['ranking']]
        rankings.append(NAMES[scene]+'\n\n'+table(['排名','菜单索引（0起）','类型等权成功下界','上界','成功/标签数','unknown'],ranking_rows))
        scope_rows=[]
        for scope,entry in results[scene]['revised']['primary_instance_class_gain_splits'].items():
            fixed=find(entry,'train_selected_fixed');learned=find(entry,'learned_scene')
            scope_rows.append([scope,cell(fixed),delta(fixed.get('delta_vs_safety') if fixed else None),cell(learned),delta(learned.get('delta_vs_safety') if learned else None),delta(learned.get('delta_vs_train_selected_fixed') if learned else None),entry.get('gain_interpretation')])
        gains.append(NAMES[scene]+'\n\n'+table(['实例范围','最佳固定成功','最佳固定−安全','学习成功','学习−安全','学习−最佳固定','解释'],scope_rows))
    save('AXIS_TABLE.md','分布式主固定对照是每场景在15菜单动作中按修订训练标签类型等权成功率选择并冻结的一个动作；平局按菜单顺序，评估结果不用于选择。固定动作与学习在同一独立评估 state、controller 与 seed 上成对比较；bootstrap 不重选动作。全部主表采用类型→controller→state等权及原5000次 bootstrap。unknown 的下界记失败，上界保留。集中式仍报三种固定修正，主state范围与分布式独立评估范围分开。分布式η10、η13、人定双η及v1/旧语义衔接参照见附表 APPENDIX_REFERENCE_AXIS.md，不再作为主最佳固定标准。\n\n'+'\n\n'.join(chunks))
    with (HERE/'pending_external_audit_20261022'/'AXIS_TABLE.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows({k:canonical(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in flat)
    save('MENU_TRAINING_ACTION_RANKING_20261022.md','每场景8类型×2个训练controller×32训练state×15动作。unknown 不删，主选择用成功下界并报告上界；每类型样本数相同，整数成功数保证平局精确。菜单动作原始η和切换参数、逐类型计数及封存选择见 TRAIN_SELECTED_FIXED_20261022_<scene>.json。圆形围场标签完整后才选择。排序只描述训练标签，不能当独立评估性能。\n\n'+'\n\n'.join(rankings))
    save('PRIMARY_CLASS_GAINS_20261022.md','新版分类把逐机器人滤波器平衡点死锁与被已到达者阻挡合并用于安全层全局失败多数判断；死锁证据只计前者。只排除controller级失败。以下固定动作均为全场景训练集选出的同一个动作，不在实例子组重选。controller级增益不计死锁解决；其他范围的成功改善也须结合实际死锁诊断。\n\n'+'\n\n'.join(gains))


def report_lead(results,cell,delta):
    lines=['本报告按用户补充修正采用训练标签选定的最佳固定动作作为分布式主对照。每场景在15动作中按训练state类型等权成功率选一次，平局按菜单顺序；选择文件先封存，评估在同一独立state、controller和seed上配对，bootstrap不重选。η10、η13、人定双η在分布式仅为附表参照，集中式仍只报这三种固定修正。']
    for scene in NAMES:
        selection=read(HERE/('TRAIN_SELECTED_FIXED_20261022_'+scene+'.json'));summary=results[scene]['revised']['primary_evaluation'];fixed=find(summary,'train_selected_fixed');learned=find(summary,'learned_scene')
        lines.append(NAMES[scene]+f"：训练选定菜单{selection['selected_menu_index']}（0起），独立评估最佳固定成功"+cell(fixed)+'，学习成功'+cell(learned)+'，学习−最佳固定'+delta(learned.get('delta_vs_train_selected_fixed') if learned else None)+'。')
    lines.append('主表见AXIS_TABLE.md；各场景15动作训练成功率完整排序见MENU_TRAINING_ACTION_RANKING_20261022.md；按新版实例类拆分的主对照增益见PRIMARY_CLASS_GAINS_20261022.md。')
    return '\n\n'.join(lines)
