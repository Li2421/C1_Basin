"""Readable old-nine regression and leave-family-out summaries from saved labels."""
from common import *
from labels import type_name,method_name
from statistics_pipeline import evaluate_group
from marginal_statistics import evaluate

def subset_report(data,methods):
    available={};missing=[];nc,ns=data['labels'].shape[:2]
    for key,choice in methods.items():
        chosen=data['labels'][np.arange(nc)[:,None],np.arange(ns)[None],choice]
        if np.any(chosen==-2):missing.append(key)
        else:available[key]=choice
    fn=evaluate_group if 'all_safety_baseline' in available else evaluate
    return dict(type_equal=fn(data,available,True,5000),controller_equal=fn(data,available,False,5000),unavailable_methods=missing)

def main():
    with np.load(HERE/'DEVELOPMENT_EVALUATION_BUNDLE.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(HERE/'DEVELOPMENT_ANALYSIS_CHOICES.npz',allow_pickle=False) as z:methods={k:z[k] for k in z.files if k not in ('names','types')}
    old=read(D/'crosstype1c_two_eta_schedule_v1/BASE_PROTOCOL.json');mask=np.isin(data['names'],old['controller_order']);assert int(mask.sum())==62 and len(set(data['types'][mask]))==9
    regression=subset_report({k:v[mask] for k,v in data.items()},{k:v[mask] for k,v in methods.items()});familyrows={};primary=read(HERE/'FROZEN_PIPELINE.json')['primary_method'];familymodels=read(HERE/'LEAVE_ONE_FAMILY_MODELS.json')
    with np.load(HERE/'LEAVE_ONE_FAMILY_RESULTS.npz',allow_pickle=False) as z:
        for record in familymodels:
            family=record['held_family'];names=list(z[family+'_names']);selected=np.asarray([list(data['names']).index(n) for n in names]);subset={k:v[selected] for k,v in data.items()};choice={k:z[family+'_'+k] for k in ('self_schedule','fixed500','no_schedule','conservative')};choice['pipeline']=choice[primary]
            for key in ('all_safety_baseline','all_S500','all_Sprog'):choice[key]=methods[key][selected]
            report=subset_report(subset,choice);familyrows[family]=dict(**report,held_types=record['held_types'],source_types=record['source_types'],ranking=record['ranking'])
    write(HERE/'SUPPLEMENTARY_DEVELOPMENT.json',dict(utc=utc(),evidence='DEVELOPMENT_ONLY_CACHED_TRAINING_LABELS',original_nine_types_regression=regression,leave_one_family=familyrows,no_new_rollouts=True,no_model_or_main_method_change=True))
    lines=['# 原九类学习型回归与留一族检验（DEVELOPMENT）\n','这些结果全部来自已经封存的训练状态标签，不增加rollout，不参与模型或主方法判定。仅原62个学习实例、相同48状态和相同26动作的比较用于回答原九类学习型是否退步，保守强化学习不混入这一合并值。\n']
    def cell(m):return f"{m['success_lower_pct']:.2f}% [{m['success_CI95_pct'][0]:.2f}, {m['success_CI95_pct'][1]:.2f}]"
    methods0=regression['type_equal']['methods'];difference=regression['type_equal']['paired_comparisons']['pipeline__minus__legacy_pipeline'];lines += ['原九类型等权：新混合动物园主方法 '+cell(methods0['pipeline'])+'；旧跨类型1c冻结主方法 '+cell(methods0['legacy_pipeline'])+f"。配对差 {difference['gain_pp']:+.2f}pp，95% CI {difference['CI95_pp']}。\n"]
    lines += ['| 留出族 | 主方法 | 固定第500步 | 不带时间表 | 源类型 |','| --- | --- | --- | --- | --- |']
    for family,r in familyrows.items():
        m=r['type_equal']['methods'];lines.append('| '+type_name(family)+' | '+' | '.join(cell(m[k]) for k in ('pipeline','fixed500','no_schedule'))+' | '+ '、'.join(type_name(t) for t in r['source_types'])+' |')
    lines += ['\n类型等权与实例等权的完整数值、四种排序准确率、支持数量及不完整人工参照覆盖见SUPPLEMENTARY_DEVELOPMENT.json。留出族之外的源类型内部选择checkpoint与分支权重，未使用留出族标签选参。']
    (HERE/'SUPPLEMENTARY_DEVELOPMENT.md').write_text('\n'.join(lines)+'\n')
    seal('SUPPLEMENTARY_DEVELOPMENT.receipt.json',[HERE/'SUPPLEMENTARY_DEVELOPMENT.json',HERE/'SUPPLEMENTARY_DEVELOPMENT.md',HERE/'supplementary_development.py',HERE/'LEAVE_ONE_FAMILY_RESULTS.receipt.json',HERE/'DEVELOPMENT_RESULTS.receipt.json'],development_only=True,no_rollouts_or_retuning=True)
    print('Original-nine regression and3 leave-family-out summaries completed',flush=True)
if __name__=='__main__':main()
