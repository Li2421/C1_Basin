"""Required model-free menu analysis, strictly before real critic fitting."""
from common import *
from baselines import *
from statistics_pipeline import evaluate_group,equivalence
import csv


def main():
    with np.load(HERE/'TRAINING_BUNDLE.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    types=data['types'];nc=len(types);lo=(data['labels'][...,0]==1).astype(float);controller=best(lo.mean(1));fixed=np.empty(nc,np.int32);meanfixed=np.empty(nc,np.int32)
    for t in sorted(set(types)):
        source=set(types)-{t};fixed[types==t],meanfixed[types==t],_=source_fixed(data,source)
    methods={'all_Sprog':np.full(nc,12),'all_safety_baseline':np.full(nc,10),'all_RHR':np.full(nc,13),'source_fixed':fixed,'source_mean_fixed':meanfixed}
    labels={k:data[k][...,:1] for k in ('labels','uids','collision','timeout','unknown')};labels.update(names=data['names'],types=types);labels['uid_class']=equivalence(labels['uids'])
    output={}
    for name,mask in [('all',np.ones(nc,bool)),('learned',np.isin(types,list(LEARNED))),('nonlearned',np.isin(types,list(NONLEARNED)))]:
        if not mask.any():continue
        subset={k:v[mask] for k,v in labels.items()};choices={k:v[mask] for k,v in methods.items()}
        output[name]={mode:evaluate_group(subset,choices,type_equal=(mode=='type_equal'),replicates=5000) for mode in ('type_equal','controller_equal')}
    rows=[];matrix={}
    for i,name in enumerate(data['names']):
        cls=CLASSES[int(repair_class(controller[i]))];kind=str(types[i]);matrix.setdefault(kind,{c:0 for c in CLASSES})[cls]+=1
        scores=lo[i].mean(0);tied=np.flatnonzero(np.isclose(scores,scores.max(),rtol=0,atol=1e-12))
        rows.append(dict(controller=str(name),type=kind,best_candidate=int(controller[i]),repair_class=cls,optimal_candidates=json.dumps(tied.tolist()),optimal_repair_classes=json.dumps(sorted({CLASSES[int(repair_class(a))] for a in tied})),seed0_best_success=float(lo[i,:,controller[i]].mean()),safety_success=float(lo[i,:,10].mean()),Sprog_success=float(lo[i,:,12].mean()),fixed_RHR_success=float(lo[i,:,13].mean())))
    write(HERE/'SELECTABLE_SPACE.json',dict(utc=utc(),results=output,repair_class_matrix=matrix,controller_preferences=rows,model_predictions_used=False,common_seed=0,training_oracle='in-sample only; generator seed1 missing by original allocation'))
    with (HERE/'REPAIR_CLASS_BY_CONTROLLER.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (HERE/'REPAIR_CLASS_BY_TYPE.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['type',*CLASSES]);writer.writeheader();writer.writerows(dict(type=t,**v) for t,v in matrix.items())
    text=['# 训练状态的模型无关可选空间','','全部为共同seed0样本内证据，先于任何真实critic拟合；不称为交叉拟合。源固定候选只按外层以外的源类型选择。','','| 子集 | 权重 | 进度切换成功率% | 源多数票固定成功率% | 按状态样本内oracle% | 按控制器样本内oracle% |','| --- | --- | --- | --- | --- | --- |']
    for cohort,group in output.items():
        for weight,summary in group.items():
            methods=summary['methods'];value=[methods[k]['success_lower_pct'] for k in ('all_Sprog','source_fixed','state_in_sample_oracle','controller_oracle')]
            text.append('| '+' | '.join([cohort,weight,*[f'{v:.3f}' for v in value]])+' |')
    text.extend(['','完整配对CI及相对两种固定基线的oracle空间见SELECTABLE_SPACE.json。每控制器最佳修法类与按类型计数分别见REPAIR_CLASS_BY_CONTROLLER.csv和REPAIR_CLASS_BY_TYPE.csv；平局按预注册进度锚点/菜单顺序，不掩盖并列最佳。'])
    (HERE/'SELECTABLE_SPACE.md').write_text('\n'.join(text)+'\n')
    seal('SELECTABLE_SPACE.receipt.json',[HERE/'SELECTABLE_SPACE.json',HERE/'SELECTABLE_SPACE.md',HERE/'REPAIR_CLASS_BY_CONTROLLER.csv',HERE/'REPAIR_CLASS_BY_TYPE.csv',HERE/'headroom.py',HERE/'TRAINING_BUNDLE.receipt.json'],before_real_critic_training=True)
    print('Model-free menu headroom and repair classes sealed before fitting')


if __name__=='__main__':main()
