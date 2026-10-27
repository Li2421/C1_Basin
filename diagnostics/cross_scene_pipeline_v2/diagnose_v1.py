"""Read-only recomputation of the supplied post-hoc diagnosis; never a gate."""
from common import *
from anchored import sigmoid
import numpy as np

V=ROOT/'diagnostics/cross_scene_pipeline_v1';O=ROOT/'diagnostics/cross_scene_correction_v1'

def ranking(y,score):
    values=[]
    for a,b in zip(np.asarray(y).reshape(-1,49),np.asarray(score).reshape(-1,49)):
        yes=b[a==1];no=b[a==0]
        if len(yes)and len(no):values.append(float(np.mean((yes[:,None]>no[None])+0.5*(yes[:,None]==no[None]))))
    return float(np.mean(values))if values else None

def cross_state(y,h=None,per_controller=False):
    y=np.asarray(y,float);C,S,A=y.shape;out=np.empty((C,S))
    if h is not None:
        h=np.asarray(h);scale=h.std(0);scale=np.where(scale<1e-6,1,scale);z=(h-h.mean(0))/scale;dist=((z[:,None]-z[None])**2).sum(-1);np.fill_diagonal(dist,np.inf)
    for s in range(S):
        other=np.argsort(dist[s],kind='stable')[:3]if h is not None else np.delete(np.arange(S),s)
        if per_controller:
            q=y[:,other].mean(1)
            for c in range(C):out[c,s]=y[c,s,q[c]==q[c].max()].mean()
        else:
            q=y[:,other].mean((0,1))
            out[:,s]=y[:,s,int(np.argmax(q))]if h is not None else y[:,s,q==q.max()].mean(1)
    return out

def main():
    assert read(HERE/'REPLAY_CHECK.json')['passed']and read(HERE/'REQUEST_RECONSTRUCTION_CHECK.json')['passed']
    v=read_evidence(V/'INPUTS.json');scenes=read_evidence(V/'SCENE_SET.json')['kept'];tables=read_evidence(O/'LABEL_TABLES.json');folds=read_evidence(V/'FOLDS.json')
    completed_v1=read_evidence(V/'EVALUATION_LABEL_TABLES.json')
    for path in(V/'REPORT.md',V/'STATISTICS.json',V/'EVALUATION_LABEL_TABLES.json',V/'REQUEST_CONSTRUCTION_CORRECTION.md',O/'SCENES.json',O/'SELECTED_CONTROLLERS.json',O/'LOSO_CHOICES.json'):
        read_evidence(path)if path.suffix=='.json'else evidence(path)
    history='''# 历史检索与证据复用

本轮为开发轮，严格使用v2说明规定的18个历史场景/11个去重场景。保留最终检验的prospective目录仅检查os.stat元数据，没有打开定义或结果。

| 历史路径 | 兼容性与复用 | 仍缺证据 |
|---|---|---|
| diagnostics/cross_scene_pipeline_v1/ | 完成状态与完整性通过；复用输入、冻结对照、逐格结果、critic预测、生成器提案用于诊断；修正后的构造器逐字复制 | 新描述、新采样及先验损失改变训练问题，旧权重不能替代v2训练；新提案只补精确缺失执行 |
| diagnostics/cross_scene_correction_v1/ | 相同控制器、物理、安全层、事件与种子语义；18×49标签、状态、控制器配置与原请求直接复用；6局逐位重放通过 | 旧49候选不重跑；仅新生成且实际需要的η可能缺标签 |
| diagnostics/cross_scene_homogeneity_audit_v1/ | 事后去重开发依据；场景集沿用v1冻结的11场景 | 不构成新拓扑上的最终检验 |
| diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/best_correction_search_v1/ | 候选精确float64、圆形五机原运行入口、各文献请求语义保留；10例完整规范化JSON和UID重建通过 | 仅新η的缺失结果 |

训练排程按完整的有序源场景集合、描述类型、数据哈希、先验、标准化、模型和种子去重；相同内部训练问题共享同一个检查点，而非重新训练。候选输入编码的变化不使原执行标签失效。所有读取路径及sha256记录在EVIDENCE_INPUTS.jsonl与开工证据清单。
'''
    (HERE/'HISTORY_CHECK.md').write_text(history)
    rows={};prior_rank={'loso':[],'lofo__bottleneck':[],'lofo__literature':[]};critic_rank={k:[]for k in prior_rank}
    for fold in folds:
        if not fold['generator']:continue
        choice=read_evidence(V/'folds'/fold['name']/'CHOICES.json');definition=read_evidence(V/'folds'/fold['name']/'definition.json')
        p0=np.mean([np.asarray(tables[n]['y']).mean((0,1))for n in fold['sources']],axis=0);group='loso'if fold['kind']=='loso'else fold['name']
        for n,ch in choice.items():
            y=np.asarray(tables[n]['y'],float);score=np.asarray(ch['full_logits'])[:,:,:49];pr=np.broadcast_to(p0,score.shape)
            ca=ranking(y,score);pa=ranking(y,pr);critic_rank[group].append(ca);prior_rank[group].append(pa)
            if fold['kind']!='loso':continue
            h=v['scene_inputs'][n]['h'];r=dict(scene_cross_state=float(cross_state(y).mean()),controller_cross_state=float(cross_state(y,per_controller=True).mean()),
                nearest3_other_states=float(cross_state(y,h=h).mean()),critic_ranking_accuracy=ca,prior_ranking_accuracy=pa)
            for key,logits in [('critic',score),('no_controller',np.asarray(ch['no_controller_logits']))]:
                best=logits.argmax(-1);scene=int(sigmoid(logits).mean((0,1)).argmax())
                r[key+'_per_cell']=float(np.take_along_axis(y,best[:,:,None],axis=-1).mean());r[key+'_whole_scene']=float(y[:,:,scene].mean())
            if n=='lit_circle_five':
                norm=definition['normalization'];z=(np.asarray(v['scene_inputs'][n]['c'])-norm['c_mean'])/norm['c_scale'];props=read_evidence(V/'folds'/fold['name']/'PROPOSALS.json')[n]
                means=np.asarray([p['generator_mean']for p in props]);r.update(raw_descriptor_z=z.tolist(),generator_corner_fraction=float(np.all(np.isclose(means,[0,-1.5,1.5],atol=1e-6),axis=1).mean()),
                    contribution_to_cross_state_gap_pp=100*(r['scene_cross_state']-float(np.asarray(completed_v1['loso'][n]['y'])[:,:,completed_v1['loso'][n]['methods'].index('pipeline')].mean()))/11)
            rows[n]=r
    q={n:np.asarray(tables[n]['y']).mean((0,1))for n in scenes};shared=[]
    for i,n in enumerate(scenes):
        for m in scenes[i+1:]:
            regret=float(np.maximum(q[n].max()-q[n],q[m].max()-q[m]).min())
            if regret<.02:shared.append(dict(a=n,b=m,minimum_shared_regret=regret))
    macro={k:float(np.mean([r[k]for r in rows.values()]))for k in next(iter(rows.values()))if isinstance(next(iter(rows.values()))[k],(int,float))}
    expected=dict(scene_cross_state=.7349,controller_cross_state=.7354,nearest3_other_states=.6790,critic_per_cell=.6458,critic_whole_scene=.6732,
        no_controller_per_cell=.6699,no_controller_whole_scene=.7040,critic_ranking_accuracy=.788,prior_ranking_accuracy=.809)
    write(HERE/'V1_DIAGNOSIS.json',dict(utc=utc(),new_rollouts=0,new_training=0,per_scene=rows,macro=macro,provided_reference=expected,
        reference_differences={k:macro[k]-v for k,v in expected.items()},ranking_by_framework={g:dict(critic=float(np.mean(critic_rank[g])),prior=float(np.mean(prior_rank[g])),
        per_scene_critic=critic_rank[g],per_scene_prior=prior_rank[g])for g in prior_rank},retained_scene_pairs=55,shared_near_optimal_pairs=len(shared),shared_pairs=shared,
        differences_are_report_only=True,whole_scene_scores='Mean predicted success probabilities; canonical candidate order for argmax.'))
    print('V1 diagnosis recomputed:',macro,'shared pairs',len(shared),flush=True)

if __name__=='__main__':main()
