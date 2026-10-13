"""Export a standalone scientific figure and a concise retrospective report."""
from pathlib import Path
import csv
import json
import numpy as np

HERE=Path(__file__).resolve().parent
SEEDS=(17,23,41)
NAMES={'eta_only_original':'Eta only','full_replay':'Full','constant_h':'Fixed h, real C',
       'constant_context':'Real h, fixed C','constant_all':'Fixed h and C'}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((HERE/'intervention_metrics.csv').open()))
    summary=json.loads((HERE/'intervention_summary.json').read_text())
    selection=json.loads((HERE/'selection.json').read_text())['arms']
    contrast=json.loads((HERE/'intervention_contrasts.json').read_text())
    colors={'full_replay':'#b63d36','constant_h':'#d98935','constant_context':'#1974a8','constant_all':'#289468'}
    fig,axes=plt.subplots(2,2,figsize=(12,8.4),layout='constrained')
    for arm in colors:
        histories=[json.loads((HERE/'fits'/arm/f'seed{s}'/'history.json').read_text()) for s in SEEDS]
        steps=[r['step'] for r in histories[0]]
        for ax,metric,title in [(axes[0,0],'TRAIN_NLL','Training probability error'),(axes[0,1],'VAL_NLL','Validation probability error')]:
            values=np.array([[r[metric] for r in h] for h in histories])
            ax.plot(steps,values.mean(0),label=NAMES[arm],color=colors[arm],ls='--' if arm=='constant_h' else '-',lw=2)
            ax.fill_between(steps,values.min(0),values.max(0),color=colors[arm],alpha=.10)
            ax.set(xlabel='Training step',ylabel='Bernoulli NLL (lower is better)',title=title)
            ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8)
    arms=['eta_only_original','full_replay','constant_h','constant_context','constant_all']
    for ax,metric,title in [(axes[1,0],'NLL','Held-out probability error'),(axes[1,1],'B15','Held-out robust selections')]:
        for i,arm in enumerate(arms):
            rr=[r for r in rows if (r['arm'],r['checkpoint'],r['split'])==(arm,'selected','test')]
            vals=np.array([float(r[metric]) for r in rr]);color=colors.get(arm,'#4d4d4d')
            ax.scatter(i+np.array([-.12,0,.12]),vals,c=color,s=38,zorder=3)
            ax.plot([i-.24,i+.24],[vals.mean()]*2,color=color,lw=2)
        ax.set_xticks(range(len(arms)),[NAMES[a].replace(', ','\n').replace(' and ','\nand ') for a in arms],fontsize=8)
        ax.set_title(title)
        ax.grid(axis='y',alpha=.18)
    axes[1,0].set_ylabel('Bernoulli NLL (lower is better)')
    axes[1,1].set_ylabel('States with selected candidate >=15/16 successes')
    axes[1,1].set_ylim(9.7,16.5)
    axes[1,1].set_yticks(range(10,17))
    axes[1,1].axhline(14,color='gray',ls=':',lw=1,label='Candidate-pool oracle =14/16')
    axes[1,1].legend(fontsize=8,loc='upper right')
    fig.suptitle('Ring FF: fixing response context recovers selected-checkpoint performance',fontsize=13)
    fig.supxlabel('32 TRAIN / 8 VAL / 16 TEST states; four ablation arms share network, initialization and minibatches.\nEta-only is the original reference. Dots/ribbons show three training seeds. VAL-selected checkpoints; retrospective audit.',fontsize=9)
    fig.savefig(HERE/'state_context_ablation.png',dpi=180)
    fig.savefig(HERE/'state_context_ablation.pdf')
    plt.close(fig)

    lines=[
        'Ring FF state/context nuisance-input root-cause diagnostic',
        '',
        '结论：支持“当前模型对随state变化的响应context过拟合，削弱eta先验”这一局部机制。',
        '没有支持“显式物理state h本身就是主要噪声来源”，也没有证明state普遍冗余或这是唯一根本原因。',
        '',
        '设计：Ring FF，32个TRAIN family、8个VAL family、16个既有TEST family，固定16个eta。',
        '四组完全相同的full_context网络，每组seed17/23/41；初始化与minibatch序列逐seed校验相同。',
        '完整2x2输入对照：真实/固定h × 真实/固定C。固定值只取TRAIN均值，C按eta分别平均。',
        '固定均值是计算上的输入干预，不声称这些平均张量对应真实可实现物理状态。',
        '训练规则与原实验相同；每组按三seed平均VAL NLL选共同checkpoint。无新rollout。',
        'constant_context为看到初始三组结果后的机制补充，记录在factorial_amendment.json。',
        '',
        'Selected checkpoint results (NLL三seed平均；B15为各seed在16个TEST state中的计数):',
        'arm | step | TRAIN NLL | VAL NLL | TEST NLL | TEST B15',
    ]
    for arm in arms:
        r=next(r for r in summary if r['arm']==arm)
        step=selection[arm]['selected_step'] if arm in selection else 'original300'
        lines.append(f"{NAMES[arm]} | {step} | {np.mean(r['train']['NLL']):.4f} | {np.mean(r['validation']['NLL']):.4f} | {np.mean(r['test']['NLL']):.4f} | {r['test']['B15']}")
    lines.extend([
        '',
        '主要判别：',
        '1. 固定h而保留真实C：结果几乎不变，不能把显式state分支当作本轮主要噪声源。C本身仍含state信息。',
        '2. 保留真实h而固定C：恢复14/16、14/16、14/16；固定全部state变化也恢复同样计数。',
        '3. 所有新组网络容量一致，排除了仅凭eta-only与full网络容量不同而作因果解释的混淆。',
        '4. 完整模型TRAIN误差更低但VAL/TEST更高；训练至2000步TRAIN NLL进一步降至约0.091，VAL却升至约0.765。',
        '5. 仅保留h变化、固定C的模型长训也会过拟合；因此C并非唯一可能发生过拟合的路径。',
        '',
        '按独立TEST family重采样的配对误差差异（负值表示改善；回顾性、描述性区间）：',
    ])
    for a in ['constant_h','constant_context','constant_all']:
        c=next(c for c in contrast if (c['checkpoint'],c['split'],c['candidate'],c['reference'])==('selected','test',a,'full_replay'))
        lines.append(f"{a} - full: delta NLL={c['delta_NLL_mean']:.5f}, 95%={c['delta_NLL_family_bootstrap95']}; B15 delta95%={c['delta_B15_family_bootstrap95']}")
    lines.extend([
        '',
        '不能省略的反证与边界：',
        '- Full的TRAIN-state平均Q仍只能达到14/11/11，说明有害影响也已写入平均eta偏好；仅推理时抹平state并不能完全修复。',
        '- 所有组同在step100比较时，固定C/固定全部只达到14/14/11，完整恢复发生在各组按相同VAL规则选点后。',
        '- Full在step2000的B15也达到14/14/14，但TEST NLL约0.829，远差于所选step100。概率拟合和top1选择并不完全一致，不能将top1差距全部归因于state噪声。',
        '- 固定h+C的TEST NLL仍略差于eta-only，未消除全部概率预测差距。',
        '- 原有8个VAL state上Full的B15是8/8，eta-only是7/8；小验证集未预测到TEST的选择下降。',
        '- 16个TEST family限制二元选择差异的统计精度；三训练seed不构成48个独立state。',
        '- 已知TEST标签后的回顾性诊断，不是新独立确认，也没有区分样本数、context噪声、特征尺度、表示与优化各自的唯一贡献。',
        '- Toy TT的已存证据是正面对照：去除state依赖后NLL由约0.281恶化为0.534，不能推广为state普遍无用。',
        '',
        '保护eta先验的缓存诊断：',
        '只按原VAL NLL选择alpha，Ring FF得到p=0.75*p_eta+0.25*p_full；TEST B15三seed均14/16，平均NLL约0.321。',
        '这是已有概率的收缩诊断，不是新训练的残差网络；恢复表现本身不证明冗余或唯一根因。',
        '',
        '完整文件：protocol.json、factorial_amendment.json、experiment_initial.py、experiment.py、analyze.py、',
        'intervention_metrics.csv、intervention_contrasts.json、selection.json、cached_metrics.csv、',
        'mixture_selection.json、replay_integrity.json、training_replay_parity.json、fits/。',
        '图：state_context_ablation.png / .pdf。',
    ])
    (HERE/'FINAL_REPORT.txt').write_text('\n'.join(lines)+'\n')
    decision=dict(
        verdict='STATE_DEPENDENT_CONTEXT_OVERFITTING_SUPPORTED_IN_RING_FF; UNIQUE_ROOT_CAUSE_NOT_ESTABLISHED',
        raw_state_noise_as_primary_cause_supported=False,
        scoped_context_ablation_recovery_supported=True,
        globally_redundant_state_supported=False,
        retrospective=True,
        independent_test_families=16,
        trained_arms=4,training_seeds=list(SEEDS),new_rollouts=0,
        checkpoint_rule='Mean three-seed source VAL NLL, identical across arms',
        limitation='B15 recovery depends on checkpoint rule; same-step and final-step sensitivities retained; small independent N; context/sampling/representation/optimization not uniquely separated')
    (HERE/'scientific_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print('Wrote FINAL_REPORT.txt, scientific_decision.json, and PNG/PDF figure.')


if __name__=='__main__':main()
