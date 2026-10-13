"""Reproducible adjudication of real-Q and synthetic state-only controls."""
from pathlib import Path
import sys,csv,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from diagnostics.state_nuisance_rootcause_v1.state_only_experiment import OUT,HERE,SOURCE,SEEDS,read,write,sha,csvout,state_effect


def report():
    from scipy.special import expit
    from diagnostics.state_nuisance_rootcause_v1.state_signal_path import counts,residual,corr
    p=read(OUT/'protocol.json');d=np.load(SOURCE/'dataset.npz');source_truth=np.load(HERE/'state_signal_path/exact_outcomes.npz')['outcomes']
    tr=np.array(p['splits']['train']);te=np.array(p['splits']['test'])
    s,f,u=counts(source_truth[1,tr,:,4:]);mask=u==0
    old=np.load(OUT/'test_logits.npz');table=[];extra=[]
    primary=read(OUT/'summary.json');synthetic=read(OUT/'synthetic_control/summary.json')
    real_test_probs={}
    for arm in ('entity_state_eta','local_state_eta','local_state_16heads','local_state_16heads_shuffled','minimal23','minimal15'):
        if arm.startswith('minimal'):
            result=read(OUT/arm/'summary.json');chosen=result['selected_step'];zz=np.load(OUT/arm/'logits.npz')
            get=lambda seed,step,split:zz[f'{seed}__{step}__{split}']
            row=result['results']['2000']
        else:
            chosen=primary[arm]['VAL_NLL']['step']
            get=lambda seed,step,split:old[f'{arm}__{seed}__{step}'] if split=='test' else np.load(OUT/'fits'/arm/f'seed{seed}'/f'step{step}_source.npz')[split]
            row=primary[arm]['last']
        table.append(dict(model=arm,selected_step=chosen,
                     TRAIN_state_rho_step2000=row['train']['state_effect_rho'],
                     TEST_state_rho_step2000=row['test']['state_effect_rho'],
                     TEST_same_eta_state_change_accuracy_step2000=row['test']['same_eta_state_change_accuracy']))
        real_test_probs[arm]=np.array([expit(get(seed,2000,'test')) for seed in SEEDS])
        if not arm.endswith('shuffled'):
            for seed in SEEDS:
                z=get(seed,2000,'train');prob=expit(z);q=np.divide(s,s+f,out=np.zeros_like(s,dtype=float),where=s+f>0)
                m=state_effect(prob,s,f)
                m.update(interaction_rho=corr(residual(prob,mask),residual(q,mask)),
                         NLL=float((s*np.logaddexp(0,-z)+f*np.logaddexp(0,z)).sum()/(s+f).sum()))
                extra.append(dict(arm=arm,seed=seed,step=2000,**m))
    csvout(OUT/'same_state_extra_seed_metrics.csv',extra)
    csvout(OUT/'real_comparison.csv',table)
    sr=[]
    sr.append(dict(model='Original92-input,32-state noisy fixture',states=32,input_dimensions=92,label='4 Bernoulli trials',
                   known_state_rho=synthetic['test']['2000']['known_state_effect_rho'],
                   observed_state_direction_accuracy=synthetic['test']['2000']['same_eta_state_change_accuracy']))
    for arm,display,lab in [('synthetic_exact_probability','92-input,32-state exact probability','Known conditional probability'),
                            ('synthetic_sufficient3','3 sufficient coordinates,32 states','Same4 Bernoulli trials'),
                            ('synthetic_256states','92-input,256 states','4 Bernoulli trials')]:
        r=read(OUT/arm/'summary.json');m=r['results']['2000']['test']
        sr.append(dict(model=display,states=r['states']['train'],input_dimensions=r['input_dimension'],label=lab,
                       known_state_rho=m['known_state_effect_rho'],observed_state_direction_accuracy=m['same_eta_state_change_accuracy']))
    csvout(OUT/'synthetic_comparison.csv',sr)
    # Whole-state bootstrap; three training seeds never counted as independent data.
    st,ft,ut=counts(source_truth[1,te]);rng=np.random.default_rng(2026100724)
    draws=rng.integers(len(te),size=(2000,len(te)))
    uncertainty={}
    for arm in ('local_state_16heads','minimal23'):
        pp=real_test_probs[arm]
        rr=np.array([np.mean([state_effect(prob[idx],st[idx],ft[idx])['state_effect_rho'] for prob in pp]) for idx in draws])
        uncertainty[arm]=dict(independent_TEST_states=len(te),mean_state_effect_rho=float(np.mean([state_effect(prob,st,ft)['state_effect_rho'] for prob in pp])),
                              family_bootstrap95=np.quantile(rr,[.025,.975]).tolist())
    sufficient=read(OUT/'minimal_physical/protocol.json')
    fixed=read(OUT/'fixed_eta/summary.json')
    fixed_summary={}
    for model0,step0 in [('linear','VAL-selected'),('small_MLP',fixed['MLP_selected_step']),('small_MLP',2000)]:
        fixed_summary[model0+':'+str(step0)]={split:{key:float(np.mean([r[key] for r in fixed['results'] if r['model']==model0 and r['step']==step0 and r['split']==split]))
              for key in ('NLL','state_rho','state_direction_accuracy')} for split in ('train','validation','test')}
    evidence=dict(
        question='Why does real Q not generalize its state dependence when controller inputs are removed?',
        facts=[
            'All controller response context and nominal/bounded flow were removed from every real state-only predictor.',
            'Real state-only networks fit TRAIN nearly perfectly but transfer little same-eta state dependence.',
            'Removing eta encoding with16 separate fixed-candidate output heads does not fix the problem.',
            'Fixing eta8 by a TRAIN-only repeatable-signal rule and fitting a401-parameter scalar Q(s) network still leaves a large train/new-state gap.',
            'Training on a whole-state label permutation can fit almost equally well, demonstrating available memorization capacity.',
            'Same seen states evaluated on additional future seeds retain high conditional agreement, on a historically selected64-cell subset.',
            'An invertible23-coordinate physical input does not remove the real-Q generalization gap.',
            'In a known state-Q function,32 states fail even with exact probability labels;256 states or3 sufficient physical coordinates restore generalization.'],
        leading_explanation='Insufficient independent state coverage relative to the representation and function complexity: models can identify32 training states and fit their Q profiles without discovering a rule that transfers.',
        established_scope='This finite-data/representation mechanism is causal in the synthetic control and consistent with all real-Q diagnostics. It is not yet a uniquely identified cause of the real-Q surface.',
        ruled_out_as_sole_explanations=['Controller-response input interference','An eta encoder that prevents all state learning','Failure to fit the training mapping','Pure finite-seed memorization on the checked subset','Simple duplicated physical input dimensions alone'],
        not_established=['That real-Q extension32to256 will achieve the synthetic result','A unique sufficient low-dimensional feature set for real Q','That more state data alone fixes every controller/scenario'],
        real_fixed_step_results=table,synthetic_fixed_step_results=sr,real_uncertainty=uncertainty,fixed_eta8=fixed_summary,
        extra_seed_limit=dict(states=8,pairs=64,seeds_per_pair=12,selection='Historical Q4-selected anchors/high/low candidates; not representative of all512 pairs'),
        normalization='Source TRAIN only; no state ID, seed namespace or outcome used as input',
        physical_representation_roundtrip_error=sufficient['maximum_reconstruction_error'],
        source_hashes={'dataset':sha(SOURCE/'dataset.npz'),'entities':sha(SOURCE/'entities.npz')},
        original_TEST='Previously opened; retrospective diagnostic and model comparisons, not a new independent confirmation',
        new_physical_rollouts=0,new_real_fit_count=21,new_synthetic_fit_count=12,
        next_real_evidence='A predeclared real-rollout state-breadth comparison with identical eta, future-seed counts, representation, optimizer and independent held-out states; synthetic expansion cannot substitute for it.')
    write(OUT/'scientific_decision.json',evidence)
    report='''去掉 controller 后，state→Q 为什么仍然学不好：诊断结果

最受支持的解释：独立state覆盖与输入/函数复杂度不匹配，模型能记住32个训练state的Q曲线，但没有学到可迁移的state规则。
这一机制在已知函数对照中得到因果验证；真实Q的实验与它一致，但还没有通过新增真实state确认它是唯一根因。

实验范围与排除项
- 固定Ring FF生成标签的底层controller；critic不输入controller ID、参数、76维response context或nominal/bounded flow。
- 物理输入只含位置、速度、目标和固定几何。原物理张量中暗含的4个flow通道也全部清零。
- 原32 TRAIN /8 VAL /16 TEST独立state划分不变；同一16eta库；TRAIN每对最多4seed，VAL/TEST16seed。
- 21个真实标签诊断fit、12个合成对照fit，均为小数据单场景机制测试；没有新物理rollout。另有解析线性/RBF正则化对照。
- TEST已在历史实验打开，只做回顾性诊断；所有checkpoint选择使用VAL，固定step2000也并列报告。

真实Q结果：首先把eta编码也拿掉
用纯物理state输入，直接输出固定16eta对应的16个Q，每个输出头只需学习state变化。
指标为去掉每个eta平均值后的state变化相关性；它不会把“某个eta整体很好”算作学会state。
step2000，三seed均值：
  原92维物理输入：TRAIN相关0.999；新TEST state相关-0.054；同eta的Q变化方向仅48.0%排对。
  将TRAIN state–标签关系随机置换：TRAIN仍约0.999，新TEST相关-0.110。
因此低训练loss并不证明学到物理规律；模型确实有记住每个训练state整条Q曲线的能力。
输入换state时模型的Q并非不变，而是变化方向与真实Q对不上。单一温度校准无法修复这类排序错误。

进一步固定单个eta，只学习标量Q(s)
按TRAIN两半2+2seed的state变化协方差，确定eta8（没有按TEST挑eta）。
不输入eta或controller，用23维物理state训练仅一个16隐藏单元、401参数的小网络。
VAL选择step300：TRAIN相关0.984，TEST0.160；step2000为约1.000和0.130。
独立的带正则线性logistic对照由VAL选择强正则，TEST相关0.016。
因此多eta共用trunk、eta编码和过大的原始网络，都不能单独解释这个泛化问题。

区分state泛化失败与随机seed噪声
同一个纯state16头模型，在见过的state上评估额外12个未参与训练的seed，state变化相关性仍达0.937。
该缓存仅有8state×64输入对，而且是历史Q4筛选后的子集，不能代表所有训练样本。
但是“同state换seed尚可、换state失效”的差异，说明主缺口不能只归结为记住4个seed的偶然结果。

检查物理表示是否遗漏信息或只有冗余维度造成问题
重新构造23维表示：4初始半径、3相对初始角、4目标半径、4目标角偏移、8局部速度。
数值验证能恢复全部92维native物理观测（全局旋转对齐后，最大误差见scientific_decision.json）。
23维模型step2000：TRAIN相关约1.000，TEST0.200，Q变化方向60.3%；按VAL NLL选中的模型TEST相关0.099。
进一步删去8个初速度维度，15维模型TEST相关仅0.123。简单删维度/删速度没有解决真实Q的问题。
原state编码器、纯物理MLP、固定eta16头、23维无损表示都存在同类训练/新state差距。
解析线性/RBF对照也没有得到稳定state信号；最近物理训练邻居的Q RMSE为0.348，随机训练state平均0.331。
这是当前表示/样本的局部一致性证据，不是说所有可能表示都不可学习。

已知函数对照：保持物理state，人工定义明确的state→Q规则
这里是合成概率函数，不是真实仿真成功率。函数由3个已知物理组合与eta决定。
使用完全相同的16头网络、初始化seed、优化器与2000更新；三seed均值，衡量与已知真实概率的state相关性：
  32个state +92维输入 +4次Bernoulli监督：0.251。
  32个state +92维输入 +完全精确的概率标签：0.307。
  32个state +仅3个已知充分物理量 +原4次监督：0.878。
  256个state +92维输入 +仍然4次监督：0.939。
对应后两项Q变化方向准确率约93.7%、97.5%，不是仅仅概率校准变好了。
按各自VAL选定checkpoint，已知概率相关性分别进一步达到0.924和0.964；固定步结果已足以显示机制。
这个受控实验说明：即使state与Q的规则真实、简单且已知，32个点加大量候选输入也可能只学会记忆。
仅消除标签噪声帮助有限；增加独立state或给出真正相关的低维表示，能让相同训练流程学到关系。
不能把合成函数的0.939直接当作真实Q增加数据后的承诺。

修正前面对controller/context和过拟合的解释
去掉controller信息没有修复state学习。原full模型step2000的新state变化相关约0.315，反而比原92维纯state模型更高。
某些删context模型的总NLL或B15变好，主要因为它们退回稳定的eta先验，并不表示学会了state变化。
当前eta先验已覆盖14/16个B15成功state，oracle同样14/16，整体选择成功率会掩盖state学习失败。
过度自信/早期checkpoint选择是另一层机制，解释NLL曲线；但不足以解释纯state模型接近随机的Q变化方向。

可以下的结论
已经证实：controller输入不是这个问题的必要条件；去掉eta编码仍失败；拟合训练state很容易；跨state迁移才是瓶颈。
最受支持的根因解释是有效state监督不足以约束当前高维函数，模型通过样本记忆降低loss。
还没有证实：真实Q只需增加state就一定修复，或者存在某一个已经找到的最优物理特征。
真实Q的最终因果确认需要固定其他条件增加独立state的实验；现有数据库同一精确controller只有56state，已经包含当前全部32/8/16。
不要再把512个state–eta对当作512个独立state，也不要仅靠总体NLL/B15判断是否学会state。

复现脚本：state_only_experiment.py；state_only_controls.py；state_only_report.py。
关键证据：real_comparison.csv、same_state_extra_seed_metrics.csv、synthetic_comparison.csv、scientific_decision.json、verification.json。
'''
    (OUT/'FINAL_REPORT.txt').write_text(report)
    frozen=read(SOURCE/'dataset_frozen.json');all_initial=read(OUT/'training_complete.json')['runs']
    assert all_initial and len(all_initial)==12
    inp=np.load(OUT/'inputs.npz');assert (inp['entity_agents'][...,3:7]==0).all()
    assert not any('context' in k or 'flow' in k for k in inp.files)
    assert len(set(tr)&set(te))==0
    ver=dict(original_dataset_unchanged=sha(SOURCE/'dataset.npz')==frozen['dataset_sha256'],
       original_entities_unchanged=sha(SOURCE/'entities.npz')==frozen['entities_sha256'],
       no_controller_derived_predictor_channels=True,disjoint_physical_state_splits=True,
       base_fits_same_batch_hash_by_seed={str(seed):len({r['batch_draw_sha256'] for r in all_initial if r['seed']==seed})==1 for seed in SEEDS},
       real23_input_recovers_native_physical_observation=sufficient['maximum_reconstruction_error']<2e-6,
       new_physical_rollouts=0,real_data_state_count=len(tr)+len(te)+len(p['splits']['validation']),
       real_models_sha256={str(path.relative_to(OUT)):sha(path) for path in sorted((OUT/'fits').glob('*/*/step2000.msgpack'))},
       evidence_sha256={name:sha(OUT/name) for name in ('real_comparison.csv','same_state_extra_seed_metrics.csv','synthetic_comparison.csv','scientific_decision.json','FINAL_REPORT.txt')},
       report_script_sha256=sha(__file__))
    assert ver['original_dataset_unchanged'] and ver['original_entities_unchanged'] and all(ver['base_fits_same_batch_hash_by_seed'].values())
    write(OUT/'verification.json',ver)
    print(json.dumps(dict(real=table,synthetic=sr,uncertainty=uncertainty),indent=2))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    real=read(OUT/'scientific_decision.json');extra=list(csv.DictReader((OUT/'same_state_extra_seed_metrics.csv').open()))
    rr=next(r for r in real['real_fixed_step_results'] if r['model']=='local_state_16heads')
    same=float(np.mean([float(r['state_effect_rho']) for r in extra if r['arm']=='local_state_16heads']))
    fig,axes=plt.subplots(1,2,figsize=(13,5.3),layout='constrained')
    values=[rr['TRAIN_state_rho_step2000'],same,rr['TEST_state_rho_step2000']]
    labels=['Seen states\nTRAIN seeds','Seen states\n12 extra seeds*','Unseen states\n16 TEST seeds']
    colors=['#427f9f','#70a8a0','#b34a46']
    axes[0].bar(labels,values,color=colors,width=.6)
    axes[0].set(title='Real Q: seen-state fitting does not transfer',ylabel='Correlation of Q variation across states',ylim=(-.25,1.12))
    for i,v in enumerate(values):axes[0].text(i,v+(.03 if v>=0 else -.08),f'{v:.3f}',ha='center',fontsize=12)
    axes[0].text(.02,.015,'* 64 cached cells across8 states; historically selected subset',transform=axes[0].transAxes,fontsize=8,color='#555555')
    rows=real['synthetic_fixed_step_results'];vv=[r['known_state_rho'] for r in rows]
    labels=['32 states\n92 inputs\n4-trial labels','32 states\n92 inputs\nExact labels','32 states\n3 relevant inputs\n4-trial labels','256 states\n92 inputs\n4-trial labels']
    axes[1].bar(labels,vv,color=['#b34a46','#cf9060','#458890','#2f8c62'],width=.65)
    axes[1].set(title='Known-function control: data and representation matter',ylabel='Correlation with known conditional Q',ylim=(-.25,1.12))
    for i,v in enumerate(vv):axes[1].text(i,v+.03,f'{v:.3f}',ha='center',fontsize=12)
    axes[1].text(.02,.015,'Synthetic probabilities only; not new simulation evidence',transform=axes[1].transAxes,fontsize=8,color='#555555')
    for ax in axes:
        ax.axhline(0,color='gray',lw=.8);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='x',labelsize=9)
    fig.suptitle('Controller-free state-to-Q diagnosis | Fixed step2000, mean of3 training seeds',fontsize=13)
    fig.savefig(OUT/'state_only_diagnosis.png',dpi=180);fig.savefig(OUT/'state_only_diagnosis.pdf')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=('report','plot'));globals()[p.parse_args().action]()
