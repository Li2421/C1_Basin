"""Freeze the functional audit and its limitations without performance tuning."""
import json,hashlib,py_compile
from collections import Counter
import numpy as np
from .audit import OUT,PRE,SCENES,folders,load,dump,ev
from .test_functional import run as tests

def avg(rr,key):
    v=[r[key] for r in rr if r.get(key) is not None]
    return float(np.mean(v)) if v else None
def fmt(x):return 'N/A' if x is None else f'{x:.3f}'

def run():
    controls=load(OUT/'shuffle_metrics.json');features=load(OUT/'feature_metrics.json')
    inter=load(OUT/'probe_interaction_metrics.json');fi=load(OUT/'feature_interaction_metrics.json')
    old=load(OUT/'existing_interaction_metrics.json');baselines=load(OUT/'baseline_metrics.json')
    summaries=[]
    for fold in SCENES:
        for name in folders(fold):
            rr=[r for r in controls if r['fold']==fold and r['model']==name]
            origin=next(r for r in rr if r['control']=='original')
            for control in ('state_shuffle','eta_shuffle','joint_shuffle'):
                rs=[r for r in rr if r['control']==control]
                summaries.append({'fold':fold,'model':name,'control':control,'original':origin,
                    **{k:avg(rs,k) for k in ['b15','unresolved','selected_Q_lower','selected_Q_upper','Q_regret_lower','Q_regret_upper','oracle_B15_gap','top1_agreement','score_spearman','score_MAE_change','severe_false_positive']},
                    'B15_min':min(r['b15'] for r in rs) if control=='state_shuffle' else None,
                    'B15_max':max(r['b15'] for r in rs) if control=='state_shuffle' else None})
    dump('shuffle_summary.json',summaries)
    # Cluster bootstrap: all quadruples involving a sampled state move together.
    # This is not a binomial interval over thousands of correlated quadruples.
    boot=[]
    for fold in ('ring','four'):
        for name in folders(fold):
            z=np.load(OUT/f'probe_{fold}_{name}_contrasts.npz')['contrasts'];n=8
            rng=np.random.default_rng(731993);strong=abs(z[:,4])>=.5;acc=[]
            for _ in range(2000):
                counts=np.bincount(rng.integers(0,n,n),minlength=n)
                w=counts[z[:,0].astype(int)]*counts[z[:,1].astype(int)]*strong
                pred=z[:,5].copy();pred[abs(pred)<1e-6]=0
                if w.sum():acc.append(float(np.sum(w*(np.sign(z[:,4])==np.sign(pred)))/w.sum()))
            boot.append({'fold':fold,'model':name,'strong_sign_state_cluster_95_interval':np.quantile(acc,[.025,.975]).tolist() if acc else None,
                'state_clusters':8,'resamples':2000,'not_accounting_for_rollout_seed_estimation_error':True})
    dump('state_cluster_uncertainty.json',boot)

    count=Counter()
    assert len(list((OUT/'probe').glob('run_*of16.json')))==16
    for p in (OUT/'probe').glob('run_*of16.json'):count.update(load(p))
    evidence=load(OUT/'probe/evidence.json');num=sum(len(r['numeric']) for r in evidence);collision=sum(r['collisions'] for r in evidence)
    from shared_rollout_db.src.rollout_db import connect
    with connect(True) as con:
        exp=con.execute('SELECT experiment_uid FROM experiment WHERE path=?',(str(OUT/'probe/execution'),)).fetchone()[0]
        persisted=con.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?',(exp,)).fetchone()[0]
        ids=[u for r in evidence for u in r['rollout_uids']]
        for start in range(0,len(ids),400):
            batch=ids[start:start+400];assert con.execute('SELECT count(*) FROM rollout WHERE rollout_uid IN ('+','.join('?'*len(batch))+')',batch).fetchone()[0]==len(batch)
    dump('db_cache_report.json',{'preflight':load(OUT/'probe/cache_preflight.json'),'execution':dict(count),
        'persisted_new_unique_seed_rows':persisted,'valid_evidence_uid_checks':len(ids),'numerical_slots':num,
        'numerical_eta_tuples':sum(bool(r['numeric']) for r in evidence),'collision':collision,
        'models_retrained':False,'new_training_labels':0,'new_evidence_diagnostic_only':True})
    models={}
    for fold in SCENES:
        for name,folder in folders(fold).items():
            fm=load(folder/'models_frozen.json')
            for kind in ('shared','eta_only'):
                s=fm[kind]['selected'];assert ev.sha(s['checkpoint'])==s['sha256'];models[str(s['checkpoint'])]=s['sha256']
    for p in OUT.glob('*.py'):py_compile.compile(str(p),doraise=True)
    dump('regression_tests.json',tests())

    decisions={'overall':'UNDERRESOLVED','STATE_USAGE':'YES: joint predictions and rankings change with state, but not uniformly usefully',
       'STATE_ETA_INTERACTION':'Ring has weak conditional-gap signal; reliable preference reversal learning is not established. Four evidence is not supportive. Toy is a positive control.',
       'INTERACTION_GENERALIZATION':'Within-scene Toy empirical evidence positive; primary-scene unseen-state/unseen-eta evidence weak; LOSO transfer not demonstrated',
       'FEATURE_MECHANISM':'Ring agent relations/goals/geometry/Flow affect conditional-gap prediction; feature swaps are not certified physical causal interventions. Constant time/count fields unidentifiable here.',
       'per_model':{'Ring_scene_specific':'ETA_ONLY_EFFECTIVELY','Four_scene_specific':'ETA_ONLY_EFFECTIVELY',
         'Ring_joint_enriched':'UNDERRESOLVED','Four_joint_enriched':'UNDERRESOLVED',
         'Toy_joint_enriched':'STATE_ETA_INTERACTION_GENERALIZES_WITHIN_SCENE',
         'Ring_LOSO':'ETA_ONLY_EFFECTIVELY','Toy_LOSO':'ETA_ONLY_EFFECTIVELY','Four_LOSO':'UNDERRESOLVED','Double_joint_enriched':'UNDERRESOLVED'},
       'classification_scope':'Frozen cohorts and finite empirical labels; effective eta-only does not claim exact architecture independence',
       'no_model_or_generator_changes':True,'no_modes':True}
    dump('decision.json',decisions)
    lines=['# Critic 是否真正学到 state–eta interaction？','',
      '**结论：Ring / Four-Way 目前不能宣称已可靠学到“随 state 正确改变 eta 相对排序”。但也不能把联合 critic 简单等同于完全不看 state 的网络。**','',
      '本轮检验的是冻结模型的函数行为，不使用 raw weight magnitude 作为证据。第一阶段不训练；第二阶段只拟合隔离的诊断 eta-only / additive 基线，未重训练任何现有 critic、generator 或控制器。新增 rollout 仅供诊断，不进入训练。','',
      '## 1. 资产、分层与定义','',
      '读取当前本地 LOSO root-cause 完成资产，而不是只沿用较早 prose。A=single_scene supervised；B=joint_enriched supervised（并保留 joint_generic 控制）；C=source-only LOSO。每个模型选用此前 source/validation 冻结的 checkpoint，不按本轮结果选模型。四个冻结候选集是 Ring60、Four24、Double24、Toy48，均为已归档的16个随机候选，排除 mean；无重新抽样。模型/数据位置见 ALREADY_TESTED.md、provenance.json。','',
      '定义 Δ=[Q(a,i)−Q(a,j)]−[Q(b,i)−Q(b,j)]。A(h)+B(eta) 在概率空间的 Δ 恒为0；但 sigmoid(A(h)+B(eta)) 的概率 Δ 可以非0，且仍不能翻转 eta 排序。因此同时报告概率双差分、logit 非可加性，以及真正反向排序。|Δ|>0 本身不等价于发生 ranking reversal。','',
      '预注册：20个场景内独立 derangement seeds；强交互 |Δ_true|≥0.5；清晰符号比较 |Δ_true|≥0.125；反转要求两边 Q gap 均≥0.25 且方向相反。真 Q 使用完整证据，数值失败不填0。所有 quadruples 共享 state/eta，不能当独立数千次统计试验。','',
      '## 2. State shuffle：保留原 state 的候选与真 Q','',
      '| 场景 | 模型 | 原 B15 / 未认证 | shuffle B15均值 [min,max] | top1一致率 | score相关 | 原→shuffle Q下界 | severe FP均值 |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        if r['control']!='state_shuffle':continue
        o=r['original'];lines.append(f"| {r['fold']} | {r['model']} | {o['b15']} / {o['unresolved']} | {r['b15']:.2f} [{r['B15_min']},{r['B15_max']}] | {r['top1_agreement']:.3f} | {fmt(r['score_spearman'])} | {o['selected_Q_lower']:.3f}→{r['selected_Q_lower']:.3f} | {r['severe_false_positive']:.2f} |")
    lines+=['','Ring enriched 58/60→57.15/60，说明原 K16 部署中的大部分好选择不需要正确匹配具体 state；但其 top1 约80%一致，仍存在 state 影响。Toy44/48→35.25/48 是检验有效的正控制。Four 的数值未认证不能当失败，不能仅比较已认证 B15 数量就断言某选择更差。','',
      '## 3. Eta shuffle / joint shuffle','',
      '| 场景 | enriched 控制 | score MAE变化 | score相关 | 相同候选位置 top1 |','|---|---|---:|---:|---:|']
    for r in summaries:
        if r['model']=='joint_supervised':lines.append(f"| {r['fold']} | {r['control']} | {r['score_MAE_change']:.4f} | {fmt(r['score_spearman'])} | {r['top1_agreement']:.3f} |")
    lines+=['','Eta/joint shuffle 的 top1 是候选位置比较，不是相同 eta 身份。recipient state 上的 donor eta 没有对应真 Q 时，B15/Q/regret 均为 N/A；绝不挪用 donor state 的标签。详见逐 seed 的 shuffle_metrics.json。','',
      '## 4. 共享 eta 直接 interaction 测试','',
      '已有 VAL 缓存中 Ring 仅1个、Four仅30个完整四元组，且没有强交互；不能把 state-specific K16 中相近但不同的 eta 当同一 eta。于是按固定 hash 从各场景既有独立确认队列取8个状态，使用共同8个预注册 Sobol eta及 eta0 参考；模型预测在新结果前冻结。该队列已被查看过，属于诊断，不是新的 untouched generalization 声明。','',
      '| 场景 | 模型 | 完整四元组 | Δ相关 | 清晰符号准确率 | 强交互数 / 方向准确率 | 清晰反转数 |',
      '|---|---|---:|---:|---:|---:|---:|']
    for r in inter:
        if r['variant']=='full':lines.append(f"| {r['fold']} | {r['model']} | {r['quadruples']} | {fmt(r['delta_spearman'])} | {fmt(r['sign_accuracy_clear'])} | {r['strong_count']} / {fmt(r['strong_sign_accuracy'])} | {r['reversal_count']} |")
    lines+=['','上表包含eta=0诊断参考；它不是部署候选，也不在非零eta冻结域内。必须另看仅8个域内eta：Ring enriched Δ相关0.078、强子集117个、符号准确69.2%；Four enriched Δ相关0.163、符号准确49.0%，仍无强交互。详见 full_domain_only 指标，不能把域外参考的误差完全归咎于部署分布。','',
      'Ring 真 Δ 范围约[-1.0625,1.0625]，enriched 预测仅约[-0.153,0.231]：明显低估状态相关的差值幅度。含参考时整体相关0.089，强子集相关0.262、符号准确68.3%，属于弱信号，不能包装成可靠 ranking learning。该68.3%的state-cluster区间约47.5%–85.1%，包含50%，且尚未计入rollout seed噪声。Four没有预注册强交互样本，含参考时enriched 双差分相关约−0.042。','',
      '两主场景面板均无达到预注册幅度的真 ranking reversal。反转 recall/direction accuracy 是 N/A；“全部预测无反转”所得接近100% detection accuracy 不构成正证据。完整 Δ 分布及强子集指标在 probe_interaction_metrics.json；按8个 state 重采样的不确定区间在 state_cluster_uncertainty.json，尚未计入 seed 估计误差。没有为了找出漂亮反转而自适应扩大全网格。','',
      'Toy 复用验证证据含252个经验反转，enriched正确反转方向约91.3%，Δ相关0.754；LOSO Toy正确反转方向为0。这说明同类共享编码/critic在有相关监督时能够表达交互，但该能力没有被证明可迁移。Toy标签含不同seed预算，其结果是有限经验Q对照，并非每个格子均Q16认证。Double已有8072个四元组、仅2个清晰反转，enriched两者均未识别。','',
      '## 5. Eta-only、additive 与 full 基线','',
      '相同各模型 TRAIN/验证家族划分和Q标签：保留已有eta-only MLP；另外用固定128维eta Fourier特征、固定ridge .01拟合 eta-only 和 A(h)+B(eta)，其中A使用同一冻结critic的h。只用TRAIN拟合，不选超参数。后者不是重新训练的同容量神经网络，不能用其欠拟合单独证明full学会interaction；优势证据必须结合直接双差分/反转。additive预测不截断后选argmax，避免制造clipping ties；仅概率误差计算clip。','',
      '| 场景 | enriched对照 | B15 | 未认证 | 完整Q MSE |','|---|---|---:|---:|---:|']
    for r in baselines:
        if r['model']=='joint_supervised':lines.append(f"| {r['fold']} | {r['baseline']} | {r['b15']} | {r['unresolved']} | {r['Q_MSE']:.4f} |")
    lines+=['','Ring full的校准/回归明显优于eta-only，但只有3个额外B15（58 vs55），且state shuffle影响很小；回归优势不等价于学会反转。Four full19+5未认证与eta-only23+1未认证不能解释成四个确定的ranking错误。所有A/B/C逐模型表见 baseline_metrics.json；原始eta是否见于TRAIN及state overlap见 unseen_state_eta_audit.json。','',
      '## 6. Feature mechanism：哪些输入参与排序','',
      '| enriched场景 | feature group | permutation top1一致 | selected B15均值 |','|---|---|---:|---:|']
    for fold in ('ring','four'):
        for group in sorted({r['group'] for r in features}):
            rr=[r for r in features if r['fold']==fold and r['model']=='joint_supervised' and r['group']==group and r['mode']=='permutation']
            lines.append(f"| {fold} | {group} | {avg(rr,'top1_agreement'):.3f} | {avg(rr,'b15'):.2f} |")
    lines+=['','Ring共享探针中，强交互方向准确率原68.3%；打乱agent relations降至约49.0%，goal-relative约52.7%，geometry约55.0%，Flow约60.1%。因此不能说联合模型完全不使用物理state：这些字段参与其弱交互信号。但这不是已证实的正确反转机制。','',
      '同场景TRAIN均值替换与5个permutation seeds均已完成，详细score/top1/Q/interaction结果见 feature_metrics.json 和 feature_interaction_metrics.json。LOSO不使用目标TRAIN均值，无法合法构造同条件source reference时记N/A。各feature组可能重叠，混合字段可能离开物理流形；因此这是函数依赖诊断，不是物理因果干预。time/count在该t0固定N/M队列中恒定，零影响意味着不可辨识，不能解释为不重要。没有用梯度或权重大小替代上述证据。','',
      '## 7. Generator 与 critic 严格分开','',
      'state-conditioned generator已先过滤候选，所以K16 oracle高、critic shuffle影响小可以同时成立。系统整体依赖state并不能证明critic依赖state。共享eta面板去掉了候选生成这一混淆；LOSO仅指critic目标标签未参与拟合，不表示此前目标监督的generator也是zero-shot。','',
      '## 8. 最终四项判断','',
      '- STATE_USAGE：联合模型确实使用h；Ring scene-specific基本eta排序，Four scene-specific在冻结K16上top1完全不随shuffle改变。\n- STATE_ETA_INTERACTION：Ring joint有弱的差值幅度信号，但可靠偏好反转未被建立；Four暂无支持性直接证据；Toy有经验正证据。\n- INTERACTION_GENERALIZATION：主场景未见state/eta上的支持薄弱，LOSO不支持交互迁移；不能因架构可表达而声称迁移成功。\n- FEATURE_MECHANISM：关系、goal、geometry、Flow参与Ring弱信号；常量time/count不可判断；依赖输入不等于用对输入。','',
      '整体分类 **UNDERRESOLVED**。这不是“没做完”，而是主场景中还缺少独立、充分的真排序反转证据，且目前直接差值预测质量不足。scene-specific Ring/Four 在本轮范围可称 ETA_ONLY_EFFECTIVELY；joint Ring/Four不能硬贴精确additive标签，因为其函数存在非可加性，但非可加性未等价为可靠科学能力。Toy joint的有限经验结果支持 STATE_ETA_INTERACTION_GENERALIZES_WITHIN_SCENE。没有任何层次支持 STATE_ETA_INTERACTION_TRANSFERS_ACROSS_SCENES。','',
      '## 9. 数据库、数值与边界','',
      f'请求2304 continuation；执行统计 {dict(count)}；新增数据库唯一seed记录 {persisted}；未认证seed {num}，涉及{sum(bool(r["numeric"]) for r in evidence)}个eta tuple；有效碰撞{collision}。full-Q计算排除未完整认证格子，保留上下界及全部numerical记录；这种缺失可能相关于eta，不隐藏其选择偏差。未使用提前停止k/n冒充Q16。','',
      '所有新增证据由标准cache preflight、journal和DatabaseSink持久化；没有生成训练标签或改变模型/归一化/OrthoFlow3/安全/成功条件。现有checkpoint哈希校验通过，函数分解与完整网络一致，合成additive/interaction回归通过。','',
      '最小下一步（本轮不自动执行）：在独立train/dev状态上，以共享eta边界面板取得足够、经独立seed块复核的排序反转，再判断是监督缺少反转还是critic拟合不了；不是先换architecture，也不是把更多高oracle K16当interaction证据。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    sources={}
    for fold in SCENES:
        for folder in folders(fold).values():
            for n in ('models_frozen.json','normalization.json','source_pairs.parquet','source_states.json'):
                sources[str(folder/n)]=ev.sha(folder/n)
    dump('provenance.json',{'frozen_models_verified':models,'sources':sources,
        'audit_scripts':{p.name:ev.sha(p) for p in OUT.glob('*.py')},
        'probe_protocol_sha256':ev.sha(OUT/'probe/protocol.json'),'probe_predictions_sha256':ev.sha(OUT/'probe/predictions_frozen.json')})
    return decisions

if __name__=='__main__':print(json.dumps(run(),indent=2))
