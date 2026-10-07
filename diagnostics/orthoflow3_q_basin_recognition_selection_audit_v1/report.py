"""Generate a compact human report, tables, and a self-contained HTML artifact."""
import base64, csv, hashlib, html, json
from pathlib import Path

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def csvread(p):return list(csv.DictReader(Path(p).open()))
def triple(rows,key):return ', '.join(str(r[key])for r in rows)
def interval(rows,key,digits=3):
    values=[r[key]for r in rows if r[key]is not None]
    if not values:return 'N/A'
    lo,hi=min(values),max(values)
    return f'{lo:.{digits}f}'if lo==hi else f'{lo:.{digits}f}–{hi:.{digits}f}'
def main():
    summary=read(HERE/'summary.json');data=summary['models'];metas=read(HERE/'cohorts.json');M={r['name']:r for r in metas}
    out=[];parts=[]
    def title(t):out.extend(['',t,'='*len(t),'']);parts.append('<h2>'+html.escape(t)+'</h2>')
    def para(t):out.extend([t,'']);parts.append('<p>'+html.escape(t).replace('\n','<br>')+'</p>')
    def table(headers,rows):
        out.extend([' | '.join(headers)]+[' | '.join(map(str,r))for r in rows]+[''])
        parts.append('<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in r)+'</tr>'for r in rows)+'</tbody></table></div>')
    def select(cohort,kind,cond='correct'):
        return sorted([r for r in data if r['cohort']==cohort and r['model'].split('__')[0]==kind and r['model'].split('__')[-1]==cond],key=lambda r:r['model'])
    choices=[('Toy TT','matched_toy_give_way_TT','full_context','eta_only'),('Toy FF','matched_toy_give_way_FF','full_context','eta_only'),
             ('Ring TT','matched_ring_exchange_TT','full_context','eta_only'),('Ring FF','matched_ring_exchange_FF','full_context','eta_only'),
             ('Ring v11 t0','ring_v11_t0','physical_context','eta_only'),('Ring K16 88138','ring_k16_88138','controller_cv_full_context','controller_cv_eta_only'),
             ('Ring K16 88139','ring_k16_88139','controller_cv_full_context','controller_cv_eta_only'),('Toy independent48','toy_generator_rep48','critic','eta_only_kernel'),
             ('Ring K2 88136','ring_k2_88136','raw_bypass','eta_only'),('Ring K2 88137','ring_k2_88137','raw_bypass','eta_only'),
             ('DB transfer24 [diagnostic]','db_transfer24','full_context','eta_only')]
    title('Learned Q 的 basin recognition → intervention selection 审计')
    para('结论：STATE_CONDITIONED_BASIN_SELECTION_SUPPORTED，限定于 Toy 的已测范围及部分 Ring 小候选任务。现有证据不支持把完整的“state- and field-dependent basin learning and reliable selection”写成普遍成立的核心结论。Field conditioning 已有识别信号，但 held-controller 上的排序反转、top-1 选择与训练 seed 稳定性仍不足。')
    para('本轮新增 rollout = 0；新增训练 / 调参 / ensemble = 0。重算 16 个 cohort、556 个不同 physical state、873 个模型/输入条件；并行模型 seed 和同一 state 的 TT/FF 不是新增独立样本。直接只读复核 29,696 条 seed 记录；TT/FF 另复用已完成逐 seed 校验的 16,384 条主样本。全部原始模型、结果和已有用户文件保留。')

    title('1. Experimental contract')
    table(['Panel','states / K / seeds','split 与用途','eta / controller'],[
        ['TT/FF Toy + Ring','16 + 16 physical states; each TT/FF; K16; 16 seeds','每场景 TRAIN32 / VAL8 / TEST16；fresh true-t0 TEST 主证据','同一 scrambled Sobol bank，seed202610042；16 eta 曾用于 TRAIN；每 formulation 单独训练'],
        ['Ring v11 true-t0','24 / 16 / 16','TRAIN12 / source VAL4；v11 和新 TEST families 都未参与 critic 训练','同一 source TRAIN eta bank：4 个高频 + 12 个几何分散 eta；predictions 在 outcomes 前冻结'],
        ['Ring v11 midtrajectory','12 / 16 / 16','新 family supporting；不冒充 true-t0','同一 v11 与上述同一 eta bank'],
        ['Ring held88138,88139','16 + 16 / 16 / 16','controller-held-out source CV 选择主模型；两套新 Flow 与新 true-t0 TEST','固定 bank：8 seen eta + 8 source 凸包内 Sobol unseen eta；含 TRAIN-common eta'],
        ['Toy generator replication','48 / 16 / 16','独立 true-t0 TEST；固定三 critic ensemble 已在旧实验冻结','各 state 的 proposal bank 不同；每个 selector 共用该 state 的同一 bank；不可做跨 state exact-eta reversal'],
        ['Ring K2 confirmations88132..88137','每 controller64 / 2 / 16','所有六次原有独立确认都保留；模型版本不同，分别报告','两个固定 seen eta；后两次 raw-bypass 在 outcome 前冻结；不是 K16 或 unseen-eta 证据'],
        ['DB transfer','24 / 16 / 16','source 不含 DB；旧 target 已打开，属 cached diagnostic','原有同一 proposal bank 比较 eta-only、h+eta/no-context、full；不新增 TEST 资格'],
        ['Four-scene controller probe + newer full','每场景 source VAL4；新版另含 Ring VAL16 个有 field changes 的 state','用于 checkpoint 选择的 VAL：只能支持机制诊断','实际同一 x、eta 的多 controller counts；不补齐 early-stop 未跑 seeds']])
    para('B15 冻结为 16 个预定 continuation seeds 中至少 15 个成功。>=15 observed successes 可确认 positive；>=2 observed failures 可确认 negative；其他为 unresolved。Q16 保留 [s/16, (16−f)/16]，numerical failure 不当作普通失败。旧 B63/Q64 几何、ball/margin 学习与 B15 完全分开；场景各自的 native success/window-progress、horizon、safety 和 continuation contract 在同 cohort 中固定，不跨场景合并标签。')
    para('实际 TT = P_U(P_U(native Flow)+g_eta)。FF 在每个 Flow pseudo-step 把 correction 放进 vector field，并投影整个 field；两者共享 physical state、eta 和后续随机流。这里的 full_context 分别在 TT、FF 上训练，不能把两套模型的预测差说成同一个模型学到了 field dependence。另一些 held-controller 实验在 true-t0 作选择，但保留相同预先承诺的初始 native action；c 指之后的实际 future Flow condition。')
    para('Critic 主要按 observed success/failure 的 Bernoulli likelihood 训练，sigmoid(Q logit) 估计单次 continuation 成功概率；它不是已经校准的 Pr[B15=1]。因此本报告把 candidate AUROC/AP 和 within-state ranking 用于识别 B15，把 trial Brier/NLL 仅作辅助。所有训练 seed17、23、41 都报告，不按 TEST 选赢家；Toy replication 的既有 ensemble 单列为 frozen。')
    para('新版 source-controller CV 使用 Toy / Four-Way / Ring 的 11,917 个 source pairs、412 个 source states，TRAIN/VAL family overlap=0，按持出 source controller 的 VAL 选择 checkpoint。两候选 raw-bypass 后期模型使用 160 个 native TRAIN families、12 个 native controllers 和既有 crossed programs。逐模型绝对 checkpoint 路径及 SHA256 见 source_hashes.json；原冻结清单可从 cohorts.json 的 source 路径和 evidence_inventory.json 定位。')

    title('2. Basin recognition：候选识别与 state 内排序分别报告')
    recognition=[]
    for label,name,kind,base in choices:
        a=select(name,kind);b=select(name,base,'none'if name=='ring_v11_t0'else'correct')
        recognition.append([label,a[0]['mixed_states'],interval(a,'candidate_AUROC'),interval(a,'candidate_AP'),interval(a,'state_AUC_mean'),interval(b,'state_AUC_mean')])
    table(['Panel','mixed states','full pooled AUROC','full AP','full mean within-state AUROC','eta-only within-state AUROC'],recognition)
    distribution=[]
    for label,name,kind,base in choices:
        a=select(name,kind)
        distribution.append([label,interval(a,'state_AUC_q25'),interval(a,'state_AUC_median'),interval(a,'state_AUC_q75'),triple(a,'state_AUC_above_half')+f' /{a[0]["mixed_states"]}'])
    table(['Per-state distribution (range over model seeds)','25th percentile','median','75th percentile','states with AUROC>0.5'],distribution)
    para('主指标是 mixed state 内正/负 eta 的两两排序准确率（ties=0.5，等于该 state AUROC），再对 state 等权平均。全 positive / 全 negative state 的 within-state AUROC 为 undefined，不置为 0.5；但全部 oracle-eligible state（含全 positive）都进入选择分母。per_state.csv.gz 保留每个 state 的 AUROC、robust rank/percentile、top4 enrichment 和 top2/top4 是否含 robust eta；recognition_selection.csv 给出分位数、95% state-bootstrap CI、AP、trial Brier/NLL。')
    para('Toy TT：full 对 eta-only 的 paired within-state AUROC 增量分别为 0.250、0.266、0.258，95% state-bootstrap CI 分别 [0.187,0.316]、[0.215,0.316]、[0.172,0.343]。独立 Toy48：full 0.925 vs eta-only kernel 0.653；增量 0.272，95% CI [0.194,0.349]。这提供了 state 内 eta 排序证据。Ring TT/FF 虽 AUROC 较高，却经常不优于 eta-only，而且选不到少数优先级反转的候选。')
    title('3. State-conditioned response / ranking reversal')
    revtable=[]
    for label,name,kind,base in choices:
        rev=[r for r in summary['state_reversals']if r['cohort']==name and r['model'].split('__')[0]==kind and r['model'].endswith('correct')and r['reversal']=='B15_swap']
        if not rev:continue
        rev=sorted(rev,key=lambda r:r['model']);revtable.append([label,rev[0]['cases'],rev[0]['unique_states'],triple(rev,'both_correct'),interval(rev,'same_eta_direction_accuracy_half_ties')])
    table(['Panel','true B15 reversal quadruples','involved states','both sides correct, seed17/23/41','same-eta state response accuracy'],revtable)
    para('严格 reversal 要求相同两个 eta 在两个 state 上发生 B15-vs-nonB15 顺序交换，Q 必须两边同时排对。Toy TT 为 1592–1645 / 1712（93.0%–96.1%），eta-only 为 0。Ring v11 为 0/42；较新 K16 在88138上为0/163，在88139上为21、2、5/398。因此高平均 AUROC 不等于掌握关键条件反转。另行提供 gap>=0.25 的保守 Q16 interval reversal，不能与 B15 reversal 分母混用。')
    para('Quadruples 大量共享 state 和 eta，不作为独立 Bernoulli trials 做显著性检验。逐 state 的分布和 CI 使用 state family 作重采样单位；不把三个训练 seed 合成三倍 N。Toy TT 联合 state+context shuffle 使 top1 从15/16、16/16、14/16降至9/16、10/16、8/16，但逐 seed McNemar p=0.03125、三 seed Holm p=0.09375：机制与效应量清楚，二元选择显著性仍受 N16 限制。')

    title('4. Field dependence：已有信号，尚无稳定完整链条')
    table(['paired actual-controller evidence','field membership response','field ranking reversal','资格'],[
        ['旧 H20 / Ring','11/14 correct direction，每个 seed','0/14 interval-certified reversals，每个 seed','同一x,eta实际三 controller；source VAL'],
        ['旧 H20 / Four-Way','10,16,18 /20','0,1,0 /7 interval reversals','source VAL；ID+eta diagnostic 6/7，不是 zero-shot controller 输入'],
        ['新版 family-CV full / Ring','1362,1265,1371 /1724 =73.4%–79.5%','60,55,73 /121 B15 swaps；196,168,229 /474 interval reversals','16 个 source VAL state；seen controller / checkpoint selection；不是主controller-CV checkpoint'],
        ['held v11, correct vs wrong base C','state AUROC 0.730,0.804,0.702 vs 0.734,0.800,0.700','top1两种context均为0,20,0 /23；每个seed跨所有state只选同一个eta','独立 true-t0 held controller TEST'],
        ['held88138, correct vs wrong C','paired state-AUROC +0.031–0.065；各 seed CI>0','top1:15,15,16 vs15,15,15 /16','独立 K16 TEST；选择提升未显著'],
        ['held88139, correct vs wrong C','paired state-AUROC +0.041–0.079；各 seed CI>0','top1:6,6,11 vs6,6,6 /16；seed41 rescue6/break1,p=.125','独立 K16 TEST；仅一个训练seed有选择收益']])
    para('因此不能声称 Q 完全忽略了 field：新版 correct C 确实改善候选排序，source VAL 也有真实 field-order reversal 的部分识别。但同样不能把 calibration / AUROC 改善升级为可靠 field-conditioned selection。88138 与88139的 TEST state集合不同，不可互相配成同一x的field counterfactual；wrong-context 是输入干预，只有 source 多controller counts 才是实际 field outcome 的配对。当前缺少新版模型在独立TEST上的同x、同eta、不同field reversal 确认。')
    para('上表不同模型代际不合并：旧probe为H20 physical_context；新版实际field reversal使用db_transfer_v1的family-CV full checkpoint；88138/88139主模型为source_controller_cv_v2的controller-CV full checkpoint。源Ring实际field配对已逐项核对相同RNG、plant、horizon、dt、safety、success criterion和committed t0 Flow；改变的是future controller。不同checkpoint之间的正面结果不能互相替代。')
    para('h-only shuffle 未必等于 state 信息移除：C(x,eta,c) 本身含 state-conditioned physical responses。no_context 是已有 h+eta 模型；eta-only 才去掉全部 state/context。context_shuffle / joint shuffle 与 wrong-controller context 分开报告。安全投影、原始 Flow 查询、短期物理 response 都是部署时允许的量；full-rollout truth、oracle 与成功标签仅进入评估。')

    title('5. Intervention selection 与 oracle gap')
    main_table=[]
    for label,name,kind,base in choices:
        a=select(name,kind);b=select(name,base,'none'if name=='ring_v11_t0'else'correct');rnd=next(r for r in summary['random']if r['cohort']==name)
        fixed=select(name,'global_train_eta')or select(name,'fixed_source_common_eta')or select(name,'eta_mle')
        den=a[0]['oracle_eligible'];main_table.append([label,f'{den}/{a[0]["N"]}',triple(a,'selected_B15')+f' /{den}',triple(b,'selected_B15')+f' /{den}',str(fixed[0]['selected_B15'])+f' /{den}'if fixed else'N/A',f'{rnd["random_eligible_rate"]:.1%}',triple(a,'ranking_failures')])
    table(['Panel','Oracle@K / all states','full selected / eligible','eta-only selected / eligible','TRAIN-fixed / eligible','random eligible expectation','ranking gap, states'],main_table)
    para('所有 triple 顺序固定为 seed17、23、41；单个数字是原 frozen ensemble。表中分母 eligible 与 Oracle@K 的全state分母明确分开。Random 为同一候选池均匀抽取的解析期望，不伪造一组随机 rollout。K16 controller88138/88139有各1个未决 candidate，random 的上界分别为67.19%和37.89%；表中为确认 B15 下界，主 controller-CV top1均已决。')
    table(['selection paired evidence','rescue / break','delta and 95% state-bootstrap CI','exact McNemar'],[
        ['Toy TT full vs eta-only','2/0; 3/0; 2/1','+12.5pp [0,31.25]; +18.75pp [0,37.5]; +6.25pp [-12.5,25]','p=.5,.25,1；不声称二元收益显著'],
        ['Toy48 full vs strong eta-only kernel','4 /1','+6.25pp [-2.08,16.67]','p=.375'],
        ['Toy48 full vs eta-only MLP','15 /0','+31.25pp [18.75,43.75]','p=.000061'],
        ['Ring88136 raw-bypass vs eta-only','24 /4，每个seed','+31.25pp [17.19,45.31]','p=.000180；本报告该family Holm p=.000783'],
        ['Ring88137 raw-bypass vs eta-only','8/4;8/6;8/5','+6.25,+3.13,+4.69pp；三个CI都含0','p=.388,.791,.581'],
        ['Ring88139 CV full vs CV eta-only','0/0;0/0;6/1','0,0,+31.25pp；seed41 CI[0,56.25]','p=1,1,.125；Holm=1,1,.375']])
    para('Toy48 的单独 policy reference：Safety31/48、fixed/common42/48、generator mean38/48；同池 full46/48、kernel43/48、MLP31/48、oracle48/48。Safety / fixed-common / mean 是另外已跑的固定 policy，不偷偷加入 critic 的16候选池；同池 selector 收益只由后三者比较判断。匹配 TT/FF 的16点 Sobol bank不包含eta=0；没有同一cohort完整Safety标签时记N/A，不补rollout。')
    top=[]
    for label,name,kind,base in choices:
        a=select(name,kind);top.append([label,triple(a,'top2_eligible'),triple(a,'top4_eligible'),interval(a,'selected_Q_lower'),interval(a,'selected_Q_upper'),interval(a,'regret_lower'),interval(a,'regret_upper')])
    table(['Panel','top2 contains robust (count)','top4 contains robust (count)','selected Q16 lower','selected Q16 upper','regret lower','regret upper'],top)
    para('统计检验均 conditional on 当前 controller、candidate bank 和固定 Q16 labels；95% CI 为10,000次 family bootstrap，seed20261006。McNemar 为 exact two-sided，另在 cohort×baseline类型内做 Holm；未决比较保留全state delta bounds，complete-case结果不冒充全样本结果。没有将任意一个事后挑出的 p 值作为整条 claim 的判据。')

    title('6. 所有重要反例与失败分解')
    confirm=[]
    for ctl in range(88132,88138):
        name=f'ring_k2_{ctl}';kind='trunk_only'if ctl<88136 else'raw_bypass';a=select(name,kind);b=select(name,'eta_only')
        confirm.append([ctl,kind,a[0]['oracle_eligible'],triple(a,'selected_B15'),triple(b,'selected_B15'),triple(a,'ranking_failures'),triple(a,'ranking_unknown'),a[0]['proposal_failures'],a[0]['oracle_unknown']])
    table(['Ring K2 controller','original focal model','oracle /64','full robust','eta-only robust','ranking error','ranking unresolved','no robust candidate','oracle unresolved'],confirm)
    para('不能只报告88132或88136。88135仍有11个eta-only-to-oracle headroom，trunk-only仅44–45/56，未优于eta-only45/56；并非饱和造成无收益。88136 raw-bypass55/59显著胜过source eta-only35/59，但 hindsight另一常数eta已经54/59，且真实B15 state reversal仅24/120判对，说明大部分收益仍来自controller整体eta偏好改变。88137更直接的state reversal56–72/117，但top1相对强先验的收益小且不显著。')
    para('Proposal failure 与 critic ranking failure 是互斥的当前bank失败原因：Toy TT 为0与0–2；Ring FF为2与1–3；v11 true-t0为1与23/3/23；88139 K16为0与10/10/5；Toy48为0与2。Field generalization failure不是第三个可相加的计数类别：它可能造成 ranking error，需结合context干预/实际field配对识别。没有用一个模型的失败率硬推 representation 的因果责任。')
    para('原Toy200 generator K-sweep全部保留：K=1,2,4,8,16的oracle分别131,166,173,184,194；learned分别131,161,167,177,181。K16的19个失败=6个无robust candidate+13个ranking error；fixed-common180/200，full181/200，rescue15/break14。K变大后oracle gap从0增至13，符合max-selection误差放大；不能把proposal覆盖提升算作Q学会basin。K200为已有打开数据，属supporting。')
    para('DB新式transfer的eta-only24/24而full13–17/24，h+eta/no_context仅13–15/24；additive nominal context24/24因eta-independent加性项不能改变同state内eta排序，不构成state learning证据。Four-Way source+spatial-eta holdout为oracle16/16，full与eta-only均16/16，属于selection饱和；source snapshots多为midtrajectory，不能冒充true-t0 field recognition。旧独立source+eta Ring holdout eta-only与full都达oracle15/15，旧true-t0 Ring60 frozen panel强eta-only58/60高于corrected full54/60；后者已打开，不作新确认。')
    para('更早continuous basin critic的simultaneous unseen-state/unseen-eta Toy rank Spearman≈0.015，DB被eta-only压过；finite-candidate排序随K退化。B63几何/margin系列分别有未通过geometry gate、无训练、source/TEST饱和或margin harm，不能从region拟合准确率推断当前Q选择成立。Basin selector matrix停在preflight，根本没有可报告的训练/selector结果。详细证据处置见evidence_inventory.json。')

    title('7. Leakage / confounder audit 与证据衔接')
    para('通过：matched TEST与TRAIN/VAL family disjoint；新Ring88132..88139与整个新版source corpus的state UID/family均0重叠，target controller UID也未在source出现；v11两组state与原probe TRAIN/VAL无UID重叠。primary cached prediction hash与冻结checkpoint匹配；原预测在target rollout前冻结的记录保留；从DB按(state_uid, exact eta_uid, controller_uid, future_index)重算counts，与缓存逐项一致。33,588条per-state metric rows与668行原selection结果独立复核一致。')
    para('限制：这是阅读旧reports后的retrospective audit，不能宣称本轮metric/figure protocol在见任何outcome前注册；本轮fixed hash代表state不依赖分数/标签，所有模型seed和完整atlas可审查。Source VAL checkpoint-selection data只作supporting；旧target打开后的adaptation、kernel、shrinkage不得升级为独立TEST结论。Seen exact eta和同Ring geometry不能外推unseen eta、cross-scene或controller population。两新controller的headroom-based第二阶段没有触发；只报告预先固定第一批16+16，未把未执行的后32state计入。')
    para('此前“state-conditioned basin”与“Field reshapes basin”是物理outcome证据，不自动证明learned Q识别了它。前轮TT/FF主样本B15为135→209/512，rescue90、break16、29/32 state membership改变；旧49→88是factorial TT→TF，不是full FF。16/16→0/16 controller-swap案例也只证明B(x,eta,c)会变。本轮额外比较learned score是否跟随变化，并保留不能跟随的反例。')

    title('8. Scientific decision 与最小下一步')
    para('选择：STATE_CONDITIONED_BASIN_SELECTION_SUPPORTED（有限场景/固定bank范围）。理由是Toy TT同时有强within-state识别、真实state reversal识别、输入置换效应和实际成功选择；独立Toy48再次出现within-state排序收益并选中46/48，部分独立Ring K2也有收益。不能升至STATE_AND_FIELD_CONDITIONED_BASIN_SELECTION_SUPPORTED：新版field响应有signal，但held-controller K16的state reversal几乎失灵、训练seed不稳定、选择提升不稳定；跨scene/phase还有反例。若论文claim特指“当前Field formulation上的普遍额外选择价值”，现有FF表中eta-only已到oracle，尚未建立该额外价值。')
    para('瓶颈：within-state及field-dependent ranking；source/target phase与controller coverage；上尾分数过于自信与max-selection放大。Representation不足是现有H20/response aliasing支持的候选解释，不能仅凭失败数定为唯一原因。简单全局单调calibration不改变argmax，无法修复错误eta ordering。已有K16 Oracle很高，优先修正识别/排序假设，继续增加K不能解决主要ranking gap。')
    para('本轮不需要新rollout便能作上述判断。若下一步要补最关键的独立field反转证据：冻结现有88138/88139模型，按state UID固定hash各选4个既有TEST state，固定使用候选索引0..3，在另一个controller下补同16seeds；最多8×4×16=512条缺失counterfactual续跑，原controller结果直接复用。不得按已知success/反转筛state或eta；事先承认可能无可辨识reversal，不因结果不理想扩样或改模型。这是后续可选的配对诊断设计，本轮未执行，也不保证足以支持总体field-generalization。')

    title('9. Reviewer figures')
    for letter,name,caption in [('A','figure_A_q_over_basin','Outcome-blind hash representative：同一3D eta bank，eta1/eta2投影，点面积表示eta3；无插值边界。GT与三个训练seed的Q并排；星号是top1。'),('B','figure_B_ranking_reversals','固定hash选真实reversal，不按Q是否成功筛选：Toy成功、Ring held K16失败、Ring source VAL field reversal部分成功。source VAL明确标注。'),('C','figure_C_selection_decomposition','逐state区分oracle存在、成功选择、ranking error与candidate-generation failure；不是只画成功率bar chart。')]:
        para(f'Figure {letter}: '+caption+'\n'+str(HERE/'figures'/f'{name}.pdf'))
        encoded=base64.b64encode((HERE/'figures'/f'{name}.png').read_bytes()).decode();parts.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="Figure {letter}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    para('完整16-cohort、556个不同physical state的atlas：'+str(HERE/'figures/all_states_atlas.pdf')+'。每个cohort页含全部state、固定hash顺序、相同列的GT与预定义主模型全部seed；其他arms和controls的完整逐state表见per_state.csv.gz。')

    title('10. Paper-ready statement')
    statement=('On held-out true-t0 Toy states, the state-conditioned critic ranked robust interventions accurately within each state (AUROC 0.965–0.976) and jointly recovered 93.0–96.1% of empirical state-induced B15 ranking reversals on a shared candidate bank. '
               'An independent 48-state Toy replication achieved 46/48 robust selections versus 43/48 for a strong eta-only kernel, although this binary selection difference was not statistically resolved (paired exact p=0.375). '
               'Correct physical context improved recognition in newer held-controller tests, but field-dependent ranking and top-1 selection remained inconsistent across controllers and training seeds. '
               'The evidence therefore supports scoped state-conditioned basin recognition and selection, but does not establish reliable learning and selection from a general state- and field-dependent basin.')
    para(statement)
    (HERE/'paper_ready_statement.txt').write_text(statement+'\n')
    title('11. Reproduction / global paths')
    para('工作目录：'+str(HERE)+'\n复现命令：bash '+str(HERE/'reproduce.sh')+'\n可选重新只读导出：bash '+str(HERE/'reproduce.sh')+' --reexport\nPython: numpy/scipy 环境与matplotlib环境路径见reproduce.sh，可用AUDIT_PY / PLOT_PY覆盖。所有分析输入已导出inputs/；export.py及verify.py的原缓存复核需要原repo。没有训练/rollout执行入口。')
    para('代码：export.py、analyze.py、plot.py、report.py、verify.py、test_metrics.py、reproduce.sh。主要结果：summary.json、recognition_selection.csv、per_state.csv.gz、state_reversals.csv、paired_comparisons.csv、random_baselines.csv、field_response_reversals.csv、current_full_field_response_reversals.csv、source_VAL_recognition_selection.csv。provenance：protocol.json、cohorts.json、source_hashes.json、snapshot_integrity.json、cache_audit.json、verification.json。全部绝对路径见ARTIFACTS.txt。')
    (HERE/'FINAL_REPORT.txt').write_text('\n'.join(out)+'\n')
    css='body{font:16px/1.65 system-ui,sans-serif;max-width:1250px;margin:30px auto;padding:0 24px;color:#183042}h2{margin-top:2em;border-bottom:2px solid #c8d8df;padding-bottom:.3em}p{white-space:normal}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}th,td{padding:8px;border:1px solid #d1dce2;text-align:left;vertical-align:top}th{background:#e8f2f4}img{width:100%;height:auto}figure{margin:25px 0}figcaption{font-size:13px;color:#52616a}'
    (HERE/'FINAL_REPORT.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Q basin audit</title><style>'+css+'</style><body>'+''.join(parts)+'</body></html>')
    decision=dict(decision='STATE_CONDITIONED_BASIN_SELECTION_SUPPORTED',scope='Toy shared-bank and independent within-scene evidence; some Ring K2 confirmations; not a universal state/field guarantee',field_selection_supported=False,new_rollouts=0,new_training=0,
                  report=str(HERE/'FINAL_REPORT.html'),paper_ready_statement=statement)
    (HERE/'scientific_decision.json').write_text(json.dumps(decision,indent=2,ensure_ascii=False)+'\n')
    with (HERE/'main_table.csv').open('w',newline='')as f:
        w=csv.writer(f);w.writerow(['panel','oracle_eligible/all','full_selected/eligible','eta_only/eligible','TRAIN_fixed/eligible','random_eligible_expectation','ranking_gap']);w.writerows(main_table)
    paths=sorted(str(p.resolve())for p in HERE.rglob('*')if p.is_file()and'__pycache__'not in p.parts and p.name!='ARTIFACTS.txt')
    (HERE/'ARTIFACTS.txt').write_text('\n'.join(paths+[str(HERE/'ARTIFACTS.txt')])+'\n');print(str(HERE/'FINAL_REPORT.html'))

if __name__=='__main__':main()
