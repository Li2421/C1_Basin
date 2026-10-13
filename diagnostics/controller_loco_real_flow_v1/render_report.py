"""Render auditable outcome reports from frozen machine-readable analyses."""
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def pct(x):
    return f'{100 * x:.2f}%'


def pp(x):
    return f'{100 * x:+.2f}pp'


def table(head, rows):
    return '\n'.join(['| ' + ' | '.join(head) + ' |',
                      '| ' + ' | '.join(['---'] * len(head)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    from controller_prior_analyze import main as analyze_prior
    prior_analysis = analyze_prior()
    from plot_results import main as plot_results
    plot_results()
    gate = read('GATE_RESULTS.json')
    result = read('STAGE_A_RESULTS_FROZEN.json')
    post = read('POSTFLIGHT.json')
    lab = read('LABEL_AUDIT.json')
    decomp = read('CONTROLLER_Q_DECOMPOSITION.json')
    env = read('ENVIRONMENT.json')
    protocol = read('protocol.json')
    eta_table = table(['η index（0-based）','三维episode级η'],[[i,', '.join(f'{v:.6f}' for v in values)] for i,values in enumerate(protocol['eta'])])
    dispatch = read('CONDITIONAL_DISPATCH.json')
    branch = dispatch['branch']
    bfile = HERE / 'stage_b/results.json'
    cfile = HERE / 'stage_c/results.json'
    b = json.loads(bfile.read_text()) if bfile.exists() else None
    c = json.loads(cfile.read_text()) if cfile.exists() else None
    if b and c:
        branch = 'STAGE B THEN STAGE C (prospective addendum)'
    if b:
        assert b and b['status'] == 'COMPLETE' and (HERE / 'stage_b/SOURCE_COUNT_COMPARISON.json').exists()
    if c:
        assert c and c['status'] in ('CONFIRMED', 'FAILED INDEPENDENT CONFIRMATION')
    prereg = [(p.name, read(p.with_suffix('.receipt.json').name)) for p in sorted(HERE.glob('PREREGISTRATION*.md'))]
    assert all(r['sha256'] for _, r in prereg)
    primary = result['primary']
    passing = primary['passing_arms']
    nnecessary = len(gate['necessary_folds'])
    if nnecessary < 2:
        classification = '(iv) 必要折不足：主问题无法检验'
        transfer = 'INCONCLUSIVE：达到预注册必要性门槛的真实 Flow 少于两个，不能用七折描述性结果判断 controller 表示的选择迁移。'
    elif c and c['confirmed_arms']:
        classification = '(iii) Stage B 配置的迁移通过独立 state 确认；覆盖瓶颈的因果解释未确认'
        transfer = f"SUPPORTED：Stage B 开发筛选并由 Stage C 新 state 独立确认的表示为 {', '.join(c['confirmed_arms'])}。此结论限于两个必要折、Ring FF、K16、H700；Stage B 的模型选择仍是开发证据。"
    elif c:
        classification = '(ii) 必要折上未得到独立确认；Stage B 开发集有正信号'
        transfer = 'INCONCLUSIVE：Stage B 的13源设置在已打开面板上过门，但冻结模型未在新state同时超过全部对照；不能宣称稳健的跨Controller选择迁移。'
    elif b and b['primary']['passing_arms']:
        classification = '(iii) 增加源 Controller 后有开发集收益'
        transfer = 'PARTIALLY SUPPORTED：Stage B 的13源设置通过同一已打开测试面板的联合门，但未有独立确认；不能把覆盖增加解释为因果修复。'
    elif passing:
        classification = '(i) Stage A 通过；独立确认未通过'
        transfer = 'INCONCLUSIVE：Stage A 的独立 seed 选择结果过门，但 Stage C 的新 state 未确认；不能宣称稳健的跨 Controller 迁移。'
    else:
        classification = '(ii) Controller 信息必要，但现有表示失败'
        transfer = 'REJECTED（本设计内）：至少两个折需要 Controller 信息，但测试的静态表示均未同时超过三个对照及错配对照。不能据此否定一切 Controller 表示。'
    gate_rows = []
    for r in gate['rows']:
        ci = r['necessity_ci95_conservative']
        gate_rows.append([r['fold'], r['controller'], r['source_prior_eta'],
                          r['target_train_best_eta'],
                          '/'.join(map(str, r['target_fixed_eta_crossfit'])),
                          pp(r['necessity_loss_lower']), f'[{pp(ci[0])}, {pp(ci[1])}]',
                          '是' if r['necessary'] else '否'])
    gate_table = table(['折', '留出真实 Flow', 'η_src index', 'train最佳固定η', '目标固定 η（交叉拟合）', 'L 下界', '95% CI', '必要折'], gate_rows)
    state_gate_table = table(['折','state-oracle交叉拟合成功率下界','vs源固定η下界'],
                             [[r['fold'],pct(r['state_oracle_crossfit_success_lower']),pp(r['state_oracle_gain_over_source_lower'])] for r in gate['rows']])
    cross = gate['cross_controller_regret']
    cross_table = table(['源 Flow', '目标 Flow', '源 η', '目标 η', '遗憾下界'],
                        [[r['source'], r['target'], r['source_train_eta'], r['target_train_eta'],
                          pp(r['cross_controller_regret_lower'])] for r in cross])
    rows = result['rows']
    endpoint = table(['折', '表示', '必要', '成功率下界', 'vs η_src', 'vs 最近源', 'vs R0', 'matched−mismatched', 'oracle regret下界', 'Q MSE'],
                     [[r['fold'], r['arm'], '是' if r['necessary'] else '否',
                       pct(r['selected_success_lower']), pp(r['conservative_gain']['src']),
                       pp(r['conservative_gain']['near']), pp(r['conservative_gain']['R0']),
                       pp(r['conservative_gain']['mismatch']),
                       pp(r['hindsight_state_oracle_regret_lower']), f"{r['q_mse']:.4f}"] for r in rows])
    contrasts = primary.get('contrasts', [])
    contrast_table = table(['表示', '配对差', '均值', '95% 同时 CI'],
                           [[r['arm'], r['comparison'], pp(r['mean']),
                             f"[{pp(r['ci95_simultaneous'][0])}, {pp(r['ci95_simultaneous'][1])}]"] for r in contrasts]) if contrasts else '必要折不足，未做确认性主对比。'
    calibration = table(['折', '表示', '目标 train state 数', '目标固定 η 成功率', 'Critic+offset 成功率', '观测 η 头数'],
                        [[r['fold'], r['arm'], k,
                          pct(v['fixed_success_lower']), pct(v['critic_offset_success_lower']),
                          v['observed_eta_heads']]
                         for r in rows if r['necessary'] for k, v in r['target_supervised_fixed_calibration'].items()]) if nnecessary else '无必要折。'
    branch_details = ''
    if b:
        comp = read('stage_b/SOURCE_COUNT_COMPARISON.json')
        branch_details = ('Stage B **完成**：同一 S_test 面板上增加七个外窗口真实 Flow 作为源；'
                          '该面板已被 Stage A 打开，属于 DEVELOPMENT。独立的新增 rollout/unknown 见 '
                          '`stage_b/POSTFLIGHT.json`。\n\n' +
                          table(['表示', '13源−6源成功率下界', '95% 同时 CI'],
                                [[r['arm'], pp(r['stage_b_minus_stage_a_lower']),
                                  f"[{pp(r['ci95_simultaneous'][0])}, {pp(r['ci95_simultaneous'][1])}]"]
                                 for r in comp['rows']]) +
                          '\n\nStage B 联合判据通过表示：' + (', '.join(b['primary']['passing_arms']) or '无') + '。')
        branch_details += ('\n\nStage B 的完整必要折终点与主对比（开发证据）：\n\n' +
                           table(['折','表示','成功率下界','vs源固定η','vs R0','matched−mismatched'],
                                 [[r['fold'],r['arm'],pct(r['selected_success_lower']),pp(r['conservative_gain']['src']),pp(r['conservative_gain']['R0']),pp(r['conservative_gain']['mismatch'])] for r in b['rows']]) +
                           '\n\n' + table(['表示','对照','差值','95%同时CI'],
                                 [[r['arm'],r['comparison'],pp(r['mean']),f"[{pp(r['ci95_simultaneous'][0])}, {pp(r['ci95_simultaneous'][1])}]"] for r in b['primary']['contrasts']]) +
                           '\n\n不能将Stage B过门直接解释为源覆盖不足被修复：R3和组合表示的Stage B−A同时区间均覆盖0；R0下降约2pp，组合表示的总体成功率也略降。fold1变好而fold4变差，源增加并非逐折单调改善。两个必要折中源先验与最近源均选择η10，因此(a)/(b)实际相同；目标自身固定η使用目标标签，仍明显更强。\n\n' +
                           table(['折','表示','目标state数','目标固定η成功率','critic+offset成功率'],
                                 [[r['fold'],r['arm'],k,pct(v['fixed_success_lower']),pct(v['critic_offset_success_lower'])] for r in b['rows'] for k,v in r['target_supervised_fixed_calibration'].items()]))
    elif c:
        branch_details = ('Stage C **完成**：新自然 state、冻结 Stage A checkpoint/归一化/选择规则；'
                          f"共 {sum(int(x) for x in c['allocation'].values())} 个 state–controller 面板，"
                          f"状态为 **{c['status']}**；确认表示：{', '.join(c['confirmed_arms']) or '无'}。\n\n" +
                          table(['表示', '配对差', '均值', '95% 同时 CI'],
                                [[r['arm'], r['comparison'], pp(r['mean']),
                                  f"[{pp(r['ci95_simultaneous'][0])}, {pp(r['ci95_simultaneous'][1])}]"]
                                 for r in c['contrasts']]))
    else:
        branch_details = 'Stage B/C **NOT EXECUTED — GATE FAILED**：必要折少于两个，按预注册不追加标签。'
    if b and c:
        branch_details += ('\n\nStage C **完成**：按 `PREREGISTRATION_STAGE_C_AFTER_B.md` 的前瞻性补充规则，'
                           '在新自然 state 上使用冻结 Stage B checkpoint、归一化和源 prior。'
                           f"共 {sum(int(x) for x in c['allocation'].values())} 个 state–controller 面板，"
                           f"状态 **{c['status']}**；确认表示：{', '.join(c['confirmed_arms']) or '无'}。\n\n" +
                           table(['表示', '配对差', '均值', '95% 同时 CI'],
                                 [[r['arm'], r['comparison'], pp(r['mean']),
                                   f"[{pp(r['ci95_simultaneous'][0])}, {pp(r['ci95_simultaneous'][1])}] "]
                                  for r in c['contrasts']]))
        branch_details += ('\n\n' + table(['折','表示','确认state数','选择成功率下界','源固定η','R0','目标标签固定η'],
                              [[r['fold'],r['arm'],r['states'],pct(r['selected_success_lower']),pct(r['source_prior_success_lower']),pct(r['R0_success_lower']),pct(r['target_fixed_success_lower'])] for r in c['rows']]) +
                           '\n\nStage C生成64个去重初态，实际两个controller各标注相同前32个初态：32个独立物理state、64个state–controller面板、8,192请求。原任务每controller64个state需16,384次，会超过10,000硬上限；此缩减在新outcome前冻结。Stage C只检验Stage B冻结的R3和组合表示以及R0对照，不重新训练、调参或选择。')
    bpost = read('stage_b/POSTFLIGHT.json') if b else None
    cpost = read('stage_c/POSTFLIGHT.json') if c else None
    stages_table = table(['阶段','执行状态','新rollout','数值unknown','决策'],
                         [['Stage A','完成',post['raw'],post['unknown'],'2必要折；主联合判据无表示通过'],
                          ['Stage B','完成' if b else '未执行',bpost['raw'] if bpost else 0,bpost['unknown'] if bpost else 0,', '.join(b['primary']['passing_arms']) if b else '门未触发'],
                          ['Stage C','完成' if c else '未执行',cpost['raw'] if cpost else 0,cpost['unknown'] if cpost else 0,c['status'] if c else '尚无确认结果']])
    reused = (14 if b else 7) * 24 * 16 * 4
    new = post['raw'] + (bpost['raw'] if bpost else 0) + (cpost['raw'] if cpost else 0)
    raw_seconds = 0.0
    for folder in [HERE / 'raw'] + ([HERE / 'stage_b/raw'] if b else []) + ([HERE / 'stage_c/raw'] if c else []):
        for path in folder.glob('*.json'):
            raw_seconds += json.loads(path.read_text())['elapsed_seconds']
    prior_tables = '\n\n'.join('**'+stage+'（EXPLORATORY）**\n\n'+table(['表示','逐state选择−学习到的controller固定η','95%同时CI'],
                                  [[r['arm'],pp(r['state_selection_minus_learned_controller_prior_lower']),f"[{pp(r['ci95_simultaneous'][0])}, {pp(r['ci95_simultaneous'][1])}]"] for r in data['contrasts']]) for stage,data in prior_analysis.items())
    cover = read('SOURCE_REGIME_COVERAGE.json')
    coverage_table = table(['Controller','组','训练经验最佳η（成功率下界）','最佳η成功率下界','η13成功率下界'],
                            [[r['controller'],r['group'],r['observed_lower_bound_best_eta'],pct(r['best_lower']),pct(r['eta13_lower'])] for r in cover['rows']])
    q_summary = table(['阶段','表示','必要折平均Q MSE（辅助）'],
                       [[stage,arm,f"{sum(r['q_mse'] for r in data['rows'] if r['necessary'] and r['arm']==arm)/2:.4f}"]
                        for stage,data in [('Stage A',result),('Stage B',b)] if data for arm in ['R0','R3','R4_action','R4_openloop','R3_R4_action']])
    handoff_branches = ''
    if b:
        handoff_branches += ('Stage B（13源，DEVELOPMENT）：\n\n' +
                            table(['表示','对照','差值','95%同时CI'],
                                  [[r['arm'],r['comparison'],pp(r['mean']),f"[{pp(r['ci95_simultaneous'][0])}, {pp(r['ci95_simultaneous'][1])}]"] for r in b['primary']['contrasts'] if r['arm'] in b['primary']['passing_arms']]) +
                            '\n\n增加源后的R3/组合表示相对Stage A总体成功率无显著改善，R0反而下降；不能据过门认定controller覆盖是唯一或主要根因。')
    if c:
        handoff_branches += ('\n\nStage C（冻结Stage B模型，32个全新物理state、两个controller、64个面板）：\n\n' +
                            table(['表示','对照','差值','95%同时CI'],
                                  [[r['arm'],r['comparison'],pp(r['mean']),f"[{pp(r['ci95_simultaneous'][0])}, {pp(r['ci95_simultaneous'][1])}]"] for r in c['contrasts']]))
    p1_prior = ('上一轮 P1 在24个开发 state 上测到合格 Flow 的全局跨 controller η 遗憾中位数17.19pp '
                '（同时 CI [3.13,28.39]pp），但该面板已经打开，不能当作本轮新确认。')
    report = f'''# Real Flow LOCO：Controller-conditioned η selection 最终报告

## A. Research Overview

主问题：在源合并固定 η 对目标真实 Flow 明显不合适的留出 controller 上，部署时可静态计算的 controller 表示能否提高每个 state 的 η 选择闭环成功率？本研究把 Q 预测与 η 选择分开，主终点为独立 seed 的闭环成功率和预注册同时置信区间。{p1_prior}

本轮分类：**{classification}**。{transfer}

{stages_table}

![Controller selection gains](figures/controller_selection_gains.png)

图中每阶段使用各自的同时CI族；单个对比为正不代表四项联合判据通过。最后一行为探索性的逐state选择相对模型自身controller固定ηprior的差值。可导出的矢量图为 `figures/controller_selection_gains.pdf`。

## B. Implementation Summary

先审计 `AGENTS.md`、Round 3 原始协议、14 个 Ring Flow checkpoint、shared DB 和历史标签。七个 [40%,95%] Flow 的 UID、checkpoint SHA、local/global observation frame 与 JAX x64=false 见 `EXECUTION_IDENTITY_AUDIT.json`。复用 Round 3 的 adapter、原生 v10 runtime、R3/R4_action/R4_openloop 静态特征和训练设置。所有新增文件在本目录；历史数据未修改、未做 Git commit。纯 Q 实验不要求视频。

用新的自然 state 序列生成128个 S_train、64个 S_test，七个 Flow 共用；全库/历史 state hash 查重为零，见 `STATE_DISJOINT_RECEIPT.json`。每个训练 state 分配8/16个 η、Q4；test 为K16×Q8，其中seed0–3仅用于目标固定 η 参照和必要性门，seed4–7仅用于最终评价。P1 的24个开发 state 仅供源验证 NLL 选步。训练为 physical23+表示→16-head MLP，3个 fit seed；只对有标签的头计损失。R0、R3、R4_action、R4_openloop、R3+R4_action用同一模型、数据、预算；R4基于η=0投影执行动作及投影后Jacobian，静态查询禁止 `env.step`。所有新 outcome 产生前的设计、决策门和时间/SHA凭证见下列预注册文件。

固定Ring为4-agent单积分器：中心障碍半径0.9、外边界半径3.25、agent半径0.16、最大速度0.52、dt=0.05、H700即35秒、到达容差0.09；安全/连续碰撞检查和成功终止语义沿用P1。η在episode内固定，K16完整保留（η13为下表index13，未删除强固定候选）。

{eta_table}

{table(['预注册文件', 'UTC 时间', 'SHA256'], [[name, rec['utc_time'], rec['sha256']] for name, rec in prereg])}

86,016个 Stage A 精确请求经官方 `python -m shared_rollout_db.plan` preflight，复用0、缺失86,016；DB experiment 行在 merge 前登记。98次计时 pilot、按 timing 的无结果重排、native v10 特征等价核查分别记录于 `PILOT_SAMPLING_DEVIATION.md`、`SCHEDULING_ADJUSTMENT.md`、`NATIVE_VALIDATION_FEATURE_AUDIT.json`。新 raw/journal 经单进程 merge，再做 postflight 和UID标签张量；数值失败保留为unknown。Stage A {post['raw']:,}个新 raw、{post['valid']:,} valid、{post['unknown']:,} unknown、0 missing、0 quarantine。七组静态特征查询均为0次 `env.step`。35组训练任务×3 fit seed＝105个 checkpoint；source-only归一化与P1验证选步。总复用P1验证标签约{reused:,}个已有逐seed格标签；本轮所有阶段新增{new:,}个 raw，原始episode时间合计约{raw_seconds/3600:.2f} CPU小时（未包含静态查询、训练、合并和排队时间）。

Stage B复用同一128个训练初态和η分配，为7个额外兼容Flow补28,672个确切缺失请求，10组必要折×表示任务产生30个fit-seed checkpoint；总计135次模型拟合。`TRAINING_PROVENANCE_AUDIT.json`逐一复算45组模型的源归一化，验证与保存值完全一致、留出controller没有进入源训练/验证列表，并记录外部代码、P1标签和物理特征资产哈希。静态表示与checkpoint推理阶段均为零env.step；新controller输入编码使用既有标签，没有另补rollout。

已有P1数据复用的21,504个逐seed格中，21,232为有效标签、272为数值unknown；unknown没有计入训练/验证NLL的成功或失败，见 `HISTORICAL_LABEL_REUSE_COUNTS.json`。每折的验证列表只含其源controller。

执行调整：原分支预注册把Stage C限定为Stage A过门，Stage B过门后按用户原始“有表示达到主判据”要求补写 `PREREGISTRATION_STAGE_C_AFTER_B.md`，冻结Stage B模型后才生成新的确认outcome。该补充是事前于Stage C、事后于Stage B的明确修订。最终资源授权为使用所有空闲CPU/GPU；因19-way采集分片已运行，保持其UID所有权，并把冻结推理/审计放入剩余CPU槽，与采集并行，GPU并发上限8。没有为了资源调整重复或中断rollout。实际调度记录在 `STAGE_C_RESOURCE_UPDATE.json`、`STAGE_C_AFTER_B_DISPATCH.json`，每个raw的内容哈希在 `RAW_CONTENT_INDEX_STAGE_AB.jsonl` / `RAW_CONTENT_INDEX_STAGE_C.jsonl`。

## C. Experimental Results

### Stage A：必要性门和 Controller–Q 关系

必要性门只读 S_test seed0–3：目标固定η用0–1↔2–3双向交叉拟合；源η来自六个源 Flow 的 S_train；对 unknown 使用配对保守下界。64个物理state成组bootstrap 2000次，门为 L≥5pp 且95%下界>0。必要折 {nnecessary}/7：{gate['necessary_folds']}。

{gate_table}

state层交叉拟合oracle（同一个已有state在独立seed上选择和评价，不是未见state的可部署selector）：

{state_gate_table}

下表为全局层跨 controller 遗憾（源 Controller 的训练最优固定η在目标上的损失）；它与 state 层后验 oracle 和零样本模型选择并非同一指标。

{cross_table}

独立 evaluation seed 的完整逐折结果如下。成功率及差值采用unknown保守下界；Q MSE只是辅助指标。最近源由**源数据归一化后的 R4_action**均值距离决定。目标自身固定η使用目标 seed0–3，只作目标监督参照。

{endpoint}

主检验仅汇总必要折。每个表示需同时超过源固定η、最近源固定η、R0 critic，且正确描述优于六个错配描述；16项比较用同一物理state配对max-T bootstrap 5000次控制同时区间。Stage A 通过表示：{', '.join(passing) or '无'}。

{contrast_table}

少样本目标校准使用不同于 S_test 的前1/2/4个 S_train state，属于有目标标签的参照；不是zero-shot。`results.json`保留全部逐fit成功率、unknown上界、oracle regret和 η 排序辅助指标。

{calibration}

辅助 Q 分解：七个 Controller 的平均Q水平标准差 {decomp['metrics']['level_sd']:.4f}；controller×η 交叉半样本能量 {decomp['metrics']['controller_eta_energy_crossfit']:.6f}；state×controller×η 能量 {decomp['metrics']['state_controller_eta_energy_crossfit']:.6f}。各自bootstrap区间、pairwise η rank相关和unknown比例见 `CONTROLLER_Q_DECOMPOSITION.json`。这些离散K16量不识别连续success basin。

辅助分解的初版曾隐式将unknown按0参与能量，已在最终审计中修正并保存旧版本。现用unknown=[0,1]的中点及正交投影/Cauchy界进行敏感性分析，bootstrap区间扩展以覆盖unknown完成值，不声称任意缺失情况下严格无偏。修正后的controller×η能量区间[0.023235,0.034666]，state×controller×η区间[0.011014,0.034132]，仍为正；详见 `DECOMPOSITION_UNKNOWN_POLICY_AUDIT.md`。主判据和模型没有受到该辅助修正影响。

### 条件分支

{branch_details}

### Controller级prior与逐state选择的分解（探索性）

使用冻结模型在128个S_train初始state上的Q预测取均值，选择一个controller级固定η，完全不读取该controller的目标标签。η及checkpoint哈希在Stage C新outcome读取前冻结，见 `EXPLORATORY_CONTROLLER_PRIOR_PROTOCOL.md`、`controller_prior/`。这是使用无标签目标初态的zero-target-label prior估计，与1/2/4个目标state的有监督calibration不同；其统计对比不改变主判据。以下比较才直接检验逐state选择是否额外优于模型自身已识别的controller η regime。

{prior_tables}

### Q预测与源η-regime覆盖（描述性/探索性）

{q_summary}

R3/组合表示的Q误差在Stage B优于R0，R4动作/短时探测并未因此获得选择过门证据。Q标签只有4个endpoint seed且含unknown，MSE是同一已观察面板上的辅助误差，不能解释成真实连续Q的精确误差或替代Stage C选择确认。

{coverage_table}

7个新增Flow在本轮训练标签上的经验最佳η为10/2/4，没有新增经验最优η13的controller。每个必要折原有6源中的η13经验最优源只有另一个必要controller；增加到13源后仍只有这一个。故Stage B增加了controller总数，却没有增加这个关键η-regime的经验最佳controller例数；这是可审计的覆盖缺口，尚不能证明它是唯一失败原因。窗口资格来自P1的24个分层开发state，不按本轮自然state成功率重新筛选Flow。完整曲线、unknown上下界及候选分配在 `SOURCE_REGIME_COVERAGE.json`。

## D. Hypothesis Evaluation

- **真实 Flow 会改变 Q level**：SUPPORTED（七个固定几何和执行条件下的描述性均值）；数值和区间见 Q 分解。
- **真实 Flow 会改变η排序**：{'SUPPORTED' if nnecessary >= 2 else 'PARTIALLY SUPPORTED'}（必要折 {nnecessary}/7；全局与state层需分开，不能由Q MSE代替）。
- **Controller仅改变Q整体水平、η排序不变**：REJECTED（本固定Ring面板）；两个必要折的固定η交叉拟合损失及controller×η交互不支持纯level解释。
- **本轮表示已达到独立确认主判据**：{'SUPPORTED' if c and c['confirmed_arms'] else 'REJECTED'}（预注册操作性成功条件）；这不等同于拒绝所有正效应，真实效应大小仍须按CI解释。
- **现有静态表示能在未见真实 Flow 上可靠改善选择**：{transfer}
- **源controller覆盖是失败的主要根因**：INCONCLUSIVE；增加源的Stage B−A配对成功率差未显示R3/组合表示显著提升，且增加controller同时增加标签量并改变源分布。不能把Stage B过门等同于因果瓶颈被定位。
- **R4_action/R4_openloop提供可靠选择迁移**：本设计中没有通过预注册联合判据；它们的普遍充分性仍为INCONCLUSIVE。Q误差改善不能替代选择检验。

## E. Controller–Q Relationship and Transferability

固定 Ring geometry、FF placement、OrthoFlow3、硬安全投影、K16、H700、同一state及seed，只改变真实Flow checkpoint。Q水平差见分解；η排序变化见必要性门、跨 Controller 遗憾及pairwise rank。R3为四组compact64拼接（256维）；R4_action为η=0投影动作与有限差分Jacobian，查询t0及沿目标方向平移0.1的静态probe（总64维，差分eps=0.001）。R4_openloop拼接t0动作/Jacobian与一次p←p+dt·a0、v←a0的解析运动探测，无env.step，亦非真实短rollout。局部Jacobian不等同于完整闭环success basin。physical23包含初始径向距离、相位、目标径向距离/角偏差及局部速度；每臂输入统一补零至343维，网络64→64→16，12,000步，AdamW lr0.0008、wd0.0001、clip5，batch128。训练controller均衡采样，验证NLL按3个fit seed均值选共同checkpoint步；特征最终标准化仅使用源训练样本。

匹配/错配差是部署表示影响选择的行为证据，不单独识别因果机制；主终点由上面的联合对比和独立确认判定。新增source后的已打开面板与新state确认分开解释；逐state选择相对学习到的controller prior的增益另列探索性对比。未测试新几何、placement或连续η，不能推广；R4的局部作用不足并不证明静态表示原则必然无效。

事实：R4的Jacobian在η=0查询，但K16第一维约0.53–1.25且没有η=0候选。机制猜想：有限幅度FF积分、硬投影active-set变化及后续交互可能使局部响应不足以预测η排序；本轮没有针对这些环节的因果消融，不能将它们写成已定位的失败根因。错配仅替换critic输入的controller描述，实际闭环标签始终来自正确留出controller；Stage B/C错配仍对原来的其他6个controller取平均。

## F. Limitations and Next Steps

七个Flow来自已有checkpoint库，不代表所有Controller；只固定Ring原生几何、FF、H700、K16及自然开发state分布。新64个 S_test state共享于七折，bootstrap按物理state成组；必要折由独立seed挑选，主终点只用seed4–7，但这仍是条件性结论。unknown按保守边界处理，代价是置信区间可能偏宽。Stage C受10,000新rollout上限约束，总64个state–controller面板会降低单折精度。零样本机制若失败，应首先诊断η排序与controller覆盖、真实闭环响应是否可由t0静态特征表征；不得从离散K16推断连续basin形状。

置信区间描述这组固定模型、固定controller上的state/seed波动，未估计无限controller总体的泛化误差；两个必要折不能代表所有真实Flow。3个fit seed先各自选择η并评价，再平均成功率，主终点不是logit ensemble；逐fit差异保留在结果与预测。1/2/4-state calibration只使用预先固定的前k个训练初态，没有重复抽取calibration子集，因此其样本效率为描述性。学习到的controller固定ηprior的增益分解是新增探索性分析，不能替换确认主判据，也不能由其零目标标签性能声称物理因果机制。

Stage C按冻结规则未通过可靠迁移主门，并不证明真实收益为零：R3的同时CI仍包含约+11pp的正效应。之后若基于这些结果修改模型，Stage A的64个和Stage C的32个已打开物理state都应作为DEVELOPMENT，不可重新包装为独立确认。`stage_c/protocol.json`中另32个生成初态只做了静态查询、没有rollout标签，`LABELS.npz`相应位置为-2（unrequested）；本任务不追加，未来如补证需新协议/预算、重新preflight并记录该出处。

## G. Reproducibility and Evidence

工作根目录 `/home/zhihan/research/Basin_C1`；Python `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`；记录的Git HEAD `{env['git_head']}`，模型/runtime均JAX x64=false。精确controller checkpoint身份、state UID、η、seed、代码哈希在 `protocol.json`、`ENVIRONMENT.json`、`EXECUTION_IDENTITY_AUDIT.json`。运行命令与Slurm依赖见 `RUN_COMMANDS.md`；原始证据在 `raw/`、`static_features/`、`LABELS.npz`、`fits/fold*_{'{'}arm{'}'}/`、`GATE_RESULTS.json`、`CONTROLLER_Q_DECOMPOSITION.json`、`results.json`及触发分支目录。`POSTFLIGHT.json`、`LABEL_AUDIT.json`、`FINAL_CONSISTENCY_AUDIT.json`和 `ARTIFACT_INDEX.json`供独立核查。所有报告中的事实均来自这些机器文件；机制解释为局部/统计解释，跨几何或连续basin推断仍属待验证假设。

`STAGE_A_RESULTS_FROZEN.json`保留Stage A原始分析；`stage_b/results.json`与`stage_c/results.json`保留分阶段结果，最终 `results.json` 的 `study_summary` / `conditional_stages` 聚合全任务，原顶层Stage A rows/primary为兼容保留。重算统计与报告无需新rollout，运行本目录 `analyze.py`（Stage A）或 `analyze.py --stage-b`、`stage_b_compare.py`、`stage_c_analyze.py`、`controller_prior_analyze.py`、`render_report.py`、`final_audit.py`。现成checkpoint可直接用于新初态静态查询；没有target label输入时使用 `stage_c_predict.py` 的冻结推理路径，不能把目标校准offset默认当zero-shot。
'''
    handoff = f'''# Controller-conditioned η selection：独立研究交接

## 核心问题与背景

在未见过的真实 Ring Flow controller 上，模型是否能只用部署时可算的初始state和controller静态查询，为每个state选η，并在**确实需要controller信息的折**上可靠超过源固定η、最近源固定η和无controller输入的critic？上一轮P1已证实Flow改变K16的偏好，但P2′留出TF/FT与源先验同为η13，无法检验controller表示。详见 `REPORT.md` 与 `../controller_q_transfer_round3_v1/HANDOFF.md`。

## 实际实施

七个P1合格Flow做LOCO，固定Ring几何、FF、H700、K16、安全投影、评价器和state。全新128训练+64测试自然state通过历史hash去重；训练每state8个η×Q4，测试K16×Q8。已有P1的24验证state标签被复用；Stage A新增{post['raw']:,}个raw（{post['unknown']:,}个unknown），全阶段新增{new:,}个，累计原始episode约{raw_seconds/3600:.2f} CPU小时。Stage A 35组模型任务、105个fit seed checkpoint。先用test seed0–3做Controller必要性门，再仅用seed4–7作主终点；按物理state配对max-T同时置信区间。分支决定：{branch}。执行身份、哈希、preflight、merge和运行命令见 `REPORT.md` B/G节。

## 最重要的结果

必要折 **{nnecessary}/7**：{gate['necessary_folds']}。Stage A过主门表示：{', '.join(passing) or '无'}。{'Stage C独立确认：'+(', '.join(c['confirmed_arms']) or '无') if c else 'Stage C未执行。'} {'Stage B 13源开发面板过门：'+(', '.join(b['primary']['passing_arms']) or '无') if b else 'Stage B未执行。'} 最终分类：**{classification}**。

{gate_table}

{contrast_table}

Stage B/C的量化结论与controller-prior分解：

{handoff_branches}

{prior_tables}

## 假设与根因边界

{transfer} Q水平、η排序和selection是不同命题。Q分解与排序相关见 `CONTROLLER_Q_DECOMPOSITION.json`，逐表示闭环成功率、配对差、Q MSE、少样本目标校准见 `results.json`。若模型失效，可支持的陈述仅限本配置下的静态表示或源Controller覆盖；不能把失败直接归因于网络结构，也不能把成功直接推广到跨几何/连续success basin。Stage B若执行，仅为同一已打开目标面板的开发证据；Stage C才是新的自然state确认。

## 后续任务入口

独立研究Agent应先读 `REPORT.md`、`PREREGISTRATION_STAGE_A.md`、`PREREGISTRATION_CONDITIONAL_BRANCHES.md`、`results.json`及 `FINAL_CONSISTENCY_AUDIT.json`。可直接复用 `protocol.json`、`LABELS.npz`、`fits/`逐seed checkpoint/predictions、`static_features/`、`GATE_RESULTS.json`、`CONTROLLER_Q_DECOMPOSITION.json`、条件分支目录和 `RUN_COMMANDS.md`。下一步最有价值的问题是：在固定闭环语义下，哪些controller静态或短时响应真正决定η排序；若七个Flow的必要折太少，需先设计能独立改变η优选的真实controller样本，再验证跨geometry。连续success basin需要额外密集η证据，本研究不作形状结论。

补充入口：`PREREGISTRATION_STAGE_C_AFTER_B.md`说明独立确认分支修订；`STAGE_A_RESULTS_FROZEN.json`、`stage_b/results.json`、`stage_c/results.json`是原始分阶段统计，最终 `results.json.study_summary` 为完整聚合。`controller_prior/`区分无目标标签的controller η regime识别与逐state选择；`TRAINING_PROVENANCE_AUDIT.json`记录源-only归一化和135个checkpoint，`RAW_CONTENT_INDEX_STAGE_*.jsonl`逐episode内容哈希。最重要的未解问题是controller表示是否仅识别全局η regime、哪些state×controller交互能迁移；增加controller覆盖的因果改善并未由Stage B−A比较证实。

已核查的关键覆盖缺口：新增7个源在训练标签中的经验最佳η为10/2/4，没有增加η13经验最优的controller。因此每个必要折6→13源后，η13最佳源例数仍只有1，见 `SOURCE_REGIME_COVERAGE.json`。下一任务应先复用现有标签/预测检验η-regime覆盖、Q拟合与排序目标、以及state残差选择；不要把controller数量增加直接等同于机制覆盖。独立确认的R3相对源固定η为+5.21pp但CI[-0.66,+11.08]pp，相对R0为+5.73pp但CI[-0.13,+11.59]pp；正确描述优于错配+7.51pp、CI[+2.53,+12.49]pp。这支持controller信息影响决策，但未达到可靠迁移主门，亦未证明真实效应为零。逐state选择相对无标签controller-prior也未得到独立正收益。

Q预测单独判断：必要折平均MSE从Stage B的R0 0.1046降至R3/组合0.0762，这是辅助拟合收益；其与选择确认失败同时成立。目标标签选出的固定η13在Stage C两个折分别64.84%/89.06%，明显高于零样本R3的31.25%/69.01%，但它使用了目标标签。1/2/4-state calibration只做固定预选子集的描述性对照，不能声称普遍样本效率。复用数据为21,504个P1 seed格（21,232有效、272 unknown）；全任务122,880新episode，1,307 unknown均保留，0 missing/0 quarantine。

证据开封边界：本轮Stage A的64个及Stage C的32个已评估物理state现已打开。下一任务若据此选模型/表示/超参，其结果只能作为DEVELOPMENT。Stage C协议中其余32个初态尚无rollout标签（-2），但已存在静态特征和冻结模型预测；使用它们补证必须另写事前协议和预算，不能覆盖本轮证据。本轮停止新增数据。

注意数字巧合：本轮Stage C的R3 +5.21pp是相对两个留出controller上的源固定η10；它与历史Ring Q512相对本controller最佳η13的+5.21pp属于不同数据、controller与baseline，不能互相替代。
'''
    (HERE / 'REPORT.md').write_text(report)
    (HERE / 'HANDOFF.md').write_text(handoff)
    from final_results import main as aggregate_results
    aggregate_results(classification)
    receipt = dict(utc_time=datetime.now(timezone.utc).isoformat(), branch=branch,
                   stage_a_necessary_folds=gate['necessary_folds'], stage_a_passing_arms=passing,
                   stage_b_executed=bool(b), stage_c_executed=bool(c),
                   total_new_raw=new, reused_p1_validation_label_slots=reused,
                   raw_episode_cpu_hours=raw_seconds / 3600)
    (HERE / 'REPORT_RENDER_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
