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
    gate = read('GATE_RESULTS.json')
    result = read('results.json')
    post = read('POSTFLIGHT.json')
    lab = read('LABEL_AUDIT.json')
    decomp = read('CONTROLLER_Q_DECOMPOSITION.json')
    env = read('ENVIRONMENT.json')
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
        classification = '(i) 迁移成立：通过独立 state 确认'
        transfer = f"SUPPORTED：Stage B 开发筛选并由 Stage C 新 state 独立确认的表示为 {', '.join(c['confirmed_arms'])}。此结论限于两个必要折、Ring FF、K16、H700；Stage B 的模型选择仍是开发证据。"
    elif c:
        classification = '(iii) Stage B 开发集通过；独立确认失败'
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
                          '/'.join(map(str, r['target_fixed_eta_crossfit'])),
                          pp(r['necessity_loss_lower']), f'[{pp(ci[0])}, {pp(ci[1])}]',
                          '是' if r['necessary'] else '否'])
    gate_table = table(['折', '留出真实 Flow', 'η_src index', '目标固定 η（交叉拟合）', 'L 下界', '95% CI', '必要折'], gate_rows)
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
    bpost = read('stage_b/POSTFLIGHT.json') if b else None
    cpost = read('stage_c/POSTFLIGHT.json') if c else None
    reused = (14 if b else 7) * 24 * 16 * 4
    new = post['raw'] + (bpost['raw'] if bpost else 0) + (cpost['raw'] if cpost else 0)
    raw_seconds = 0.0
    for folder in [HERE / 'raw'] + ([HERE / 'stage_b/raw'] if b else []) + ([HERE / 'stage_c/raw'] if c else []):
        for path in folder.glob('*.json'):
            raw_seconds += json.loads(path.read_text())['elapsed_seconds']
    p1_prior = ('上一轮 P1 在24个开发 state 上测到合格 Flow 的全局跨 controller η 遗憾中位数17.19pp '
                '（同时 CI [3.13,28.39]pp），但该面板已经打开，不能当作本轮新确认。')
    report = f'''# Real Flow LOCO：Controller-conditioned η selection 最终报告

## A. Research Overview

主问题：在源合并固定 η 对目标真实 Flow 明显不合适的留出 controller 上，部署时可静态计算的 controller 表示能否提高每个 state 的 η 选择闭环成功率？本研究把 Q 预测与 η 选择分开，主终点为独立 seed 的闭环成功率和预注册同时置信区间。{p1_prior}

本轮分类：**{classification}**。{transfer}

## B. Implementation Summary

先审计 `AGENTS.md`、Round 3 原始协议、14 个 Ring Flow checkpoint、shared DB 和历史标签。七个 [40%,95%] Flow 的 UID、checkpoint SHA、local/global observation frame 与 JAX x64=false 见 `EXECUTION_IDENTITY_AUDIT.json`。复用 Round 3 的 adapter、原生 v10 runtime、R3/R4_action/R4_openloop 静态特征和训练设置。所有新增文件在本目录；历史数据未修改、未做 Git commit。纯 Q 实验不要求视频。

用新的自然 state 序列生成128个 S_train、64个 S_test，七个 Flow 共用；全库/历史 state hash 查重为零，见 `STATE_DISJOINT_RECEIPT.json`。每个训练 state 分配8/16个 η、Q4；test 为K16×Q8，其中seed0–3仅用于目标固定 η 参照和必要性门，seed4–7仅用于最终评价。P1 的24个开发 state 仅供源验证 NLL 选步。训练为 physical23+表示→16-head MLP，3个 fit seed；只对有标签的头计损失。R0、R3、R4_action、R4_openloop、R3+R4_action用同一模型、数据、预算；R4基于η=0投影执行动作及投影后Jacobian，静态查询禁止 `env.step`。所有新 outcome 产生前的设计、决策门和时间/SHA凭证见下列预注册文件。

{table(['预注册文件', 'UTC 时间', 'SHA256'], [[name, rec['utc_time'], rec['sha256']] for name, rec in prereg])}

86,016个 Stage A 精确请求经官方 `python -m shared_rollout_db.plan` preflight，复用0、缺失86,016；DB experiment 行在 merge 前登记。98次计时 pilot、按 timing 的无结果重排、native v10 特征等价核查分别记录于 `PILOT_SAMPLING_DEVIATION.md`、`SCHEDULING_ADJUSTMENT.md`、`NATIVE_VALIDATION_FEATURE_AUDIT.json`。新 raw/journal 经单进程 merge，再做 postflight 和UID标签张量；数值失败保留为unknown。Stage A {post['raw']:,}个新 raw、{post['valid']:,} valid、{post['unknown']:,} unknown、0 missing、0 quarantine。七组静态特征查询均为0次 `env.step`。35组训练任务×3 fit seed＝105个 checkpoint；source-only归一化与P1验证选步。总复用P1验证标签约{reused:,}个已有逐seed格标签；本轮所有阶段新增{new:,}个 raw，原始episode时间合计约{raw_seconds/3600:.2f} CPU小时（未包含静态查询、训练、合并和排队时间）。

## C. Experimental Results

### Stage A：必要性门和 Controller–Q 关系

必要性门只读 S_test seed0–3：目标固定η用0–1↔2–3双向交叉拟合；源η来自六个源 Flow 的 S_train；对 unknown 使用配对保守下界。64个物理state成组bootstrap 2000次，门为 L≥5pp 且95%下界>0。必要折 {nnecessary}/7：{gate['necessary_folds']}。

{gate_table}

下表为全局层跨 controller 遗憾（源 Controller 的训练最优固定η在目标上的损失）；它与 state 层后验 oracle 和零样本模型选择并非同一指标。

{cross_table}

独立 evaluation seed 的完整逐折结果如下。成功率及差值采用unknown保守下界；Q MSE只是辅助指标。最近源由**源数据归一化后的 R4_action**均值距离决定。目标自身固定η使用目标 seed0–3，只作目标监督参照。

{endpoint}

主检验仅汇总必要折。每个表示需同时超过源固定η、最近源固定η、R0 critic，且正确描述优于六个错配描述；16项比较用同一物理state配对max-T bootstrap 5000次控制同时区间。Stage A 通过表示：{', '.join(passing) or '无'}。

{contrast_table}

少样本目标校准使用不同于 S_test 的前1/2/4个 S_train state，属于有目标标签的参照；不是zero-shot。`results.json`保留全部逐fit成功率、unknown上界、oracle regret和 η 排序辅助指标。

{calibration}

辅助 Q 分解：七个 Controller 的平均Q水平标准差 {decomp['metrics']['level_sd']:.4f}；controller×η 交叉半样本能量 {decomp['metrics']['controller_eta_energy_crossfit']:.6f}；state×controller×η 能量 {decomp['metrics']['state_controller_eta_energy_crossfit']:.6f}。各自bootstrap区间、pairwise η rank相关和unknown比例见 `CONTROLLER_Q_DECOMPOSITION.json`。这些离散K16量不识别连续success basin。

### 条件分支

{branch_details}

## D. Hypothesis Evaluation

- **真实 Flow 会改变 Q level**：SUPPORTED（七个固定几何和执行条件下的描述性均值）；数值和区间见 Q 分解。
- **真实 Flow 会改变η排序**：{'SUPPORTED' if nnecessary >= 2 else 'PARTIALLY SUPPORTED'}（必要折 {nnecessary}/7；全局与state层需分开，不能由Q MSE代替）。
- **现有静态表示能在未见真实 Flow 上可靠改善选择**：{transfer}
- **更多源 Controller 能修复失败**：{'PARTIALLY SUPPORTED（同一已打开面板）' if b and b['primary']['passing_arms'] else 'INCONCLUSIVE（未得到独立确认）'}。
- **短时静态探测表示优于投影动作/Jacobian**：仅看相应预注册配对差；若同时区间未过门则 REJECTED（本设计内），不得以Q MSE替代。

## E. Controller–Q Relationship and Transferability

固定 Ring geometry、FF placement、OrthoFlow3、硬安全投影、K16、H700、同一state及seed，只改变真实Flow checkpoint。Q水平差见分解；η排序变化见必要性门、跨 Controller 遗憾及pairwise rank。R3为compact64，R4_action为η=0投影动作与投影Jacobian，R4_openloop为静态短时探测；局部Jacobian不是完整闭环的精确线性化。匹配/错配表示的成功率差是机制表示的行为检验；Q MSE不证明selection。结果在七个真实Flow和必要折范围内解释，不能推至未知几何、placement或连续η空间。源 Controller 数从6→13只在Stage B实际执行时可检验，并且同一目标面板上的改善仍需独立确认。

## F. Limitations and Next Steps

七个Flow来自已有checkpoint库，不代表所有Controller；只固定Ring原生几何、FF、H700、K16及自然开发state分布。新64个 S_test state共享于七折，bootstrap按物理state成组；必要折由独立seed挑选，主终点只用seed4–7，但这仍是条件性结论。unknown按保守边界处理，代价是置信区间可能偏宽。Stage C受10,000新rollout上限约束，总64个state–controller面板会降低单折精度。零样本机制若失败，应首先诊断η排序与controller覆盖、真实闭环响应是否可由t0静态特征表征；不得从离散K16推断连续basin形状。

## G. Reproducibility and Evidence

工作根目录 `/home/zhihan/research/Basin_C1`；Python `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`；记录的Git HEAD `{env['git_head']}`，模型/runtime均JAX x64=false。精确controller checkpoint身份、state UID、η、seed、代码哈希在 `protocol.json`、`ENVIRONMENT.json`、`EXECUTION_IDENTITY_AUDIT.json`。运行命令与Slurm依赖见 `RUN_COMMANDS.md`；原始证据在 `raw/`、`static_features/`、`LABELS.npz`、`fits/fold*_{'{'}arm{'}'}/`、`GATE_RESULTS.json`、`CONTROLLER_Q_DECOMPOSITION.json`、`results.json`及触发分支目录。`POSTFLIGHT.json`、`LABEL_AUDIT.json`、`FINAL_CONSISTENCY_AUDIT.json`和 `ARTIFACT_INDEX.json`供独立核查。所有报告中的事实均来自这些机器文件；机制解释为局部/统计解释，跨几何或连续basin推断仍属待验证假设。
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

## 假设与根因边界

{transfer} Q水平、η排序和selection是不同命题。Q分解与排序相关见 `CONTROLLER_Q_DECOMPOSITION.json`，逐表示闭环成功率、配对差、Q MSE、少样本目标校准见 `results.json`。若模型失效，可支持的陈述仅限本配置下的静态表示或源Controller覆盖；不能把失败直接归因于网络结构，也不能把成功直接推广到跨几何/连续success basin。Stage B若执行，仅为同一已打开目标面板的开发证据；Stage C才是新的自然state确认。

## 后续任务入口

独立研究Agent应先读 `REPORT.md`、`PREREGISTRATION_STAGE_A.md`、`PREREGISTRATION_CONDITIONAL_BRANCHES.md`、`results.json`及 `FINAL_CONSISTENCY_AUDIT.json`。可直接复用 `protocol.json`、`LABELS.npz`、`fits/`逐seed checkpoint/predictions、`static_features/`、`GATE_RESULTS.json`、`CONTROLLER_Q_DECOMPOSITION.json`、条件分支目录和 `RUN_COMMANDS.md`。下一步最有价值的问题是：在固定闭环语义下，哪些controller静态或短时响应真正决定η排序；若七个Flow的必要折太少，需先设计能独立改变η优选的真实controller样本，再验证跨geometry。连续success basin需要额外密集η证据，本研究不作形状结论。
'''
    (HERE / 'REPORT.md').write_text(report)
    (HERE / 'HANDOFF.md').write_text(handoff)
    receipt = dict(utc_time=datetime.now(timezone.utc).isoformat(), branch=branch,
                   stage_a_necessary_folds=gate['necessary_folds'], stage_a_passing_arms=passing,
                   stage_b_executed=bool(b), stage_c_executed=bool(c),
                   total_new_raw=new, reused_p1_validation_label_slots=reused,
                   raw_episode_cpu_hours=raw_seconds / 3600)
    (HERE / 'REPORT_RENDER_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
