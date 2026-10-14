from core import *

def main():
    prereg();gate();cur=read(HERE/'CURVE_RESULTS.json');model=read(HERE/'CLASSIFIER_RESULTS.json');mech=read(HERE/'MECHANISM_RESULTS.json');post=read(HERE/'main_POSTFLIGHT.json');assert post['status']=='PASS'
    audit=dict(utc=utc(),preregistration_hash_verified=True,consistency_cases=32,main_requests=10240,postflight=post,prefix_audit=read(HERE/'PREFIX_AUDIT.json')['passed'],inherited_asset_audit=read(HERE/'INHERITED_ASSET_AUDIT.json')['passed'],historical_inputs_verified=7168,resource_correction=read(HERE/'RESOURCE_CORRECTION.json'),DEVELOPMENT=True)
    result=dict(status='COMPLETE',DEVELOPMENT=True,curves=cur,classifier=model,mechanisms=mech,audit=audit);write(HERE/'results.json',result);write(HERE/'FINAL_AUDIT.json',audit)
    half=', '.join(f"{n.replace('_macflow','')}={o['t_half']}步（unknown范围{o['t_half_unknown_range']}）" for n,o in cur['controllers'].items())
    stage='η效应呈controller依赖的混合时序：base_u/v10_local的η13收益需要早期暴露，而v9_local/plus_w_v4_wide的伤害仍可在较晚阶段引入'
    implication='这支持后期闭环交互是短时controller表示漏掉的证据，200步探测可能截断了决定性阶段；下一轮应检验后期状态或中途η选择。' if cur['H_late']=='SUPPORTED' else '这表明R5失败不能简单归因于200步horizon太短，更应检查早期统计量是否捕捉了相关状态与交互。' if cur['H_early']=='SUPPORTED' else '这没有把零样本controller泛化失败的根因统一定位到200步之后；需要保留controller各自的时刻曲线，而不能仅凭早期推进速度归因。'
    state_interpretation='可以把中途state-only critic作为下一轮候选另行验证，尚无未知controller的零样本认证。' if model['H_state_mediated']=='SUPPORTED' else '本轮固定状态模型不足以替代controller身份，不支持直接宣称已找到controller无关的中途critic，也没有未知controller的零样本认证。'
    text=f"""# Ring η切换时刻实验

**DEVELOPMENT。{stage}。** H_late={cur['H_late']}，H_early={cur['H_early']}；半效应时刻：{half}。{implication}本轮测量的是固定controller的闭环路径在何时仍能被η改变。状态预测门H_state_mediated={model['H_state_mediated']}；{state_interpretation}

已完成32例逐位一致性门和10240主干预请求；有效{post['valid']}，unknown {post['unknown']}。unknown保留、不重跑、不作为失败；所有timeout保持timeout。环场景原runtime没有deadlock分支，因此停滞候选不能改名为deadlock。阶段0复用7168历史记录，主曲线t0/t700用旧纯η标签，不补采完整基线。

| controller | D=S13−S10 | t_half | unknown可行范围 | 保留半效应的网格时刻 |
|---|---|---|---|---|
"""
    for n,o in cur['controllers'].items():text+=f"| {n} | {o['D_point']:.4f} | {o['t_half']} | {o['t_half_unknown_range']} | {o['passing_times']} |\n"
    text+=f"\n同一state bootstrap中，后期联合门通过比例={cur['late_bootstrap_gate_pass_fraction']}（有效抽样{cur['late_bootstrap_valid_draws']}/5000），早期联合门通过比例={cur['early_bootstrap_gate_pass_fraction']}（有效抽样{cur['early_bootstrap_valid_draws']}/5000）。这些是预定点门的抽样稳定性描述，不是后验概率；SUPPORTED仍是DEVELOPMENT点门结论，不是已证明的前瞻因果定理。\n"
    text+='\nH_early只对应两个η13受益controller的预定半收益门，不能推广到η13伤害的时刻，也不能解释为后续干预无法改变结局。H_late的两个收益时刻条件未满足，而两个伤害controller的300步保留伤害/200步反向恢复条件均满足，因此报告混合时序并保持INCONCLUSIVE。\n'
    text+='\n反向曲线提供直接事实：base_u前500步η13、后η10的成功下界为215/256=83.98%，纯η13为136/256=53.13%、纯η10为60/256=23.44%；v10_local前100步η13、后η10为252/256=98.44%，纯η13为236/256=92.19%、纯η10为148/256=57.81%；wide前500步η13、后η10为248/256=96.88%，纯η13为67/256=26.17%、纯η10为162/256=63.28%。这些是注册臂的描述性比较，含unknown时成功上界见曲线；没有据此认证或部署新的最优时刻。它们提示早期推进与后期收敛可以需要不同η，固定全程η的正负regime会压缩这种阶段差异。\n'
    text+='\n![配对成功率曲线](SUCCESS_CURVES.png)\n\n曲线为64个state成簇、5000次bootstrap的95%区间，seed在state内平均；unknown上下界均保留。CURVE_RESULTS.json包含与两个纯η基线的配对区间和全80比较的max-T同时区间，及t_half bootstrap分布/非正分母数。非单调曲线不插值、不强行单调。半效应是注册网格中最晚通过时刻，不能解释为连续物理过程的唯一瞬间。\n\n'
    text+=f"固定C=1、64 state留一的physical-state模型AUROC={model['metrics']['physical']['AUROC']:.4f}，Brier={model['metrics']['physical']['Brier']:.4f}；加入身份AUROC增益={model['ID_AUROC_gain']:.4f}、95%CI={model['ID_AUROC_gain_CI']}，logloss增益CI={model['ID_logloss_gain_CI']}。无ID充分={model['state_prediction_sufficient']}，ID增益等价小={model['ID_gain_equivalent_small']}。符合全部预定阈值才支持H_state_mediated；不显著不能证明等价。源fold独立标准化、没有调参；CLASSIFIER_FOLDS.json与CLASSIFIER_OOF.npz保存全部拟合和预测。\n\n"
    text+='H_state_mediated的REJECTED只针对本轮25个物理特征、固定线性logistic的预注册操作命题：此模型下controller身份仍有明显增益。它不证明所有非线性state-only表示都不可能，也不是正式因果中介分析。\n\n'
    text+='![机制描述](MECHANISM_GOAL_DISTANCE.png)\n\n事实：MECHANISM_RESULTS.json记录每臂timeout/collision/success/unknown及其发生步数、切换前终止比例、投影激活、实际simulator swept clearance；MECHANISM_ROWS.json记录每10步机器人位置导出的目标距离、速度、停滞候选、最小表面间距与近距离相反目标机器人所在位置。停滞定义仅是2秒最大位移<.02m且未到目标。轨迹时间曲线只使用当时仍有真实记录的轨迹，存在终止后的组成变化；不据此比较未观察到的失败反事实。近距离位置不是已验证的会车因果事件。\n\n'
    text+='![停滞候选与投影激活](MECHANISM_STALL_PROJECTION.png)\n\n![固定初态完整观测路径](MECHANISM_FIXED_PATHS.png)\n\n路径图固定选state0/seed0、300步切换，不按成功结局选择；纯η轨迹来自一致性门，不补跑。圆点为真实初态、星号为目标、叉号为切换位置。若案例unknown，只展示已观测的部分并明确标注；图像不作为安全或deadlock视频验收证据，安全仍以simulator连续检查为准。MECHANISM_EXAMPLES.json给出完整来源/hash。\n\n'
    text+='| controller | η方向与切换步 | success / timeout / collision / unknown | 终态停滞候选均数 | 安全投影比例均值 | 最小simulator swept clearance |\n|---|---|---|---|---|---|\n'
    for name,arms in mech['summary'].items():
        for arm,m in arms.items():
            o=m['outcomes'];text+=f"| {name.replace('_macflow','')} | {arm} | {o.get('success',0)} / {o.get('timeout',0)} / {o.get('collision',0)} / {o.get('numerical_failure',0)} | {m['mean_final_stall_candidates']} | {m['mean_projection_fraction']} | {m['minimum_swept_clearance']} |\n"
    totals={}
    for arms in mech['summary'].values():
        for m in arms.values():
            for outcome,count in m['outcomes'].items():totals[outcome]=totals.get(outcome,0)+count
    text+=f"\n主干预的结局类型总计为{totals}。上述表对每臂使用同一256个初态/seed请求；数值异常单列unknown。切换前已终止比例和各结局发生步数分布见MECHANISM_RESULTS.json。它们说明η效应是否主要体现为timeout转success或其他结局，而不是把timeout解释为已证实的死锁。\n\n"
    text+='在预先关注的300步切换上，与同初态/seed的纯source η基线相比，结局转移如下（unknown仍为numerical_failure）：\n\n'
    for name,arms in mech['summary'].items():
        for arm in ('10to13_t300','13to10_t300'):text+=f"- {name.replace('_macflow','')}，{arm}：{arms[arm]['paired_outcome_transitions_from_pure_source']}。\n"
    text+='\n'
    text+='解释和边界：切换干预能比较同初态、同seed在两段η执行下的结局；它不能把控制器身份与其诱导状态分布做正式因果中介分解。物理状态分类器只覆盖四个已知controller、存活到切换时刻的样本，存在survivor选择，不能宣称未知controller零样本泛化。旧大批基线没有轨迹，机制基线只有32例门验证完整轨迹；没有为机制补跑基线。PREFIX_AUDIT.json核对有门轨迹可配对的主轨迹在切换前逐位相同。\n\n'
    text+='执行与收据：预注册UTC和SHA见PREREGISTRATION.receipt.json；阶段0摘要在主运行前保存。schedule独立controller identity含原controller、两η float64 hex、时刻和冻结runner/core hash，标准eta_uid携带eta_a，DB schema不变。官方preflight只运行缺失项，worker只append journals，单进程官方merger使用本地精确现代seed/physical-timestep适配器，postflight核对所有raw/DB/trace hash。历史输入重新核对7168 SHA，没有修改历史资产、不commit。\n\n'
    text+='操作记录：初次资源统计误将ming用户4CPU计入zhihan配额；纠正后保留314条已完成raw，重新preflight其余9926缺失项，改为17 worker加1编排/分析CPU。不重新执行任何已完成或numerical_failure记录；旧任务取消时尚未写出记录的在途请求仍属missing。launch.py仅在journal transport把numpy数组转JSON，没有修改冻结仿真语义。所有已完成raw在merge前重放journal以恢复中断时尚未flush的记录，重复写入由唯一UID幂等去重。\n\n'
    text+='视频适用性：本轮单Ring固定Flow的成功概率Q和时间机制诊断，按AGENTS的纯Q例外与本轮机制范围，不运行四场景视频验收，不追加其他场景rollout。全程每步轨迹/swept安全证据已保留；没有将timeout或停滞候选伪造为deadlock_show。\n'
    (HERE/'REPORT.md').write_text(text)
    (HERE/'HANDOFF.md').write_text(f"""# Round9 独立交接

全部DEVELOPMENT。研究问题：Ring下controller依赖的η10/13效应在哪个阶段产生。先读REPORT.md（H_late={cur['H_late']}，H_early={cur['H_early']}，H_state_mediated={model['H_state_mediated']}），再读results.json。用户任务冻结全文在PREREGISTRATION.md，收据SHA已校验。

实验目录：{HERE}。Python={PY}。共享DB实验UID={EXP}。4 Flow×64原loco_test state×seed0..3×双向×5时刻=10240，另32一致性门；H700、FF、JAX x64false，唯一干预action index>=t替换η。原baseline来自round4，不补跑；unknown不是失败、不能重跑。

代码：core.py/run.py/protocol.json被冻结，勿修改；launch.py只转换journal数组。design.py登记/官方preflight；submit.py统计zhihan自己的CPU任务；merge.py调用官方单进程merger并精确保留现代seed_key；postflight.py核对raw/DB/trace；analysis.py统计/机制，classifier.py固定LOO，report.py报告。finalize.py串行恢复journal→merge→postflight→analysis→classifier→report→COMPLETE，失败留FINALIZE_ERROR，不自动补跑unknown。

证据：main/raw与main/traces，gate/raw与gate/traces；STAGE0_SUMMARY.md和HISTORICAL_RAW_INDEX.json；CONSISTENCY_GATE.json；两phase CACHE_PREFLIGHT/POSTFLIGHT；SLURM_SUBMISSIONS.json、RESOURCE_CORRECTION.json、merge_audits、FINAL_AUDIT.json。CLASSIFIER_INPUT/OOF.npz和FOLDS、CURVE_RESULTS、MECHANISM_ROWS/RESULTS完整可重算。所有统计只读真实证据。

复算（不rollout）：先核对COMPLETE与POSTFLIGHT及预注册hash，运行analysis.py、classifier.py、report.py；会重写派生报告，保留输入immutable。没有新几何/未知controller认证，无正式中介证明。下一轮是否采用后期state-only critic需依本轮门结果另行预注册。本轮无git commit、不删资产；DB只登记本实验新身份/rollout并更新本实验metadata。\n""")
if __name__=='__main__':main()
