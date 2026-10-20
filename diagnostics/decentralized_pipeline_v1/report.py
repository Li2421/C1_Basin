from core import *
def pct(v):return f'{100*v:.2f}%'
def pp(v):return f'{100*v:+.2f}pp'
def regret(v):return f'{100*v:.2f}pp'
def bounds(x,fn=pp):return '['+', '.join(fn(v) for v in x)+']'
def main():
 r=read(HERE/'results.json');v=read(HERE/'videos/manifest.json');audit=read(HERE/'STATE_DEDUP_AUDIT.json');d1=r['DC1'];d2=r['DC2'];d4=r['DC4_status'];status={'SUPPORTED':'通过','REJECTED':'不通过','INCONCLUSIVE':'unknown 边界跨阈值，尚不能确定'}
 safety=f"主评估出现 {r['collisions']} 条碰撞，以下成功率解释均以此安全问题为前提。" if r['any_collision'] else '主评估连续运动检查记录中未出现碰撞；这只是本批样本证据，不是普遍安全证明。'
 intro=f"在 D-shared 分散式安全层下，时间表优势 DC1 **{status[d1['status']]}**，一律 S500 的 controller 独立性判据 DC2 **{status[d2['status']]}**，相对冻结右手规则两基线的 DC4 **{status[d4]}**。{safety}"
 if r['any_collision']:intro=safety+' '+intro.replace(safety,'')
 lines=['# Round28：分散式执行管线评估','',intro,'','## 1. 数据、冻结与执行','',f"14 个原 Ring Flow best.pkl × 32 个新状态 × 14 个执行臂 × Q2 = {r['main_n']:,} 条主评估；另有 32 次特权专家规划，以及用户要求的 16 例历史 TT 一致性复现。无训练、无 critic 选择。组件门通过，单机器人投影最大误差为 0；集中式 TT 的 first action、事件、长度与完整位置逐位一致。",'',f"预注册、参数、种子区间与实现 SHA256 见 PREREGISTRATION.md、PREREGISTRATION.receipt.json、IMPLEMENTATION_FROZEN.receipt.json。pilot 保留 202610280000–202610280007；评估候选 202610281000–202610281999，按结果无关去重取前 32，实际采用 202610281000–202610281031。检查历史 {audit['DB_rows']} 条 state 的身份与可恢复物理状态；{audit['unresolved_physical']} 条历史记录无法恢复物理状态，因此不能声称对这些记录也完成了逐物理状态核验（所有结构身份均查重）。详见 STATE_DEDUP_AUDIT.json 与 HISTORY_AUDIT.json。",'','资源：主评估 18 个 CPU shard、0 GPU；专家在主评估已有空闲 CPU 后使用 1 个 CPU；最终分析等待两者完成，未叠加超过用户授权额度。全部导入后显式恢复 JAX x64=False；几何/投影保留原生 float64。','',f"主评估 unknown={r['unknown']}，不重跑、不作为已知失败；表内下界暂代 0、上界代 1，识别区间与 CI 分开报告。Ring 未启用正式 deadlock 判据；timeout 与 ≥2 秒低速诊断均不称为 deadlock。",'','### 流程偏差（保留原始证据）','','main Slurm 21301 在 12:29:10+08 提交，preflight 于 12:29:11.694609+08 封存，workers 于 12:29:13+08 启动。依赖保证 preflight 先于执行，但不满足 README 的“先于提交”顺序。原收据保留为 main_PREFLIGHT_ORIGINAL_RECEIPT.json，当前收据已更正，SUBMISSION_TIMING_AUDIT.json 记录事实；不声称本轮零流程偏差。此偏差未触发按结果改参数或重复 rollout。专家 preflight 则在提交之前完成。 首次后处理因主环境没有 matplotlib 中断，已改为调用已有绘图/视频环境；统计仍使用指定 Python，未改环境或重跑 rollout，见 POSTPROCESS_REPAIR.json。最终后处理作业为 21339，因正式视频角色缺口预期返回非零。','', '## 2. 成功率与安全','', '| 管线/臂 | 成功下界–上界 | 下界统计的 95% CI | 碰撞 | unknown | timeout |','|---|---:|---:|---:|---:|---:|']
 for name,x in r['rates'].items():lines.append(f"| {name} | {pct(x['success_lower'])}–{pct(x['success_upper'])} | {bounds(x['ci_lower_bound'],pct)} | {x['collisions']} | {x['unknown']} | {x['timeouts']} |")
 lines+=['','每格 896 条；保留配对的两粒随机未来种子。95% CI 为先 controller、后 state 成簇抽样 5,000 次；臂和 seed 保持配对，每次重新选最佳臂。CI 针对本 14 配方/32 状态总体，不等于更广泛 controller 家族证明。','', '## 3. 预注册终点','',f"- **DC1 {status[d1['status']]}**：最佳时间表减最佳固定 η = {pp(d1['estimate'])}，下界标签统计 CI {bounds(d1['ci_lower_label_bound'])}；unknown 相容赋值下的 CI 下端点范围 {bounds(d1['lower_ci_endpoint_range'])}；controller 优势覆盖确定 {d1['dominance_definite']}/14、可能 {d1['dominance_possible']}/14，门槛 10/14。",f"- **DC2 {status[d2['status']]}**：always-S500 相对四臂 oracle 平均遗憾 {regret(d2['mean_regret'])}，unknown 识别区间 {bounds(d2['regret_identification_interval'],regret)}，门槛 3pp；最差 controller 的下界标签遗憾 {regret(d2['worst_controller_regret'])}。S500 减逐 controller 最佳固定 η = {pp(d2['S500_minus_best_fixed']['estimate'])}，CI {bounds(d2['S500_minus_best_fixed']['ci_lower_label_bound'])}。",f"- **DC4 {status[d4]}**：以下两项必须同时 CI 下界 >0。"]
 for name,x in r['DC4'].items():lines.append(f"  - S500 − {name}：{pp(x['estimate'])}，CI {bounds(x['ci_lower_label_bound'])}；unknown 下 CI 下端点范围 {bounds(x['lower_ci_endpoint_range'])}，{status[x['status']]}。")
 if d1['status']=='SUPPORTED' and d2['status']=='SUPPORTED' and d4=='SUPPORTED':conclusion='依照预注册映射，时间表优势、低 controller 遗憾与对冻结 RHR 基线的优势在本轮 D-shared 管线中均保留；策略仍需要共享观测。'
 elif d1['status']=='SUPPORTED' and d4!='SUPPORTED':conclusion='时间表优势仍存在，但未证实优于两个冻结右手规则基线；方法相对经典基线的定位需要据此限制。'
 elif d1['status']=='SUPPORTED':conclusion='时间表优势和对冻结右手规则基线的优势均保留，但 controller 独立性还需分别按 DC2 判据解释。'
 elif d1['status']=='REJECTED':conclusion='分散式主管线未通过时间表优势门；依照预注册映射，此前优势不能脱离集中式安全层条件直接推广。DC3 的成功与停滞变化列于下节。'
 else:conclusion='unknown 的相容边界跨越主门，不能确定时间表优势是否在分散式安全层下保留。'
 if d2['status']!='SUPPORTED':conclusion+=' 一律 S500 的 controller 独立性判据也未通过，不能保留相应泛化主张。'
 lines+=['',conclusion]
 lines+=['','以上 oracle 是已评估四臂中逐 controller 的事后最佳，仅用于定义预注册遗憾/效应，不能当作无需标签的部署选择器。RHR 仅指本轮冻结的 .01m/s、2s、.15m/s 顺时针切向启发式；不能代表所有经典方法。','', '## 4. 分散化代价与感知敏感性','', '| 比较 | 臂 | 成功率差 | 95% CI（下界标签统计） | 碰撞率差 | unknown率差 | 停滞 episode 率差 |','|---|---|---:|---:|---:|---:|---:|']
 for key in ['DC3_shared_minus_central','DC5_local_minus_shared']:
  for a,x in r[key].items():
   m=x['descriptive_metrics'];lines.append(f"| {key} | {a} | {pp(x['estimate'])} | {bounds(x['ci_lower_label_bound'])} | {pp(m['collision_rate']['difference'])} | {pp(m['numerical_failure_rate']['difference'])} | {pp(m['stall_episode_rate']['difference'])} |")
 lines+=['','成功差异 unknown 识别界及碰撞/数值/停滞诊断的配对 CI 均在 results.json。','', '| 管线/臂 | 停滞 episode比例 | 平均停滞机器人数 | 平均累计停滞机器人秒 | 最长单次秒 |','|---|---:|---:|---:|---:|']
 for name,x in r['rates'].items():lines.append(f"| {name} | {pct(x['stall_episode_rate'])} | {x['mean_stalled_robots']:.3f} | {x['mean_stall_total_seconds']:.3f} | {x['maximum_stall_seconds']:.2f} |")
 lines+=['','### 新管线失败构成','']
 for a in ['F10','F13','S300','S500']:
  q=r['rates']['shared/'+a];z=r['rates']['central/'+a]
  lines.append(f"- {a}：D-shared / 集中式 TT 的 timeout 为 {q['timeouts']} / {z['timeouts']}，unknown 为 {q['unknown']} / {z['unknown']}，停滞 episode 比例为 {pct(q['stall_episode_rate'])} / {pct(z['stall_episode_rate'])}。")
 lines+=['','局部约束要求每个机器人独自承担半份 pair 下界；集中式约束只限制两者贡献之和，允许在机器人间重新分配避碰责任。因此局部可行域增加了个体责任限制；它是本轮结构差异。停滞和终止构成提供失败模式诊断，不足以证明经典力平衡死锁。']
 lines+=['','停滞：执行速度 <.01m/s、未到达目标且连续至少 40 步；累计时长包括该合格低速段的全部步。数值失败前的诊断段保留，受未知后续截尾影响。人数为每条 episode 至少一次合格低速段的不同机器人数。','', '### 数值失败类型','']
 for e in r['numerical_failure_types']:lines.append(f"- {e['pipeline']} / {e['type']}: {e['count']} 条；{e['message']}")
 if not r['numerical_failure_types']:lines.append('无数值失败。')
 e=r['expert'];t=e['planning_seconds'];lines+=['','## 5. 集中式专家上界参照','',f"32 个状态各规划一次，成功 {e['success']}/32，unknown {e['unknown']}，碰撞 {e['collisions']}；成功率识别区间 {pct(e['success_lower'])}–{pct(e['success_upper'])}。规划耗时均值 {t['mean']:.3f}s、中位数 {t['median']:.3f}s、最小 {t['min']:.3f}s、最大 {t['max']:.3f}s、总计 {t['total']:.3f}s。候选规划内部采用原生双方向搜索；所选轨迹再按实际 simulator 验证并保留连续运动检查。这是特权离线参照，不是与策略具有同等信息/在线预算的竞争方法。",'', '## 6. 视频交付与缺口','',f"已渲染 {len(v['videos'])} 个完整仿真视频（每帧、1×速度、从真实初态到终态），视频/trace 哈希、控制器、checkpoint、seed、native swept clearance 和初态哈希见 videos/manifest.json。",'']
 for item in v['videos']:lines.append(f"- [{item['role']}](videos/{item['file']})：{item['frames']} 帧；正式 deadlock 角色={item['formal_deadlock_role']}。")
 lines+=['',f"统一入口已执行，退出码 {v['official_entry']['exit_code']}，见 official_video_entry.log；Ring-only 检查结果：{v['official_entry']['ring_scope_validation_error']}。Ring 没有正式 deadlock 判据，两个正式 deadlock 角色不能满足，未将 timeout/停滞伪装成死锁。因此 videos/manifest.json 的 complete=false，最终 launcher 非零退出，**本轮数值分析完成，正式 batch 视频验收未完成**。本任务只授权 Ring 评估，未为其他三个场景额外生成 rollout。",'','## 7. 解释边界与交付','', 'D-shared 和 D-local 的安全优化、修正与切换均由机器人本地计算，但策略仍需要共享全队观测和公共 RNG/时钟；不能声称完全独立观测部署。D-local 只限制安全约束和 B_rel 的 1.5m 邻居，不改变策略观测。每对约束各承担一半、假设邻居零速度；本轮未交换投影动作。物理、成功、碰撞与事件语义均未改动。controller payload 中遗留的 chain=FF 是复制原 Flow profile 的描述字段；本轮实际执行以 conditioning/execution_semantics、冻结 runner/pipeline 哈希及 TT 一致性门为准，未将该字段用于缓存复用。见 METADATA_AUDIT.json。', '', '数据链路：main_PLANNED_ROLLOUTS.json → main_CACHE_PREFLIGHT.json → main/raw + main/traces → journals → 单进程官方 merger → main_CACHE_POSTFLIGHT.json。专家独立执行身份走同样流程。MERGE_RECEIPT.json 核验 DB、原始记录和 unknown 一致，unknown 不触发补跑。results.json、rollout_metrics.csv、analysis_arrays.npz、figures/ 为分析产物；所有历史源文件保持原样。', '', '![三管线成功率](figures/pipeline_success.png)','','![右手规则比较](figures/baseline_comparison.png)','','![停滞诊断](figures/stall_diagnostics.png)','']
 (HERE/'REPORT.md').write_text('\n'.join(lines))
 handoff='''# Round28 handoff

Scope: docs/prompt_round28_decentralized_pipeline_20261015.md. All new files live in this directory; historical code unchanged, no commit. Python: /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python. Start by reading REPORT.md, results.json and PREREGISTRATION.md plus receipts.

## Execution and evidence

- design.py/setup and main_design already executed; do not recreate frozen files or states.
- pipeline.py and run.py are frozen new decentralized implementation and rollout runner. gate.py aggregates the 16 historical TT consistency replays; UNIT_TESTS.json covers local components. IMPLEMENTATION_FROZEN.receipt.json hashes immutable action code.
- Main Slurm21301: 18 CPU workers, no GPU, 12544 cases, full traces. Expert21319 uses one freed CPU after7 main shards finished:32 states, one privileged plan each. Worker errors are unknown, never rerun.
- Main submission preceded preflight sealing by roughly two seconds; dependency guaranteed sealing before execution. Read SUBMISSION_TIMING_AUDIT.json and original/current receipt; do not erase this deviation or claim preflight preceded submission.
- merge.py is the only SQLite merger; wraps official Ingestor with exact identity/hash validation. main/expert_CACHE_POSTFLIGHT.json + MERGE_RECEIPT.json reconcile all12576 new task rows. Gate replays are not inserted as new labels. Journals become .merged.
- Fresh-state audit checked DB identity and recoverable geometry, but some historical rows have unresolved physical state. STATE_DEDUP_AUDIT.json records the limitation, no fabricated complete physical dedup claim.

## Reproduce analysis from existing labels only

Run from repository root, one CPU, with OMP/MKL/OPENBLAS threads1:

```bash
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/decentralized_pipeline_v1/analysis.py
/home/zhihan/research/Basin_C1/.venv-video/bin/python -B diagnostics/decentralized_pipeline_v1/videos.py
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/decentralized_pipeline_v1/report.py
```

videos.py returns2 because formal deadlock roles are unavailable; run report.py regardless of that expected rejection. Never use `&&` to suppress report after this video gap. r28_finalize.sbatch captures the status, generates report, then propagates nonzero video acceptance. Existing supplementary videos are truthful diagnostics, not formally accepted deadlock videos. Manifest complete=false is intentional and required. No extra simulations to find illustrative successes.

## Statistical details and boundaries

14 controllers,32 states,2 future seeds,14 arms. Hierarchical bootstrap5000 controller thenstate clusters, all arms/seeds paired, oracles reselected per draw. Unknown preserved; success lower0/upper1 only for bounds, not a known failure. DC1/DC4 decision uses range of LOWER CI endpoints across compatible unknown assignments, not full CI overlap. DC2 exact regret interval excludes S500 self comparison. The four-arm oracle is descriptive in-sample. DC3/DC5 safety and stalls are descriptive. Ring has no formal deadlock criterion.

Policy inference still shares observations/public key; each robot executes only its own queried row. New local safety/correction does not establish independent-observation policies. RHR comparison covers only preregistered fixed heuristic. Geometry and native continuous safety/event checks unchanged, x64False restored after all imports; local QP geometry remains float64. No tuning after outcomes. Plot rendering uses the existing .venv-c1-plots Python3.12; videos use existing .venv-video Python3.12. Numerical analysis and all rollouts remain on prescribed .venv-c1 Python3.11. No environments were modified. POSTPROCESS_REPAIR.json preserves the missing-matplotlib failure and source hashes.

## Status

Numerical analysis complete. Formal batch completion remains FALSE due to unavailable formal Ring deadlock evidence. Read videos/manifest.json, VIDEO_ACCEPTANCE.json and official_video_entry.log. Keep all raw data, traces, checkpoints and DB evidence; no cleanup authorized.
'''
 (HERE/'HANDOFF.md').write_text(handoff);write(HERE/'DELIVERY_STATUS.json',dict(utc=utc(),numerical_analysis_complete=True,formal_batch_complete=False,report_sha256=sha(HERE/'REPORT.md'),handoff_sha256=sha(HERE/'HANDOFF.md'),video_manifest_sha256=sha(HERE/'videos/manifest.json'),video_role_gap=['deadlock_show','deadlock_resolution']));print('Report written; formal delivery incomplete')
if __name__=='__main__':main()
