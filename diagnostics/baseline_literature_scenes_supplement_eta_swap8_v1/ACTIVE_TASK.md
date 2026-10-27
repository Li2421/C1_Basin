# 2026-10-23 21:52 SGT 原主线已完成；用户新增搜索已获明确授权

S COMPLETE于2026-10-23T13:00:21Z（21:00 SGT）生成，status=complete_with_explicit_gaps。本轮再次核验COMPLETE内全部报告/DB/结束审计SHA一致。DB核查187936 distinct requests（包括保留的旧约束/旧布局作废证据和复用，不是全部有效主线规模），31 unknown保留，pending journals0/errors0；原八机有效主线34176。旧driver/resources/local guard/processwatch均已按完成条件退出，勿重启原driver或把PROGRESS最后statistics_pending视为当前未完成。

用户明确答复第三阶段即刚完成主线，要求接着搜索最佳correction，前置条件已满足。另最新要求先给已跑完报告，本回合先报告，尚未提交任何新搜索rollout。历史证据已读原16固定/6配对及Ring R9/R10/R21设计；旧Ring动力学/安全层/修正语义与现有八机不兼容，不能直接复用执行标签；八机已有标签精确身份可复用。新增搜索需独立协议/收据/EXP/新状态验证、训练与验证分开、原COMPLETE/报告保持原SHA；见FOLLOWUP_BEST_CORRECTION_REQUEST_20261023与FOLLOWUP_AUTHORIZATION_20261023_AFTER_COMPLETE。后续默认八机free/static/moving，重点moving，固定参数+时间/进度切换及必要多次切换；有限搜索内最佳，不承诺全局最优或必胜eta13。

新资源要求：无其他用户占用时可用全部24CPU含管理（23worker+1管理）；有他人使用时本用户总上限18，仍受物理空闲容量约束。当前shaowen6和seunghwan6在跑，另1pending，勿抢CPU；管理动态pin，勿硬编码CPU23。

已完成开发评估：free eta9 32.29% vs eta13 18.49%，差+13.80pp CI[2.73,26.69]；static eta0_to_3 29.56% vs28.12%，+1.43pp CI[-10.94,12.89]；moving从free迁移eta15_to_9 31.70% vs25.89%，+5.80pp CI[-2.83,14.73]，碰撞237/672 vs263/672，仍不安全。全部是开发证据，moving仅7类合格（MPPI零合格）；free/static训练池缺MPPI但评估含该类型。

下方均为历史状态，不应覆盖上述COMPLETE与新增授权。

# 2026-10-23 20:08 SGT 当前接手入口

用户要求继续并盯进程。driver PID3928776、resource guard3119937、local CPU guard3166717、观察监控3939898存活；PROCESS_WATCH_STATUS每20秒刷新，异常在PROCESS_WATCH_INCIDENTS/ERRORS和DRIVER_ERROR/FATAL_STOP。监控只观察，不改raw、不自动重试unknown、不把长运行当deadlock。管理CPU动态pin到空闲核心（最近17），不要硬编码旧23。

new_pair_train1344全部完成合库：615success/729timeout/0collision/0unknown，前后只读stat通过。NEW_FROZEN_CHOICES已在评估前封存：free eta9与eta15_to_9；static eta0与eta0_to_3，两者都缺centralized_mppi训练类型。仅训练赢家，不提前宣称评估超过eta13。当前new_eval job28486_0..9，官方5632请求=2816精确复用+2816新补，第一批raw已产出，无协调器错误。其他用户12CPU+1pending，因此本批10worker，管理1；配额上限仍18，不抢他人CPU。后续driver自动moving_eval（free赢家迁移）→统计/诊断/DB/结束审计→part2 COMPLETE，当前COMPLETE不存在。

最新最佳correction后续授权见FOLLOWUP_BEST_CORRECTION_REQUEST_20261023，第三阶段后才做；最终HANDOFF生成器已加入指针，原主分析不变。早期更下面的状态已过时，但冻结/身份/只读规则保持。

# 2026-10-23 后续授权及恢复

新增用户任务：第三阶段完成后寻找最佳correction，固定η及时间/进度切换表一并纳入后续设计；见FOLLOWUP_BEST_CORRECTION_REQUEST_20261023.json。当前原流程继续，不提前启动新增搜索，不用已看过的评估集证明泛化。用户所指第三阶段完成需核实，S COMPLETE本身不自动代表该前提。

new_fixed_train已完成合库。19:32 pairs planner因缺少np导入退出，已补辅助脚本显式import，两scene全部16排名/独立均值校验通过，原错误归档。启动同一driver.py续跑，marker避免重复raw；冻结科学源码和实验结果未改。

# 2026-10-23 19:05 SGT 实际进度更新

能力门修订两版49152已完成合库：static45/48、moving42/48，均collision0/unknown0；free43/48精确保留。旧场景fixed/pairs/eval及free_main全部BATCH_DONE。现job28436_0..16跑swap_main，17worker+1管理，driver PID3747920健康，无DRIVER_ERROR/FATAL_STOP。静态主线11712条全部完成，无碰撞；移动仍在跑，具体动态计数见STATUS_SNAPSHOT_20261023_1905.json及PROGRESS。当前移动碰撞局均有episode-level约束违反和restoration，但尚未做collision-step因果审计，不把此统计写成已证明原因。unknown按规则保留不重试。后续new fixed/pairs/eval、moving transfer及统计审计未完成，S COMPLETE不存在。下方更早进度已过时，科学冻结/不可变规则仍有效。

# 最新优先：2026-10-23用户修正已执行

用户最新明确只修moving约束/static布局，旧场景筛选和free不中断，碰撞/成功/95%门不变。该修正覆盖下方历史中的moving0合格/static12合格作为主分析的状态，旧两版现已标“原约束/原布局（作废）”保留，不进主分析。

当前科学实现：`swap_geometry_user_fix_20261023.py`、`swap_safety_user_fix_20261023.py`、`swap_runtime_user_fix_20261023.py`、`nominal_geometry_user_fix_20261023.py`、`controllers_layout_user_fix_20261023.py`、`case_factory_user_fix_20261023.py`、`worker_user_fix_20261023.py`，已由USER_FIX_IMPLEMENTATION_FROZEN与USER_FIX_READY封存。原IMPLEMENTATION_FROZEN所有SHA仍一致，不改已封存文件；未封存调度/分析辅助可修，新科学bug须新文件/新身份，不覆盖证据。

移动：去掉.0125一步膨胀，n·u>=.25−(gamma/.05)g，g=distance−(.35+.001001)，epsilon=.01*.1001，gamma=.3。六个gamma各164初始膨胀余隙×200步全过，10000随机快照8844 premise/1156 outside，premise failures0，unknown0；outside全部保留，SNAPSHOTS.npz SHA由门文件固定。推导见USER_FIX_DERIVATION：||relative(tau)||>=D+(1−gamma*tau/dt)g>=D，障碍速度界也涵盖步内waypoint转折；范围外.5−(.5+.25)*.05=.4625>D在原速度球满足时安全；放松速度/违反约束则不冒充保证。

static：角度/向心凹口保持，R=1.1m，L=.2726464281937122m。两变量LP纯几何最大化同时避开4条±.15名义走廊及32×8实际扰动直线的机器人净空。名义最小.1690593>.15，实际最小.055101>.0551(原physical margin.055+buffer.0001)，固定数值余量1e-6。仅名义走廊上界.293276会让state26实际直线净空.03645<.055，所以不是满足全部条件的最大值。设计、活跃约束/对偶、实际path检查见LAYOUT_REVISION_DESIGN；未看controller标签选几何。

物理SwapEnv.step和moving_path直接继承原函数，参数/state/seed都不变；8类型free已有完整轨迹逐位回放全过。静态nominal复制旧controllers_interior仅改墙geometry accessor import，算法/params不改。write-path新strace probe通过，旧native libs SHA一致，PA/CBS full key包含wallbytes/margin，旧layout memo不会错误复用。所有输出仅S/default journals。

当前入口仍`driver.py`，现为runpy静态部署`driver_user_fix_20261023.py`；原driver源码保留driver_before_user_fix_20261023.py，勿运行备份。PID看DRIVER_PID(最近3747920)。切换coordinator时未给old/free worker任何信号，16个old_eval job28402_0..15始终继续，当前约2200新raw/7072 missing。guard/resources原进程保持，预算CPU总18含1管理，GPU≤2实际CPU。local_cpu_guard仍识别driver.py入口，管理优先空闲CPU23。

新增能力门49152（static+moving各24576）已官方注册/preflight0reuse，队列QUEUES/capability_user_fix_20261023封存。先运行5min单shard28418，USER_FIX_PRIME_DONE及postguard通过，原始raw保留；现单shard28419 job名lit_supp_fix_aux占余下1份额，与old_eval16并行共17science+1management=18。6:57约11656新raw均success，尚是成本排序优先的search/gradient子集，不能代表48全体。新aux从现有claim继续，old/free不受信号。`monitor_aux_user_fix`退出后立即stat postguard。coordinator到regular revised capability前调用`takeover_aux_user_fix`只USR1这个辅助新能力门worker，等其保存当前局退出、stat检查、重开已停止owner的partial claims；未知raw不重跑。

调度顺序：继续old_eval→free_main(11328原UID/原worker，首次main)→辅助cap安全接管→capability_user_fix_20261023(已跑的raw直接保留，其余17worker补完)→生成QUALIFICATION_PRIMARY_20261023与SCREEN_SELECTION_PRIMARY_20261023（只从原资格表复用free；两新场景读修订cap）→swap_main(三场景PRIMARY，free_main精确复用，新两版新UID)→new_fixed_train/new_pair_train/new_eval→moving_eval(用free赢家)→finish_supplement_user_fix_20261023。后者用修订statistics/finalize，回填保存轨迹诊断、5000 paired bootstrap、官方DB postflight和结束审计后才COMPLETE。

原QUALIFICATION/SCREEN_SELECTION/CAPABILITY_RESULTS都保留，不能误作当前两修订版资格；CURRENT参数看PARAMETERS_USER_FIX和PREREGISTRATION_USER_FIX。原invalidated sidecar明确capability旧两scene作废，free保持；primary CASES由修订factory生成，原两版本不入主分析，postflight仍审计它们被保留。finalizer核验原资格与三state SHA不变、新/旧receipt、SNAPSHOTS SHA；原moving4448碰撞只作历史问题，不代表修订版correction效果。

原ITERATION_LOG已记录“这是用户指出说明设计错误后的修正，不是看结果调参”。没有stop标志/DRIVER_ERROR，首批revision无unknown/碰撞；后续若失败读DRIVER_USER_FIX.log及driver_actions/finish_actions，不重复启动coordinator，不覆盖raw/source/receipt。原第一部分H COMPLETE保持只读。以下均为历史上下文。

# 第二部分接手

当前任务：用户要求完整执行docs/prompt_baseline_literature_scenes_resume_and_supplement_20261023.md，先H COMPLETE后S；H已2026-10-22T20:46UTC完成，67712请求0碰撞，unknown1488(含门口5)，保留接受的34440旧查询缓存例外及原失败审计。H现在只读。当前S本目录已开始（2026-10-23T03:46UTC）。

目录及代码：所有新增仅S；官方journal默认shared_rollout_db/journals/exp_baseline_literature_scenes_supplement_eta_swap8_v1，官方single merger。不得改旧源/venv/git。原clone HJB-GNN在external/hjbgnn，仅参数只读，不导入/运行/安装。没有subagent授权，不派代理。CPU新加坡23日09:00后18总份额(17worker+1管理)，GPU≤2，实际CPU；全机24CPU，另有shaowen6CPU和1pending，本批16worker，充分尊重他人。

持续后台入口：driver.py，DRIVER_PID.json及DRIVER_LAUNCH.json；独占DRIVER.lock。资源resources.py + RESOURCE_GUARD_PID.json。不要重复启动。driver按阶段执行：capability→qualify→old_fixed_train→old_pair_train→old_eval→swap_main→new_fixed_train→new_pair_train→new_eval→moving_eval→COMPUTATION_READY→finish_supplement.py。后者backfill保存轨迹诊断→两项5000配对统计→guard→DB postflight→finalize_supplement后才COMPLETE。任何DRIVER_ERROR必须读具体driver_actions/finish_actions日志修复；未知raw不重跑、已完成raw不可覆盖。恢复只重开未完成group claims，限制8轮。

初始科学门：INTEGRITY_START1540个源/配置文件；BACKEND_BUILD记录静态复制图，重建native_resume/cbs_kernel_resume.so仅改缓存根目录为S/runtime_cbs_query_cache；旧4场景每8条SI baseline共32条完整逐位回放全部PASSED，strace crowdCBS全部持久写入仅S；WRITE_PATH_AUDIT及READY已封存。注意一次程序失序：批28322在慢stat返回前提前提交，已停止保留29回放，完整stat后28338只补3条；审计和prereg都照实披露，没有发现新只读写入。慢audit_supplement.check两个进程已停止，正确优化版guard_fast.py每批前后stat约936万项，包含H/v1/v2，已通过。并行任务目录仅stat，新增prompt仅metadata；新越界整体停。

科学文件已冻结：IMPLEMENTATION_FROZEN.receipt.json包括swap_geometry/safety/runtime/diagnostics/experiment/registry/case_factory/worker和静态backend及3活动.so。不要改已冻结文件或收据。辅助调度、分析、新文件尚可修，若科学bug必须新函数/文件和合法新收据/身份，不能篡改封存raw。case_factory.py新状态配置及RNG为scene1/8；moving机器人精确复用free，机器人RNGfree/moving按父identity匹配，障碍seed2026102313，与controller/seed无关，solo保留完整父障碍轨迹。

新场景：32state/scene；free R1，static R1.4+prompt预定4L，moving free+半径.3速度.25圆盘。r=.05,dt=.05,vmax=.5(声明假设：官方三维代码只有机体分量限.3及高层缩放2.0，无二维范数界)，goal_tol=.08；时限16s/22.4s。官方论文p8图7实景I形物体无标尺，420dpi查看无法可靠米制恢复，使用允许4L后备。corner不在直径不代表整个臂不穿直线，GEOMETRY_VALIDATION已明确部分臂穿过直线，但96states全部单机器人静态连通与容量通过，不宣称联合可解。

安全SI：局部距离.5以内pair半责、静态/动态全责；移动按当前位置+.0125膨胀，物理真实半径、步内转折piecewise扫掠。正常v2求解/同问题重试；无解先仅速度球半径slack，仍无解安全minimax，速度输出不剪裁，逐步记录违反/速度超限；数值失败unknown。修正Brel还是v2全部其他机器人bound后平均+Rc=.125125门，广播假设披露，名义不见移动障碍。8类48参数不改，私有controllers两份仅新增静态墙cost几何branch，旧4场景逐位证明。

能力门：73728逻辑请求=3×48×32×8agent×2seed，官方prefight0复用。单机器人成功下界≥.95合格。确定性gradient_mpc/PA/CBS seed1通过官方alias保存零物理步，有独立UID。每个parent agent都在group key中，避免串group。所有case/group/receipt已封存。大capability_CASES/MISSING各872MB，不随意重复parse；worker只读individual group files，3GB每worker。非随机cache是纯函数，仅S。

历史：HISTORY_EXECUTION_INDEX27477兼容键，HISTORY_TEMPLATES5120，旧baseline+修订fixed+修订learninglabels9份case清单来源已SHA；旧SI数据方法名不影响复用，保留旧rng_state_uid。case_factory.make优先复用原UID和raw，未知不再跑；新η完全不同才新UID。两类旧semantics不兼容要区分。

协议缺口：circle的v2训练MPPI0/1都不合格，严格训练池只有7/8类；已在HISTORY_CHECK/PREREG向用户披露，不擅自用评估controller补位。新场景也只在v2训练池16个controller中取首个合格/类型，缺类单列；照规则继续部分筛选，最终GAPS不能伪称8类完成。fixed16→top3全部6有序pairs，theta.968/6/7；训练first16、seed0，评估oldsplit/newlast16/allqualified/seed0；bestfixed/pair先FROZEN后preflight；moving不筛，转移free两赢家；比较同statectlseed的eta13与human。

分析：statistics_supplement.py复制原stats_kernel 5000次bootstrap seed2026102003，类型→实例→state等权，seed配对；eta单seed；不重新选。分类用v2最终classify即4a′+blocked_by_arrived majority，死锁证据只4a′。swap_diagnostics用保存第一投影实际effective rows，static/moving4a″分开，canonicaljoint按原coupled sum，实际乘积锥按effective rows。只有新main需要full9griddiag，capability不必诊断；backfill从trace补，不调controller。postflight核查所有请求DB identity/outcome/rawcanonicalSHA/traceSHA和pendingjournals。

当前运行进度看PROGRESS.json、squeue、science_<job>_<shard>.log。2026-10-23T04:13UTC约2.9万/7.37万capability raw；前3000抽查unknown0，moving真实collision、static timeout按规则保留。要持续监控，不能只看作业不存在就称科学完成。最终COMPLETE当前不存在。不要提前创建。

## 2026-10-23T05:25UTC继续进度

能力门73728已完成合库，unknown0；Q free43/static12/moving0，moving最高solo成功率.859375、碰撞4448/24576。CAPABILITY_RESULTS.md完整表，GAPS必须披露moving修正及转移评估不可运行。旧固定训练5888逻辑请求，734精确复用+2旧unknown保留+5152新补；job28370_0..15运行，当前约1500新raw，主要circle PA/CBS每局数十至300秒，存在同一state多η串行长尾。当前队列不改；queue_supplement.py今后新建队列group key增加requested_action，seed0/1仍同组，纯调度不改变case/物理/RNG。

修复未冻结的evidence_store.sources_index：原来glob *_CASES会误把REPLAY_CASES当case并KeyError，现在只索引*/raw/roll_*.json；5条实际不同终止结果跨阶段查找验证通过，EXACT_REUSE_LOOKUP_VALIDATION。postflight只遍历具有REGISTRATION收据的case清单，排除MISSING/回放。冻结科学文件所有SHA核验仍通过。

新增SAVED_DIAGNOSTIC_VALIDATION：从capability旧保存trace验证5种scene/outcome，九格diagnostic通过，无rollout/controller调用；CAPABILITY_DB_SPOTCHECK验证5条raw与官方DBcanonicalhash/identity/outcome一致。运行中midstat约936万项通过，无新只读写入。

statistics_supplement/finalize辅助尚未封存可修：补充原weighted_rates数学内核输出主线合格实例类型等权诊断率；moving未合格实例的4a只报告原始候选，不作为通过能力门支持的死锁证据。精确规模补充PREREGISTRATION_REQUESTS_ADDENDUM_20261023.md：新main19776，new fixed2304，new pairs864，moving_eval0，均须各自官方preflight。所有下一阶段仍由现有driver PID3119938推进，勿重复启动。资源guard3119937、本地CPU guard3166717，当前management固定空闲CPU23，worker16，另人6运行+1pending，合计遵守18用户额度及全机24。
