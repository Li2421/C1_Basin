# Round9 独立交接

全部DEVELOPMENT。研究问题：Ring下controller依赖的η10/13效应在哪个阶段产生。先读REPORT.md（H_late=INCONCLUSIVE，H_early=SUPPORTED，H_state_mediated=REJECTED），再读results.json。用户任务冻结全文在PREREGISTRATION.md，收据SHA已校验。

实验目录：/home/zhihan/research/Basin_C1/diagnostics/controller_eta_switch_time_v1。Python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python。共享DB实验UID=exp_controller_eta_switch_time_v1。4 Flow×64原loco_test state×seed0..3×双向×5时刻=10240，另32一致性门；H700、FF、JAX x64false，唯一干预action index>=t替换η。原baseline来自round4，不补跑；unknown不是失败、不能重跑。

代码：core.py/run.py/protocol.json被冻结，勿修改；launch.py只转换journal数组。design.py登记/官方preflight；submit.py统计zhihan自己的CPU任务；merge.py调用官方单进程merger并精确保留现代seed_key；postflight.py核对raw/DB/trace；analysis.py统计/机制，classifier.py固定LOO，report.py报告。finalize.py串行恢复journal→merge→postflight→analysis→classifier→report→COMPLETE，失败留FINALIZE_ERROR，不自动补跑unknown。

证据：main/raw与main/traces，gate/raw与gate/traces；STAGE0_SUMMARY.md和HISTORICAL_RAW_INDEX.json；CONSISTENCY_GATE.json；两phase CACHE_PREFLIGHT/POSTFLIGHT；SLURM_SUBMISSIONS.json、RESOURCE_CORRECTION.json、merge_audits、FINAL_AUDIT.json。CLASSIFIER_INPUT/OOF.npz和FOLDS、CURVE_RESULTS、MECHANISM_ROWS/RESULTS完整可重算。所有统计只读真实证据。

复算（不rollout）：先核对COMPLETE与POSTFLIGHT及预注册hash，运行analysis.py、classifier.py、report.py；会重写派生报告，保留输入immutable。没有新几何/未知controller认证，无正式中介证明。下一轮是否采用后期state-only critic需依本轮门结果另行预注册。本轮无git commit、不删资产；DB只登记本实验新身份/rollout并更新本实验metadata。
