# Round8 独立交接

任务COMPLETE，UTC 2026-10-13T06:19:48.044745+00:00，全部DEVELOPMENT；工作根目录/home/zhihan/research/Basin_C1；Python /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python。不依赖本对话。

结论：少样本适应可行：预设regime候选集R下，E-safe在128条目标标签首次达到SPT/CT；H和冻结critic没有建立额外预算节省。7真实LOCO、必要base_u/v10_local，R E-safe B128首次SPT/CT：E1 +7.25pp CI[+2.24,+12.61]；E2 +24.42pp CI[+6.91,+42.78]；E3 possible/certain与any-sample均0。必要目标分别+26.47/+22.38pp，仍距固定oracle约3.80/3.79pp，因此必要B_2pp均>128；五个非必要B_2pp=0。C-F2-safe同样B128，预算节省0，其收益较E-safe低3.91pp，同时CI[-6.71,-.97]。H在128内无SPT且部分设置灾难多，不能写H≈E。U/A E-safe亦>128；R预设候选菜单有描述性工程价值，与H/C的额外先验价值分开。

逐state价值：B>0没有C-Cctl成功率正同时CI；R B128 CF2safe-ctl点差-1.61pp、CI[-3.91,+.77]，INCONCLUSIVE。CF2safe过SPT而ctl仍有2次target×repeat灾难，是均值/rare risk区别，不作逐state普遍优越或无价值结论。

执行：先读AGENTS/round8及旧证据，在2026-10-13T05:55:06.921183+00:00冻结PREREGISTRATION，sha256 f5987985713acf18ada1d8f2d1ef712ceaad225b79ef0751df42b1904d61df1d；新分析在收据后。7主目标pool各5632格(TRAIN4096+P11536)，unknown70/72/74/85/26/69/15。P1按名称对齐；TRAIN/P1/test UID和physical两两无交集；endpoint只testseed4..7，oracle仅0..3。StageB7目标pool只TRAIN、13源、P1描述评价，C无合适冻结模型不适用。所有采集200重复，无放回，预算0/4/8/16/32/64/128是路径前缀；U/R/A、E/H/C/ctl/safe全部按预注册固定。A由H posterior驱动并给全部方法相同标签，E-A也带源采集帮助。

H实现为原source成功/失败count+Beta1的mixture共轭后验，没有自行选prior强度；source每head约256样本浓度是方法局限。C概率residual offset=sum(y-q)/(n+8)，unknown只占预算，safe3fits一致且优势>=.10。R0/F2 REAL6冻结，只补pool预测、复用旧endpoint logits，JAXx64false；无模型训练。ADAPTATION_FREEZE早于新终点读入。5000精确state×采集重复交叉bootstrap，每B统一max-T(主491效应)，预算decision bootstrap保留原CI宽度且右删失，不作跨B全局确认。

全部阶段完成，新增rollout/probe0，DB连接/写入0，只写本目录，不commit，纯Q无视频。共同18CPU/3GPU，GPU推理1shard，CPU采集/统计最多14job。FINAL_AUDIT PASS，1588历史输入不变；unknown/无放回/源排除目标/H公式/offset/零预算旧数值/交叉bootstrap独立检查均过。

下一步优先预设少数候选+目标标签安全适应的前瞻确认。4新controller：目标采集512episode+独立必要性384+终点16384=17280完整H700episode，本轮未执行。两个新η13源按旧TRAIN/P1补数据需要11264完整+64短probe，Flow训练另计，额外先验预算收益尚未验证。R5每C6400步约9.14H700上限请求或9.25–10.65条已记录TRAINepisode步数，缺成功标签，信息不等价。

主要入口results.json、REPORT.md、SAMPLE_EFFICIENCY.json、POOL_AUDIT.json、FINAL_AUDIT.json；逐采集/后验/offset/choices在simulations/；模型pool预测在predictions/；endpoints/保存所有B主/StageB结果与bootstrap；figures/和tables/完整曲线。REPORT末尾有复算命令。继续前读预注册，绝不生成新rollout、连接DB写入或用testseed4..7改策略。边界Ring/FF/H700/K16固定几何、真实必要目标仅2、DEVELOPMENT。
