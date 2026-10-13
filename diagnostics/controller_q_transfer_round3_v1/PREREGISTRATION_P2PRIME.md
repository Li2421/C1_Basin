# P2′ 条件性 placement/κ 留出 controller 预注册

本文件在 P1 完整 outcome、P2′ 新 outcome 之前登记，仅当 P1 预注册资格门未通过时生效。所有新 state 必须用原生 Ring `sample_instance(seed,'development',LocalFrameConfig)` 的固定新seed序列生成；训练 `2026110000..2026110255`、验证 `2026120000..2026120031`、留出测试 `2026130000..2026130063`。三集合与历史512/256 state 的 UID 必须不相交。不得依结局筛 state。固定 geometry、native v10 Flow checkpoint、H700、dt0.05、原FF安全层/OrthoFlow3 η基底、同seed方法和成功判据。

源controller：原生 FF、原生 TT、endpoint-matched κ∈{−1,0,+1}，共5个；留出 controller：TF 和 FT。TT、κ沿用历史精确执行 UID；新 TF/FT 以独立wrapper、frozen factorial执行代码和全部源hash构造UID。不能把旧 conditional-Flow/旧评价标签混入。原生FF历史512train K16Q4作补充源，训练mini-batch按源controller均衡，使额外FF旧标签不支配模型。

新标签：5源 controller×256train state×每state8个η×Q4 =40960；5源×32val×K16×Q4=10240；2留出×64独立test×K16×Q8=16384；总 **67584 gross new requests**，低于70000的保守上限。每个train state的8个η由rng2026101207的permutation前8固定，五源共享同一子集；K16始终包含η13，在验证和测试中全覆盖。测试seed0–3选择目标固定参照，seed4–7独立评价。缺失精确标签才rollout，unknown保留且不得重跑；preflight核对gross与实际missing后再提交。所有source模型拟合/归一化/step选择只使用source训练与验证，测试标签永不反馈。

五个主表示：R0无C、R1源controller onehot（held全零）、R3 compact64、R4_action η0投影动作+投影Jacobian（t0和固定0.1目标向probe）、R3+R4_action。相同physical23 + 固定最大输入宽度 + 两层64 SiLU +16head，表示以零填充对齐、模型/训练步数相同；三fit seed17/23/41，AdamW0.0008、wd0.0001、clip5、batch128、最多12000步、step选择为source validation NLL（P0冻结）。同样架构和预算比较source-controller数量3（FF、TT、κ0）与5全体；无法在70000条上限内再增加到7源，不能伪称测试了7源效应。

主要终点：held TF/FT 的top1闭环成功率seed4–7，与(a)源训练合并最佳固定η、(b)按验证state静态R4距离最近源controller的训练最佳固定η、(c)R0 critic、(d)目标自身hindsight最佳固定η参照比较。目标seed0–3 best fixed只能作标注对照，不进入zero-shot模型或source prior。R3/R4错配用held TF/FT互换描述，state与η不换；matched−mismatched须按独立物理state bootstrap≥5000次且CI下界>0，并在至少一个held controller上对(a),(b),(c)的配对gain也全部CI>0，才称可靠跨controller选择收益。整体结论须同时报告两held controller、所有K16 fixed simultaneous CI、unknown上下界、多fit一致性。Q MSE、controller×η模式相关、source validation及训练controller数为次要诊断。

若held TF/FT固定η饱和而无可测选择空间，应标为selection端点INCONCLUSIVE；不能删η13或挑选有利test state。若完整preflight新数>70000，停止rollout并明确记录缺口。所有结论仅涉及同一Ring几何和v10 Flow上的placement/κ变化，不能冒充未见真实Flow checkpoint泛化。
