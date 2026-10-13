# Stage 2：覆盖×表示（草案；pilot未通过前不执行、不盖阶段收据）

识别controller→η regime映射在增加机制覆盖与改变表示后的留对迁移；Q回归仅辅助。沿用完整round5 prompt；阶段正式冻结需Stage1至少3/4内部pair过门，另有不可改SHA收据及protocol。

96个旧round4 S_train（DEVELOPMENT）共享12个blend controller；原前96state平衡8-head mask（每η48state）Q4。复用Stage1精确重叠2304格；14原Flow128state标签全部复用。32全新自然teststate由原generator序列2026170000..31产生，全库及历史协议content/UID去重。全12controller K16Q4，0–1discovery/门、2–3终点。probe16state序列2026180000..15且训练/测试/历史去重。controller参数λ固定，不看pilot标签调整网格/锚点/state。

请求闭环61440，已有pilot重叠2304，新59136；R5全26controller×16state×2η=832短episode，总Stage2新增≤59968，cap65000。96state不减配，除非精确计划实际超过cap，唯一后备train64；CPU时间成本本身不等于rollout数量上限。实测成本/申请时限另记。数值失败unknown=-1，不重跑；未请求=-2，不编码失败。DB preflight/journal/登记/单merge/postflight。

LOPO4fold：hold pair编号0/1/2/3的3lambdacontroller；DENSE14原Flow+9其余pairblend，SPARSE14原Flow。次要leave-one-A-anchor-out2fold各hold6blend；原14Flow仍保留，所以该次要试验只留出包含此A的融合实例，并非真实未见锚点checkpoint。训练预算每task3fitseed[17,23,41]、AdamW lr.0008 wd.0001 clip5、12000steps、batch128、两个64-hidden silu层、16logit heads；与round4优化/Net完全一致。各表示同宽度pad513，physical23后总536输入；维度差不增加不同网络参数预算。只计已知labelhead损失。训练controller等权采样，归一化只用源训练feature（无目标输入）。源验证NLL使用14原Flow已有P1的24stateK16Q4；融合源没有P1标签，不新增验证outcome；这意味着验证只覆盖原Flow，作为限制明确报告。3fit共用平均验证NLL选step，与round4一致。

R0=zeros；R3=256dim compact；R4_action=64dim投影动作+Jacobian（原代码、eps.001、t0及位置+0.1goal方向probe，env.step封锁）；R_oracle=[lambda,R3_A(s),R3_B(s)]513dim，原Flow视为selfblend[0,R3_C,R3_C]，是有已知融合组成的特权表示，不是任意真实controller可部署表示。R5=16固定probe state×[η10,η13]×5统计量展平160dim，每controller常数，不含标签、controllerID、probeID或seed字段。静态query与R5分别记录部署成本。

R5每probe用原seed0，200步原单积分器动力学与安全controller；为取得统一200步窗口，probe env仅关闭success/collision早终止，不读取这些flag、reward、done或summary，不改变动作/动力学/安全几何。5统计定义：(初末总目标距离差)/初始总距离；至少一次名义参考或pseudo-field投影改变目标速度范数>1e−6的步数比例；200步中agent中心pair最小距离；全agent/step平均速度m/s；全agent平均速度<.01m/s的step数。出错保留partialtrace+unknown，不填0；任何缺失R5输入不得静默删目标/选择有利probe，需报告真实技术边界。无success标签的probe单独study/index/journal，不写作H700success标签；复用兼容trace先查历史。

源prior在每coverage×fold内对每controller的训练state有效Q均值，再controller等权均值选argmax（ties最小index）。最近源根据source-only zscore R3在共享前96trainstates的controller均值欧氏距离，选该源训练最佳固定η。门：目标32teststate seed0选fixedη、seed1评；再反向，L为target双向交叉拟合fixed−sourceprior，unknown保守配对相同η取消；state bootstrap5000，L≥.05、95%CI下界>0。每coverage分别门，少于4必要controller时该臂主判INCONCLUSIVE；不得换门/候选来增数量。只seed2–3终点。所有拟合/feature程序禁止打开testoutcomes。

对每表示分别同时比较[源prior、最近源fixed、R0、错配平均]，3fitseed先平均；全部必要controller等权，并以共享32物理state为cluster同步bootstrap5000。max-T取绝对标准化bootstrap均值偏差95%分位，给同时CI；4下界都>0才判该表示可靠收益，按所列四种非R0表示分别做内部family（跨表示探索不宣称全研究错误率已控制）。错配每target按同state取全部该臂源controller表示，身份不可输入；不是换物理state，原Floworacle按selfblend约定；报告每错配控制器以及等权平均。

DENSE−SPARSE对同一target/sharedphysicalstate配对，用两臂必要controller的union作为主覆盖效应，intersection和所有12target作为描述性敏感性；union只由独立discovery seed定义，在同一个固定目标集合上比较两臂，避免比较不同门选择的群体。多个表示覆盖效应用max-T校正。只有DENSE过联合收益门、SPARSE不通过且直接覆盖差CI>0，才强支持覆盖改善；仅显著/不显著差异不足以证明因果。额外融合源同时增加总label数量，coverage与labelvolume并非完全独立，需限定根因解释。

k=1目标state校准取S_train index0固定、不看testlabel；8ηQ4共32fullrollouts（已有标签）+未标η保留sourceprior，比较R5的32个200step无标签probe成本。另报告k2/4，critic logitoffset仅由这些trainstates已知head估计。无目标标签的controller-prior通过目标controller在96个旧train初态的预测均值选固定η，与逐state选择单独比较；此为额外描述性分解，不替代主门。

SPARSE各fold源集合、source特征/label、验证及fitseed完全相同：每表示仅优化一次3fits，其余fold直接复用这些冻结checkpoint并单独推理留出目标；不得为了不同fold名称重复训练相同源问题。DENSE LOPO/LOAO源集合不同，独立优化。LOPO主4fold+LOAO次2fold共60个预测task、180个per-fold checkpoint引用，最多105个独立fit（DENSE90+SPARSE15）。记录复用hash与source签名。

R5若有数值失败导致任何200step统计不可得，保留完整partial trace并记unknown，该表示主结果INCONCLUSIVE/NOT EXECUTED（输入不完整），不删probe/controller、不填0、不用新seed重跑；其他完整表示继续。共享DB success字段非空，不能将无标签probe填为failure；probe的独立UID/manifest先官方preflight，worker写raw/journal，独立单进程将journal登记为UNLABELED_SHORT_PROBE source_file证据，原H700 rollout标签表不写伪标签/不改schema。probe postflight检验832个UID与无终局label字段。

辅助Q MSE在已知testlabel的percell比例，注明Q2噪声；controller×η predictedpattern相关只辅助。regret报告目标discovery fixed参照与same-evaluation state hindsight upper，后者不当可部署selector/连续ηoracle。

用户结论映射保留：DENSE R3/R4/R5有可靠收益、SPARSE无→覆盖不足；DENSE privileged oracle成功而static失败→表示不足；若R5成功→闭环behavior proxy较static局部表征有效；DENSEoracle也不通过→本方法/协议未学会controller→regime映射（方法缺陷候选）。最后一项不把未显著等同真实收益为零或普遍不可学，须结合CI/优化/样本量边界。若oracle已选对target固定regime、只是不能显著超过同样选对的最近源baseline，则“联合selection门失败”和“controller→regime映射学不会”逻辑不同，必须单独说明这个benchmark ceiling，不将前者冒充后者。必要controller<4→INCONCLUSIVE优先于根因归类。probe相关不证明因果；合成controller上的迁移不自动等价真实checkpoint上的迁移。

CPU最多24/18、GPU8/4按资源截止授权；每个GPUjob1CPU计总CPU预算。纯Q无视频，不commit，仅新目录。
