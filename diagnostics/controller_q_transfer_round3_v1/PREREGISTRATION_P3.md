# P3 条件性失败定位预注册

只在 P2′ 的主要 held-controller selection 端点失败或因固定η饱和而不确定时执行；不新增rollout。若P1通过但P2因70,000上限停止，本阶段不使用P1开发state冒充独立P2确认。

同一冻结P2′数据和源controller划分。诊断一：Q误差分解为controller平均水平、η均值曲线以及state×controller×η残差，使用held控制器的完整K16 Q8标签；分解只作为定位，不能代替成功率。诊断二：对同一16-head网络、3fit、同source数据和相同验证NLL规则，比R4的仅t0前32维、原R4_action的t0+预定目标向probe64维，以及**执行η0首动作后的单步open-loop预估state**描述。预估只使用当前state的静态动作和已知dt: `p′=p+dt*a0`, `v′=a0`；不调用env.step、不使用目标outcome、未来轨迹或成功标签，再以固定query key计算η0动作/Jacobian。完整原始动作与新probe合成64维，与R4_action输入宽度保持一致。比较未见TF/FT的top1成功率、相对固定η的gain、matched−mismatched、Q分项误差；均标为辅助诊断。

诊断三：复用P2′已预注册的source-controller数量3 vs5对照，不重新按test调参。按64个held物理state配对bootstrap5000次；若差CI跨0，controller数量不足只作猜想。若选择端点零headroom，P3不能判定表示是否充分。任何新特征在读取测试标签后才实现的变化必须单独记录并降格为探索性。
