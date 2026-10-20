from core import *
precision()
def main():
 assert (HERE/'INTEGRITY_START.json').exists();assert not (HERE/'PREREGISTRATION.receipt.json').exists()
 prompt=ROOT/'docs/prompt_round30g_student_line_mechanism_closeout_20261017.md'
 text='''# 分布式学生线机制收尾预注册（round30g）

本轮仅读取既有文件，不生成任何rollout、不训练、不查询controller或checkpoint、不写共享DB，不使用git写命令。全部结果为DEVELOPMENT，不能把相关诊断升级为因果机制。共享DB默认完全不用；若确有必要，只能sqlite3以file:...?mode=ro、uri=True打开，并记录只读查询。所有新增输出限本目录；已有源码不修改。历史教师、学生、零动作控制器的标签、轨迹、冻结critic选择全量复用。逐UID/初始物理字节/随机键别名和seed配对；状态UID的历史规范化差异不等于物理状态不同。只使用最终有效原学生：local教师索引8/10/12/13来自round30，其余来自round30b/c；不纳入round30坐标错误的global学生。每组14controller×32state×2seed；专家32条确定性轨迹，不把其复制为14个controller或两seed。

## 指标细则（在计算任何新结果前封存）

绕行方向沿用ring_exchange/expert.py的circulation_signature：角度逐步unwrap；累计角位移严格>0.15为逆时针、严格<−0.15为顺时针，其余未绕行。全队两种符号同时出现为不一致；只有一种符号为一致；全零为静止。整局及第100/200/300步均计算；早于landmark结束的已知局使用其终止前缀并单列早停数量。还报告真正四机器人全部同向的严格指标与已绕行机器人数，防止把只有一个机器人运动误读为全队协调。方向稳定时刻是事后定义：最终一致的局中，最早起保持同一全队同向标签直至已记录终点的步；这不是在线意图或因果承诺。

碰撞只使用已有碰撞标签。类型按native swept几何公式离线复算：中心距离减0.90与0.16，外环3.25减0.16与最大端点半径，机器人间相对线段距离减0.32；中心/外环<=0.005、机器人间<=0.0为接触。允许同一步多类型共存；另报类型互斥汇总（仅机器人间、仅中心、仅外环、多类型、几何无法复核），不改成功或事件标签。首次碰撞步以最早满足条件的已存线段确定；记录最接近处机器人中心及机器人间中点，明确不是额外物理仿真。

迎面相撞的主定义：在首次机器人间接触的执行步，两机器人执行速度均>=0.01m/s；各自朝向对方的单位相对位置余弦均>=0.5，且两速度夹角余弦<=−0.5。报告至少有一对迎面的碰撞局占机器人间碰撞局的比例，以及首次接触pair事件比例；再报告这些pair是否具有相反的累计绕行符号，区分迎面与模式冲突。

超时最后显著进展时刻：全队目标总距离D，从上一个显著改善锚点起累计下降>=0.05m即更新锚点和时刻（单调新低记录），否则不更新；初始为0。另报字面相邻步D(t−1)−D(t)>=0.05的最后时刻，避免阈值解释隐藏。未到达=最终距离>0.09m。相向且方向相反pair的主几何诊断：仅未到达机器人、累计绕行符号相反，按该符号构造当前位置切向单位向量，两向量朝向对方的相对位置余弦均>=0.5；补报真实执行速度相向pair（速度>=0.01且满足上述迎面余弦）。停住时没有可观测速度朝向，几何切向诊断不能证明controller意图。

停滞段沿用round28：每个post-step机器人速度<0.01m/s、未到达且连续>=40步，统计段数、首次开始、总机器人秒和最长段。仍运动但无进展：t>=40、速度>=0.01、未到达、其目标距离E(t−40)−E(t)<0.05m；统计满足的endpoint持续时间（机器人秒，每endpoint只计一次）、连续段、最长段，另报全队版本（D差<0.05、队均速>=0.01）。不是把重叠窗口各计2秒。

停滞位置与原碰撞位置：对同teacher/variant、state、seed，nominal碰撞而safety超时的局，只比较原碰撞涉及且安全层下最终仍未到达的机器人。位置为安全超时末40步中心均值与原碰撞最接近时的同机器人中心之距离；主“附近”阈值0.50m，另报0.25/1.0m敏感性、距离分位数和同原碰撞步的安全轨迹距离。不混用不同机器人或无关全队质心。未共享参与机器人时单列不可比。

到达分布：全队成功为原生标签，成功时间T*0.05；以全部已记录局为分母画成功累计发生率，碰撞是竞争事件，不把它当普通无信息删失。unknown保留下界/上界。逐机器人首次进入0.09m目标球另作描述，不能代替全队成功。

时间表切换：固定比较[400,500]与[500,600]步的D下降速度，主要报告真实观察到600步的局，并披露条件化带来的存活者偏差。另报全部已知局的吸收终止前向填充轨迹；成功后的D变化记为0只是画图约定，不是新rollout。使用同state/seed全程第13组修正参数作配对参照的前后差之差作为描述性敏感性；时间窗口固定，不按结果选择。早期进展用D(0)−D(t)、t=100/200/300，已知局早停吸收填充，unknown不外推。

统计：bootstrap seed=2026101707，5000次，先同教师身份controller有放回抽样，再为每个抽中的controller抽32个state簇；保留两seed、各臂和各学生变体配对。二元比例/连续指标以每局等权（每controller样本数相同），条件比例重抽分子分母而非仅对已选失败controller重抽。专家与零动作只有一个controller，controller层退化为单例，只抽state簇；零动作的两个确定性seed不当64个独立state。平均/差值均给95%分位CI；未知成功按0/1界、配对差用保守交叉界，unknown不算已知失败。不一致的成功/失败条件比例与相关系数在无分母/零方差时写undefined，不能填0。Pearson和Spearman相关在14教师的成功率间计算，Pearson另给controller bootstrap CI；按global/local分层，避免相关性仅由分组零成功驱动。

论文表各单元为成功率下/上界与CI、碰撞率、超时率、unknown、样本数、来源轮次和DEVELOPMENT。冻结critic采用历史已冻结通用critic（A3）在四臂F10/F13/S300/S500的选择；若某组没有兼容冻结选择，写缺，不重新训练/重新选择。事后最优定义每controller在四修正臂的总体成功率最大者，bootstrap内重新选，不能逐局挑成功者。集中式专家仅填原生规划参照，其余臂写缺。联合教师只跑臂不区分尚未启用的安全管线，可在集中/分布式两行复用同nominal记录，但明确不是额外样本；集中式safety和公平调参RHR没有同管线已证据时写缺。不用round28原参数右手规则冒充公平调参规则。

资源：CPU only，所有用户Slurm作业合计最多18 CPU shard/2 GPU shard。本轮最多8个CPU分析worker，提交前读取squeue，资源不足等待，不取消他人作业，不增加GPU。所有历史目录开始/结束完整hash审计，复制或只读导入源文件逐一记hash，x64=False在全部import完成后恢复。无视频。

---

完整任务原文：

'''
 (HERE/'PREREGISTRATION.md').write_text(text+prompt.read_text());seal('PREREGISTRATION.receipt.json',[HERE/'PREREGISTRATION.md',prompt,HERE/'INTEGRITY_START.json'],before_new_metrics=True,evidence_level='DEVELOPMENT',precision=precision());write(HERE/'DB_ACCESS.json',dict(opened=False,writes=0,policy='No DB needed; immutable experiment raw artifacts and saved canonical rows reused. If needed later only sqlite URI mode=ro.'));print('preregistered',utc(),flush=True)
if __name__=='__main__':main()
