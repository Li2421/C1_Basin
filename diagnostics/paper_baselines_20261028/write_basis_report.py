"""Interim report generated only from completed, cache-indexed experiments."""
import json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
def fmt(v):return '—'if v is None else f'{v:.3f}'
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |'for r in rows])
def main():
    s=json.load(open(HERE/'basis/summary.json'));families=s['families'];sel=json.load(open(HERE/'basis/FINAL_FROZEN.json'))
    fixed=[]
    for name,m in s['references'].items():fixed.append([name,'3'if name.startswith('B3')else'0',str(m['success'])+'/16',fmt(m['success_mean_time']),fmt(m['truncated_mean_time']),m['collisions'],m['timeouts'],m['numerical_unknown']])
    for e in families:
        m=e['fixed'];fixed.append(['['+','.join(e['columns'])+']'+('（默认 B3）'if e is families[0]else''),e['dimension'],str(m['success'])+'/16',fmt(m['success_mean_time']),fmt(m['truncated_mean_time']),m['collisions'],m['timeouts'],m['numerical_unknown']])
    oracle=[]
    for e in families:
        m=e['oracle'][-1];oracle.append([','.join(e['columns'])]+[r['success']for r in e['oracle']]+[fmt(m['success_mean_time']),fmt(m['truncated_mean_time']),e['union_with_B3_at32']['success']])
    pair=[]
    for e in families[1:]:
        m=e['vs_historical_strong_B3'];pair.append([','.join(e['columns']),m['paired_success_n'],fmt(m['paired_method_time']),fmt(m['paired_B3_time']),str(m['rescued'])+'/'+str(m['lost']),str([round(x,3)for x in m['physical_block_rate_difference_percentile95']])])
    scene=[]
    for sc in families[0]['by_scene']:
        scene.append([sc]+[e['by_scene'][sc]['success']for e in families])
    text=f'''# C1：B 统一候选库压缩与论文基线接入（阶段报告）

分支 `research/paper-baselines-20261028`，基于 `daacc10`；旧 B3、原 nominal、安全层、动力学和成功判据未改。**论文基线仍在接口验证，本文完成部分是 B 的独立比较；不能视为全部五篇论文已复现完成。**

## 当前结论

B 是从 G/P/R、T1–T5、P2、R2/R3、Y1–Y4、S1、C1/C2 中选取的列矩阵，不是另一个方向类别。未经论文对应的 RHR-A/B、persistent/tuned RHR、Merry heuristic 和 partial-MGR 已从本轮正式方法注册和比较中移除。原始历史证据保留，研究候选列没有因此删除。

开发集统一比较60个固定配置，各22状态，最优为二维 `[G,C2]`，20/22成功。新独立集上该固定配置只有10/16，低于历史冻结强B3的11/16；四维 `[G,R,Y3,C2]` 达到12/16，但它是本次测试中的观察结果，不能据此重新冻结为已验证优胜者。当前**没有稳定固定策略优越性的证据**。

纯族 Oracle@32：`[T5]`、`[G,C2]`、五维组合都16/16，B3为15/16。说明这批初态的有限池覆盖不需要五列；并不证明一维T5可以恢复全部真实死锁。历史失败轨迹恢复试验中T5在6个十字口停滞状态为0/6，而C2为6/6，见 [前轮机制报告](../basis_discovery_20261027/report.md)。必须区分提前预防和停滞后的脱困。

## 证据复用和预注册

检索并复用 `../basis_discovery_20261027/` 的字典实现、逐候选记录、轨迹、静态审计、冻结设计和回归结果，以及 `../b5_oracle_20261027/INDEPENDENT_FROZEN.json` 的强B3。没有训练，未用GPU。原执行源 `shared_control/basis_dictionary.py` 和旧 `runner.py` 未修改，因此旧 controller UID 可精确复用。

开发完整交叉比较1320条请求，精确复用888条，补跑432条。缺口是这些已存在配置在同一22状态上的完整覆盖，不能用不同分母的历史单列成绩直接比较。候选按每个实际非零支持维度1–5最多12个、每支持集最多2个配置选出，选择只看开发seed0。

新测试预定states22–25，但源场景实际只有24个状态（索引0–23）。在注册和运行前即发现越界，记录修正为未使用的states22/23、future seed5；未生成或挑选替代有利状态。四场景×2初态×2原nominal，共16例、8个物理状态块。32候选×6族=3072新rollout，完整执行不按成功提前停止；墙碰撞计失败。随后补充旧冻结强B3及仅安全过滤32条，参数来自更早冻结文件，无测试集重拟合。

每批均先保存完整 `*_planned.json` 并执行数据库preflight后提交CPU Slurm。原始 `basis/candidates.jsonl`、`basis/traces/`、各manifest和cache日志保存在本目录；数据库单进程合并。大轨迹没有声称已由Git备份。

## 固定参数：全部方法

时间单位秒。成功均值只在成功样本计算；截断均值给所有未成功实例计原场景horizon，碰撞提前终止不能借此得到更短完成时间。timeout不标为deadlock。

{table(['配置','列数','成功','成功条件均值','全队列截断均值','碰撞','timeout','numerical unknown'],fixed)}

## 配对成功样本与不确定性

以下与历史冻结强B3在同一初态、nominal、future seed配对。最后一列以8个物理状态块进行20000次bootstrap，保留同初态下两nominal相关性；样本少，仅作探索性区间。全部数值、各组Wilson描述区间及计算开销在 [summary.json](basis/summary.json)。

{table(['B','双方成功数','B时间','强B3时间','补救/损失','成功率差95%区间'],pair)}

## 有限候选 Oracle 预算曲线

每族保持一个开发冻结固定配置作为首项，加31个固定Sobol点；G盒[.5,1.25]、P[-.5,.5]、R[0,.75]，新列沿已有signed/nonnegative定义。纯族保留其精确支持；联合回退池另报前16个B3加前16个对应族，总预算32。不能把联合池叫纯一维或纯二维。

Oracle成功时间为冻结顺序内第一个成功候选的时间，非最快成功时间；不是在线策略。候选全运行成本和所有失败计数另列CSV，不能只算Oracle最终选中候选的成本。

{table(['B','@1','@4','@8','@16','@32','@32成功时间','@32截断时间','B3联合@32'],oracle)}

[逐预算机器可读曲线](basis/budget_curves.csv)。Oracle未找到成功不等于连续族不可能成功。原B3候选作为联合候选池子集完整保留；等预算比较优先看B3@32与联合@32。

## 全部场景的固定策略结果

每格分母4。场景分别为四向十字口、拥挤方形、对径交换、障碍圆环交换；原Toy/Double-Bottleneck/Four-Way等旧checkpoint场景的本轮完整重跑尚未完成。

{table(['场景']+[','.join(e['columns'])for e in families],scene)}

两种nominal为原`centralized_sampling`和`gradient_mpc`。逐nominal数据见summary。B是原安全参考a之后的加性修正，移除G/P/R列不代表移除了nominal。字典、临时角色和T5 recipe规则沿前轮冻结实现。新补跑没有更换solver或修复已知墙体安全间距差异。

## 论文基线当前状态

正式允许文献只有用户指定五篇；Wang的Consistent Perturbation与完整Deadlock Resolution分开标注。DAC、MGR、Chen优先调用完整作者核心；Wang和ICRA为逐公式独立实现。全部适配差异、作者commit和验证状态将写入 `PAPER_REGISTRY.json`。

截至本阶段，DAC作者完整几何分类/CLF shaping已完成四类开发场景pilot；MGR完整环岛+障碍旋转+CBF代码已运行，斜墙刚体坐标适配正在回归；Chen完整优先级/右手惩罚/安全走廊已在无障碍场景跑通，旧OMPL依赖接入中。Wang三种分类分支、ICRA辅助CBF梯度与MGR旋转矩形几何已有数值校验，见 [mechanism_checks.json](mechanism_checks.json)。这些局部检查不等于整篇论文验证。

论文方法有自己的控制器/安全模块，不能在其内部偷偷套原B安全层。DAC原生SI；MGR为完整决策栈的SI接口移植；Chen执行原预测轨迹第一步位移；Wang将原DI加速度积分成原SI环境输入。共同物理环境、速度上限、碰撞/成功/horizon保持原值；移植不继承论文连续时间/原动力学保证。未完成或不适用项将独立标为unknown，不能伪装成普通timeout，也不计作论文方法性能定论。

## 可恢复命令

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$PY diagnostics/paper_baselines_20261028/basis_analysis.py
$PY diagnostics/paper_baselines_20261028/write_basis_report.py
# 新rollout先使用对应脚本 --plan；检查preflight后再通过Slurm提交。
```

本轮已验证的结论限于上述冻结候选池和样本。尚不能宣称最小连续Basis、五篇论文全面优越、跨所有旧checkpoint稳健，或安全层允许但全字典数学上不可达。
'''
    (HERE/'report.md').write_text(text)
    print(HERE/'report.md')
if __name__=='__main__':main()
