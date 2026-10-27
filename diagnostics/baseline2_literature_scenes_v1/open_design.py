from experiment import *
import csv
import numpy as np

def main():
    assert not (HERE/'PREREGISTRATION.receipt.json').exists()
    rows=[]
    def target(scene,method,metric,values,ns=None):
        for n,v in zip(ns or [None]*len(values),values):rows.append(dict(scene=NAMES.get(scene,scene),scene_code=scene,method=method,N=n,metric=metric,value=v))
    ns=[2,4,6,8,10,12,14]
    for m,v in [('无限时域模型预测控制与死锁解除',[100]*7),('缓冲Voronoi单元与右手规则',[100,100,97,94,83,65,41]),('增量序列凸规划',[100,99,94,81,58,39,12]),('分布式模型预测控制',[100,99,98,91,95,73,63])]:target('lit_crowded_random',m,'success_percent',v,ns)
    for m,v in [('无限时域模型预测控制与死锁解除',[0]*7),('缓冲Voronoi单元与右手规则',[0,0,1,0,0,2,0]),('增量序列凸规划',[0,1,6,19,20,61,88]),('分布式模型预测控制',[0,1,2,9,5,27,37])]:target('lit_crowded_random',m,'infeasible_percent',v,ns)
    for m,v in [('无限时域模型预测控制与死锁解除',[1.98,2.28,2.72,3.16,4.31,5.06,6.20]),('缓冲Voronoi单元与右手规则',[2.01,2.75,3.26,4.51,6.21,7.97,10.85])]:target('lit_crowded_random',m,'mean_completion_seconds',v,ns)
    for method,vals in [('集中式',[7.45,22.15,12.98,-.002,0,0]),('去中心化全责',[7.55,67.20,17.44,-2.84,3,27]),('去中心化互惠',[7.55,84.75,17.26,-1.53,4,32]),('协作避碰调度',[7.60,31.25,14.63,-1.35,4,0]),('伪协作避碰与单步延迟',[7.35,23.75,12.76,-.015,0,0]),('伪协作避碰与0.2秒低通',[7.35,21.65,12.68,-.067,0,0])]:
        for metric,value in zip(['min_seconds','max_seconds','mean_seconds','h_min','gridlocks','infeasible_trials'],vals):target('lit_circle_five',method,metric,[value],[5])
    for scene in ('lit_doorway','文献交叉口场景'):
        for method,value in [('模型预测控制与控制障碍函数',100),('论文活性方法',0)]:target(scene,method,'deadlock_percent',[value],[2])
    with (HERE/'PAPER_TARGETS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    inv=read(ROOT/'diagnostics/axis3_cross_scene_v1/HISTORICAL_INVENTORY.json')
    write(HERE/'FIXED_ETA.json',inv['eta'])
    # Independent frozen finite candidate pool, extracted before any labels.
    # All original cross-type-six proposals are inventoried, never retrained.
    frozen=read(ROOT/'diagnostics/crosstype6_shared_correction_controller_information_v1/FROZEN_PIPELINE.json')
    write(HERE/'DESIGN.json',dict(utc=utc(),grade='DEVELOPMENT',state_seed=2026102001,menu_seed=2026102002,bootstrap_seed=2026102003,
          controller_instances=48,state_counts=dict(lit_narrow_passage=64,lit_crowded_random=700,lit_circle_five=100,lit_doorway=64),
          paper_rollouts=dict(narrow=2,crowded=1400,circle_primary=300,doorway=2,intersection_secondary=2,circle_secondary=300),
          planned_flow_upper_bound=dict(solo_capability=616320,safety=89088,fixed_corrections=267264,nominal_background=1856,
                                     scene_learning_labels=24576,scene_learning_evaluation=12288),
          seeds=[0,1],training_seeds=[17,23,41],thresholds=[.90,.93,.95,.968,.98],
          capability_rate=.95,scene_informative_below=.98,stationarity_seconds=[2.5,5,10],cone_tolerances=[.02,.05,.1],bootstrap=5000,
          narrow_start=[[.25,0],[0,.25],[0,-.25]],narrow_goal=[[-1,0],[0,.25],[0,-.25]],
          narrow_digitization_error_m=.025,narrow_first_frame_time=1.8,narrow_real_initial_unobserved=True,
          circle_sampler='Uniform area in center disk radius 11; reject same-set pair separation <=4; no cross-set rejection; zero initial velocity',
          circle_outer_soft_penalty=1e4,circle_soft_penalty_status='paper unspecified; preregistered assumption',
          circle_infeasible='Minimize maximum nonnegative pair-row violation then project LQR onto that minimally relaxed feasible set; outer-circle slack separate',
          circle_integration='exact held-acceleration double integrator',
          training_split='narrow/doorway 1..32; evaluation 0 and 33..63. Crowded train N8/N14 0..15 each; eval16..31 each. Circle train0..31 eval32..63',
          frozen_zero_shot='Not applicable: frozen conditioning explicitly restores RingExchangeEnv and needs 100-D ring observation plus Flow, 33-D context; no defined observation for new geometries',
          resource_cpu_limit=18,resource_gpu_limit=2))
    (HERE/'PREREGISTRATION.md').write_text('''# 预注册：四个文献场景的修正流程迁移

全部为开发证据（DEVELOPMENT）。设计数值、种子、规模与分割以 DESIGN.json 为准。此文件与 DESIGN.json 在首条新 rollout 前一起封存。预先规定成功率比较、失败拆分、敏感性与5000次配对分簇 bootstrap；类型、controller、state 依次重抽，两个seed保持同簇，unknown给下界与上界。

论文版先执行：窄通道场景2局、二维拥挤随机场景1400局、圆形围场五机随机场景300局、门口场景2局；第二优先级文献交叉口2局，协作避碰调度及伪协作避碰300局。窄通道场景按400dpi图8第一帧读数，蓝机器人(.25,0)、挡路机器人(0,±.25)、穿行目标(-1,0)，读数误差±.025米。第一帧实际为1.8秒，真正t0未展示；本轮把该帧当声明的reset假设，不能声称精确原始实例复现。右手规则活锁必须见反复离开、返回相同平衡附近的轨迹，单次timeout不够。

二维拥挤随机场景使用官方GUI起点、终点两个采样函数原体（SAMPLER_PROVENANCE.json），坐标[0,2]²，不加论文未写的边界收缩。流程版随后按原半径膨胀墙检查并剔除无效state，不补抽改变样本分布。窄通道场景63个起点扰动±.05，门口场景63个起点扰动±.1，仅拒绝物理重叠。窄通道已到达者按现有流程到达停止，所有零初始目标距离的机器人进度定义为到达时1、离开目标容差时0；有移动任务者按原clip公式。此扩展必要，因为原ProgressSwitch对零初始距离有断言；不得删掉已到达机器人。

圆形围场五机随机场景按论文未明确的均匀分布约定采用圆盘面积均匀采样、起点和目标各自不重叠，不要求起点与目标跨集合互斥。初速度0，论文版采用精确保持加速度积分；外圆软障碍的松弛惩罚论文未写，固定为10000，不能据结果调整；QP无解按最小最大约束违反的词典序规则恢复，该细节亦为本轮假设。流程版速度取全部论文版名义LQR轨迹速度95分位，测定后追加单独封存收据，才允许流程版运行。外切正128边形最大向外偏差11(sec(pi/128)-1)。

门口场景论文版保留官方重复障碍、顺序更新、多步执行和活性分支；原main.py强制intersection及动画，使用新驱动调用原MPC类与其原方法，不改算法。流程版用两段紧致圆包围矩形[.9,1.1]×[.3,1.1]和[.9,1.1]×[-1,-.2]，明确矩形填充了相邻圆外缘的凹口。有限墙上下均可从无界平面绕行，绕行通道宽度无上界。时限按4*sqrt(9+.16)/.3，上取整控制步。机器人间目标恰为相切，必须报告终点可行性与到达容差。

流程版用单积分器、原控制周期/速度/半径/时限、原默认安全约束与原到达阈值.08。窄通道场景/门口场景各64、二维拥挤随机场景700、圆形围场五机随机场景100，合计928。48实例×928×2=89088条只加安全投影，三种固定修正267264条；全部单机器人能力门上限616320条，名义背景参照1856条。实际提交数量必须在逐state几何剔除、复现门、能力门及精确cache preflight后进一步封存，不能据未知结果改阈值。不得把计划数写为完成数。

修正动作η10、η13精确值来自FIXED_ETA.json。人定双η：首次平均进度≥.968或round(6H/7)动作前切换。冻结流程零样本的原状态编码没有定义于四个新场景，记不适用，不改模型。场景内重新学习每类型按清单前两个实例，32训练state×12动作×seed0，8类型四场景上限24576标签；48实例×32评估state×2seed四场景上限12288。动作种子固定2026102002，具体11个冻结候选在生成标签前另封菜单。输入更换不使任何同执行语义旧标签失效。critic架构、三训练seed与训练规则照跨类型六，仅替换状态为physical12，留一controller整类型，冻结选择早于评估首局。能力门不通过者不进入迁移主要结论，场景全部合格controller≥98%则只做固定修正。学习型行为克隆试点在48实例全部场景固定修正完成之后。

资源总量不超过18CPU及2GPU，每次提交前查squeue并记录。只读目录不修改；外部依赖仅本轮.venv-ext；绘图使用指定只读绘图环境放到数值封存之后。只在本轮目录及官方DB允许的登记/journal/merge处写。所有确切请求先写manifest、执行官方preflight再跑缺失seed。实现错误只重跑受影响部分并写ITERATION_LOG.md；数值unknown保留，不重试。复用已有执行代码必须先通过16条历史全轨迹逐位复现；不通过则该依赖分支记缺口。
''')
    seal('PREREGISTRATION.receipt.json',[HERE/n for n in ('PREREGISTRATION.md','DESIGN.json','FIXED_ETA.json','CONTROLLER_MANIFEST.json','PAPER_TARGETS.csv','SAMPLER_PROVENANCE.json')],before_any_rollout=True)
    (HERE/'ITERATION_LOG.md').write_text('# 实现迭代记录\n\n- 开工审计使用系统python失败（命令不存在），改用指定只读流程环境；没有rollout。首版审计遍历范围过深，终止后改为明确的源码层级；没有打开禁止前缀目录。\n')
    (HERE/'STATUS.md').write_text('# 当前状态\n\n'+utc()+'：已完整阅读prompt、共同协议与外部代码；历史检索、完整性开始快照、外部源码清单、论文目标、设计封存完成。当前阶段：论文版实现，尚无闭环结果。Slurm开工占用0CPU/0GPU。\n')
    print('preregistration sealed',sha(HERE/'PREREGISTRATION.md'))

if __name__=='__main__':main()
