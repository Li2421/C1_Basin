from experiment import *

def main():
    rows=[]
    def add(scene,name,value,source,paper=None,flow=None,reason='相同'):
        rows.append([scene,name,str(value),source,str(value if paper is None else paper),str(value if flow is None else flow),reason])
    for scene in ('窄通道场景','二维拥挤随机场景'):
        for name,v in [('最大速度',1.),('最大加速度',1.5),('最小间距',.3),('机器人半径',.15),('警戒带宽',.1),('终端位置权重',30.),('排斥参数ρ0',2.),('参数η0',2.)]:
            source='Chen等，arXiv2202.06071v4，第五节A（第10页）'
            if name=='最大加速度':add(scene,name,v,source,flow='不适用，速度控制',reason='流程版单积分器，无加速度输入界')
            elif name=='最小间距':add(scene,name,v,source,paper='官方uav.py:55的r_min=.335（用于优化缓冲，不是物理直径）',reason='官方扩展缓冲硬编码.335；论文要求sqrt(.3²+h²*1²)')
            elif name=='终端位置权重':add(scene,name,v,source,paper='run.py:121–130，Sigma=15系数对应0.5*QK（QK=30）',flow='不适用，各controller原参数',reason='官方另含cost_index重加权与Delta_P平滑；不能只按15对30判断完整目标失配')
            elif name=='排斥参数ρ0':add(scene,name,v,source,paper='.2（avoid.py:101–106）',flow='不适用',reason='官方硬编码与论文数值不同，保持源码')
            elif name=='参数η0':add(scene,name,v,source,paper='eta初始1、每次+.3、上限4（uav.py:92、169–176）',flow='不适用',reason='官方自适应实现与论文符号对应不能断言等同')
            else:add(scene,name,v,source,flow='不适用' if name=='警戒带宽' else None,reason='流程版安全投影替代警戒带' if name=='警戒带宽' else '相同')
    add('窄通道场景','机器人数',3,'第五节A.2、图8')
    add('窄通道场景','穿行机器人reset','(.25,0) ±.025米','图读：第10页图8上排第一帧，400dpi，IMPC_page10_400dpi.png',reason='图中时刻1.8秒，不是明确t0；声明以此帧reset')
    add('窄通道场景','挡路机器人起点/目标','(0,±.25) ±.025米','图读：图8，400dpi')
    add('窄通道场景','穿行目标','(-1,0) ±.025米','图读：图8，400dpi')
    add('窄通道场景','墙','无','图8')
    add('窄通道场景','采样时间',.2,'第五节A')
    add('窄通道场景','预测时域',10,'第五节A',flow='不适用，controller各自原时域',reason='迁移controller结构不同')
    add('窄通道场景','时限','论文A.2未写','本轮取第五节B的50秒',paper=50,flow=50,reason='预注册假设')
    add('二维拥挤随机场景','工作区','[0,2]²米','第五节B.1、官方app.py:588、638',paper='仅采样区；官方优化无边界墙',flow='边界线段硬墙',reason='流程版按prompt新增硬边界；大量原采样state可能贴墙无效')
    add('二维拥挤随机场景','机器人数','2/4/6/8/10/12/14','第五节B.1、表II')
    add('二维拥挤随机场景','每数量试验数',100,'第五节B')
    add('二维拥挤随机场景','起点采样间距','r_min+.05=.35','官方app.py:573–613；SAMPLER_PROVENANCE.json')
    add('二维拥挤随机场景','终点采样间距','r_min+2*epsilon=.5','官方app.py:618–655；SAMPLER_PROVENANCE.json')
    add('二维拥挤随机场景','采样时间',.15,'第五节B.1')
    add('二维拥挤随机场景','预测时域',12,'第五节B.1',flow='controller原参数',reason='迁移controller结构不同')
    add('二维拥挤随机场景','时限',50,'第五节B',paper='驱动ceil=334步；报告严格50秒内结果时从既有轨迹截取前333步',flow='floor(50/.15)=333步=49.95秒',reason='流程版不执行超出50秒的控制周期；论文版不重复生成轨迹')
    for name,v in [('机器人数',5),('机器人半径',2.),('外圆半径',11.),('采样周期',.05),('LQR的Q','.2*I4'),('LQR的R','I2'),('CBF的l0',6),('CBF的l1',5),('协作避碰调度ρ',2),('时限',100),('目标距离阈值',.1),('速度阈值',.1)]:
        add('圆形围场五机随机场景',name,v,'Jankovic等，arXiv2207.04915，第五节、式10–16、表I',flow='不适用' if name.startswith(('LQR','CBF','协作')) else '.08' if name=='目标距离阈值' else '任务判据不要求速度，另报论文判据' if name=='速度阈值' else None,reason='流程动力学、安全层、到达事件不同' if name.startswith(('LQR','CBF','协作','目标','速度')) else '相同')
    add('圆形围场五机随机场景','采样分布','论文未写概率密度','第五节：拒绝机器人彼此及外圆物理重叠',paper='面积均匀半径11提案、拒绝中心半径>9及同集合间距<=4',flow='复用同一100组位置/目标',reason='预注册缺失参数约定；不重抽')
    add('圆形围场五机随机场景','初速度','论文未写','PREREGISTRATION.md',paper=0,flow=0,reason='预注册假设')
    add('圆形围场五机随机场景','外圆约束','软约束','第五节',paper='h=9²-|p|²、二阶软CBF；松弛惩罚10000为假设',flow='外切正128边形硬线段墙，表面安全额外.005米',reason='最大外偏差11*(sec(pi/128)-1)=0.00331401345米；软墙改硬墙')
    add('圆形围场五机随机场景','速度上限','论文未写','PREREGISTRATION.md；CIRCLE_SPEED.receipt.json',paper='无速度界',flow=read(HERE/'CIRCLE_SPEED.json')['nominal_lqr_speed_p95'],reason='同一100组名义LQR初值、0–100秒、50ms保持双积分器的代数预测；不是另生成rollout')
    add('圆形围场五机随机场景','正式期刊版对照','正式元数据确认；正式正文未取得','IEEE Xplore document10294301，DOI10.1109/TCST.2023.3324531；LITERATURE_ACCESS_AUDIT.json',paper='使用arXiv全文；IEEE正文需JS验证、stamp入口不可访问',flow='同参数源',reason='正式刊期2024年3月32(2):705–712、在线2023-10-24；不能断言正文/表格与arXiv逐项相同')
    for name,v,source in [('机器人半径',.1,'config.py:34'),('速度上限',.3,'main.py:92–94覆盖config.py:14'),('角速度上限',3.8,'config.py:15'),('采样时间',.1,'config.py:6'),('预测时域',3,'config.py:7'),('CBF的γ',.1,'config.py:9'),('额外安全距离',.03,'config.py:10'),('位置权重','diag(11,11,.005)','config.py:24'),('控制权重','[1,.5]','config.py:25')]:
        add('门口场景',name,v,source,flow='不适用，单积分器与默认投影' if name in ('角速度上限','预测时域','CBF的γ','额外安全距离','位置权重','控制权重') else None,reason='动力学/安全层/名义controller变化' if name in ('角速度上限','预测时域','CBF的γ','额外安全距离','位置权重','控制权重') else '相同')
    add('门口场景','起点','(-1,.5,0)、(-1,-.5,0)','main.py:43–46',flow='去除朝向，原二维位置',reason='单积分器无朝向状态')
    add('门口场景','目标','(2,.1,0)、(2,-.1,0)','main.py:45–46',flow='去除朝向；保留两个目标恰好相切',reason='原任务不改；到达容差.08，实际碰撞判据保持<=直径')
    add('门口场景','墙几何','x=1，r=.1，y=.4..1.0及-.3..-.9','main.py:68–71',paper='保留重复y=-.3三次，绘图矩形不作计算障碍',flow='两紧包矩形[.9,1.1]×[.3,1.1]、[.9,1.1]×[-1,-.2]',reason='矩形包住圆，填充局部凹口；门宽.5米')
    add('门口场景','绕行通道','墙有限长，上下均可绕行','main.py墙列表',paper='无外边界，净通道宽度无上界',flow='同样无外边界',reason='不得补加外墙')
    add('门口场景','另一机器人障碍半径',.02,'main.py:99',flow='真实半径.1联合投影',reason='论文代码把他人作为当前位置的小静态圆，流程版真实联合碰撞约束')
    add('门口场景','时间语义','论文未写时限','官方main.py:N=50；mpc_cbf.py:383–435，两方法间模拟器不重置',paper='每轮两次各3步，共.6秒；50轮实际30秒。第二方法仅重置传给MPC的观测，不重置模拟器',flow='4*sqrt(9+.16)/.3，上取整至控制步，40.4秒',reason='早期raw物理时间15秒字段有误，独立分析勘误；原6步轨迹完整保留、不重跑')
    add('门口场景','期刊层级','Autonomous Robots 49:12 (2025)','任务指定文献',paper='不错的期刊；报告不按IJRR/IEEE T-RO同一档表述',flow='同文献背景',reason='期刊层级为本轮写作约定，非实验测量')
    for scene in ('窄通道场景','二维拥挤随机场景'):
        add(scene,'二阶离散化','论文式(1)前向Euler：位置输入块0','Chen等式(1)；官方uav.py:295–296；OFFICIAL_DYNAMICS_AUDIT.json',paper='官方先覆盖A后用于B：位置块h³/[2(1-h/2)²]，速度块h/(1-h/2)，均乘I',flow='单积分器速度输入：p_next=p+h*u',reason='官方不是正确Tustin；h=.2时B位置.00493827、速度.22222222。原代码/轨迹保持，不以修正实现匹配论文')
        add(scene,'缓冲Voronoi单元终端约束','Zhou等式(6)–(11)无末速度0约束','Zhou等2017；PREREGISTRATION_ADDENDUM_02.md',paper='修正分支去除误加的末速度0；首版仅作为变体留存',flow='不适用，各controller原算法',reason='右手规则检测1秒/.01米、右侧.5米与退出.1米仍为论文未写明的预注册假设')
    add('门口场景','绘图矩形与计算圆范围','main.py绘图上墙y=.3..1.3、下墙y=-1.3..-.3','main.py:147–148；计算圆main.py:68–71',paper='仅绘图；计算障碍实际上墙.3..1.1、下墙-1..-.2',flow='以计算圆紧包矩形为准',reason='绘图上端多.2米；下端多.3米且顶部少.1米，不把绘图几何当物理障碍')
    for scene in NAMES.values():
        add(scene,'流程动力学/执行链','论文分别为二阶或独轮车','任务第五节',paper='官方或论文方程原动力学',flow='单积分器；名义→速度界/到达停止→同一安全投影→OrthoFlow3→同一安全投影',reason='两个版本证据严格区分')
    (HERE/'PARAMETERS.md').write_text('# 参数逐项对照\n\n|场景|参数|论文值|出处|论文版|流程版|差异/原因|\n|---|---|---|---|---|---|---|\n'+'\n'.join('|'+ '|'.join(cell.replace('|','\\|').replace('\n',' ') for cell in r)+'|' for r in rows)+'\n')

if __name__=='__main__':main()
