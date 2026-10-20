from pathlib import Path
import json,hashlib,time,difflib
H=Path(__file__).resolve().parent;O=H.parent/'multiscene_generalization_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for n,a,b in [('common.py',"EXP='exp_multiscene_generalization_v1'","EXP='exp_multiscene_navigation_competence_v1'"),('core.py',"('XDG_CACHE_HOME','cache')","('XDG_CACHE_HOME','cache'),('CUDA_CACHE_PATH','cuda_cache')"),('registry.py','round=39','round=40'),('registry.py','Multiscene generator cross-scene generalization','Multiscene navigation competence pilot'),('submit_stage.py',"'r39_'","'r40_'" )]:
 p=H/n;p.write_text(p.read_text().replace(a,b))
for n in ['reproduction.py']:
 (H/n).write_bytes((O/n).read_bytes())
p=H/'reproduction.py';s=p.read_text().replace("STAGE='reproduction_v2'","STAGE='reproduction_round40'").replace("c['seed']=3900","c['seed']=4000").replace('future index3900','future index4000').replace("    subprocess.run([sys.executable,'-B',str(HERE/'test_geometry.py')],check=True)","")
p.write_text(s)
for n in ['runtime.py','controllers.py','core.py','common.py','registry.py','submit_stage.py','reproduction.py']:
 (H/(n+'.diff')).write_text(''.join(difflib.unified_diff((O/n).read_text().splitlines(True),(H/n).read_text().splitlines(True),fromfile=str(O/n),tofile=str(H/n))))
pr='''本轮导航能力预注册（串联瓶颈走廊与房间与门试点优先）

只写本轮目录；共享数据库仅官方追加登记、预检、缺失运行、默认 journal、单进程合并与后检。旧 unknown 不重跑。开始审计与复制来源已封存。单机器人环境和生成几何逐字节复用上一轮通过的等价性测试，不重做；执行链在本轮复制版重现十六条历史记录，未来索引4000，确定性控制器不使用未来键。先完成复现再产生新闭环。

导航场栅格0.05米，墙体余量为点到墙距离减原碰撞边距；边代价为长度乘两端权重均值，权重1+4max(0,0.15-c)/0.15。8邻域边要求原静态扫掠几何可通行；目标作为虚节点连接0.10米以内可见格点。从目标Dijkstra整局固定。跟随点用双线性代价梯度下降，固定0.025米积分步长；不可插值、梯度为零或指向不可通行处时回退可见8邻域最大下降方向。目标可见且剩余距离不超过剩余跟随弧长和0.10米时取目标。专家朝0.3米跟随点，原最大速度，目标0.3米内线性减速，原目标停止。观测保留原35维，追加世界坐标0.5米和1.5米跟随点相对位置。导航场不看其他机器人。

训练几何每族原十六个加四十八个；新训练种子基202640000000+族序号10000000+地图索引10000，48个索引0至47；导航留出基202650000000+族序号10000000+索引10000，8个。每地图拒绝后取下一种子最多10000，不与旧所有角色及本轮其他地图种子或矩形内容重叠。几何与原路径及共享资源判据继承，联合可解沿用顺序专家原默认10000步；旧几何重用旧联合证据不重跑。新地图须官方流程获取联合证明。

专家门每训练地图32条单机器人回合，起终点按原task；任务种子202660000000+族序号100000000+地图索引100000+回合索引100，非法任务取下一个种子至多100次，机器人索引回合索引模4；各自H沿原最长路径预算规则。每族至少2048条，成功>=99%且零撞墙才采集示范。专家门同时报告距原Geometry.path超过0.5米的状态比例（描述性，墙体余量路线可能有意绕远），不作死锁标签。若在原十六地图的专家门子集已出现撞墙，零撞墙条件已不可能满足，可提前判定该族专家门不通过并停止该族后续几何扩展及训练；子集分母必须明确，不冒充完整2048条。仍继续另一试点族。这个逻辑提前停止只使用专家门必要条件，绝不对失败设置修订重跑。

通过专家门才运行示范与训练。第零轮每训练地图32训练回合、8开发回合；后三轮每地图每实例16回合，固定任务种子各轮分离，逐个访问状态用专家重标注，碰撞前状态保留，开发集固定不增。每实例每轮从其种子重新初始化，在累计数据上训练60000步，Adam0.0003，batch256，每250步固定2048开发样本取最早最小损失。网络三层256，确定性均方误差、五高斯混合密度，种子17/23，第三轮固定作为最终；第零轮与第三轮都作诊断，禁止留出选择轮次。导航门只用新八地图十六回合>=15成功；两试点都>=2实例通过才扩展。

本轮不修改物理、成功标准、机器人半径、墙体边距。若试点不通过按用户指令停止并交付；训练未达到阶段时第零轮和第三轮失败图标未运行，不伪造模型结果。修正标签保持未打开。进入修正阶段才冻结完整选择模型设计并执行原协议；未进入时终点、规模曲线与锁定测试均缺口，不能画零曲线。所有结果只描述实际达到的阶段。

初始历史审计：上一轮七族28策略全部未过合法导航门。串联瓶颈走廊原专家257/640撞墙，房间与门134/640；旧35维数据与当前专家及39维定义不兼容，不能充作新示范。原训练几何、路径和联合可解原始证据可复用；原检查点可作旧基线而不重复训练。原评估几何已参与导航设计，新的导航门另立种子。多机器人修正仍按本轮提示保留上一轮CATALOG_FROZEN原状态；注意它的旧评估几何与训练重叠，不影响本轮新导航留出，但修正实验若进入必须明确原样沿用。正式测试与开发对照不混淆。
'''
(H/'PREREGISTRATION.md').write_text(pr)
(H/'PREREGISTRATION.receipt.json').write_text(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),sha256=sha(H/'PREREGISTRATION.md'),files={str(H/'navigation_field.py'):sha(H/'navigation_field.py')}),indent=2)+'\n')
(H/'HISTORY_AUDIT.md').write_text(pr.split('初始历史审计：')[1])
print('configured and preregistered')
