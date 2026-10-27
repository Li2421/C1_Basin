from experiment import *

def main():
    p=HERE/'PREREGISTRATION_ADDENDUM_04.md'
    p.write_text('''# 预注册补充四：共同协议背景参照与行为克隆试点

在这些分支第一条rollout和首次训练前写定，不改已运行的主要实验。共同协议要求名义输入到安全投影的逐局转移，因此为48个原控制器及名义朝目标白盒参照补齐不加安全层的背景：几何剔除后256状态×49实例×2seed=25088条。白盒的只加安全投影及三种固定修正为256×1×2×4=2048条。执行名义输入时不调用安全过滤器，仍沿用原有速度界、到达停止与物理碰撞事件。原预注册的1856条是未剔除928状态的白盒名义参照上限，没有包含共同协议要求的48控制器背景转移；本补充明确该计数扩展及必要性，精确请求和缓存收据仍早于提交。

行为克隆试点严格等全部四场景原48实例与白盒的背景/固定修正完成后开始。使用每个场景封存32个训练状态中既有冲突搜索多机规划成功轨迹；每个状态按原清单次序取第一个成功实例的seed0轨迹，不新增专家rollout；index模4等于0为内部dev，其余为train。评估状态不参与专家筛选或归一化。若内部train或dev无示范则记缺口。二维拥挤随机场景严格限定8机、14机；预定状态均被硬墙几何检查剔除，不改成2机/4机训练。

各场景确定性行为克隆策略、混合密度行为克隆策略均采用原TRAINING_CONFIGS.json中该family第一份global配方，两个独立训练seed2026103301/2026103302（验证seed2026104301/2026104302）。这是同一5000步配方的两个seed，不把原清单其他配方的60000或100000步混入试点。三层256 ReLU、Adam .0003、batch256、每250步在固定2048个dev样本上验证、最早最小损失checkpoint；混合密度为5分量且log_std界[-5,1.5]。原世界坐标自身/目标/其他机器人相对状态字段保持；只把固定四机循环改为场景N，把原圆环几何5字段改为最近墙相对向量、机器人半径、场景尺度与墙净间隙。均值/尺度只由新训练示范计算，尺度下限.001。N固定的联合网络单机能力查询以零输入占位编码被移除者，只有实际保留者进入物理/投影/成功判据；保留者原槽位写入controller身份，防止额外state元数据未进入物理三元组导致缓存冲突。能力门仍为>=95%，不改阈值。试点不声称可直接泛化N=100。

所有数值unknown保留、不重新试验。新critic编码或新增类型不使已有执行语义相同的标签失效。学习型试点若进入新类型留一学习，原8类标签与既有同动作评估通过官方merger/preflight复用，扩展分支单独报告，不覆盖8类结果。
''')
    paths=[p,*[HERE/n for n in ('pilot_training.py','pilot_policies.py','pilot_runtime.py','pilot_nominal_runtime.py','pilot_design.py','nominal_runtime.py','design_background.py')]]
    seal('PREREGISTRATION_ADDENDUM_04.receipt.json',paths,before_any_pilot_training=True,before_any_background_rollout=True)
    sources=[ROOT/n for n in ('diagnostics/blackbox_families_v1/TRAINING_CONFIGS.json','diagnostics/blackbox_families_v1/train_policies.py','diagnostics/blackbox_families_v1/learned.py','ring_exchange/environment.py')]
    write(HERE/'PILOT_REUSED_SOURCE_START.json',dict(utc=utc(),files={str(p):sha(p) for p in sources},phase='before pilot execution; supplements opening integrity snapshot'))

if __name__=='__main__':main()
