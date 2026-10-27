from experiment import *
from states import config
import collections

def main():
    text=['# 场景构造与证据层级','','论文版与流程版参数逐行差异见PARAMETERS.md；所有结论均为开发证据。几何有效、单机器人有路、联合可解、控制器成功、死锁解除是五层不同证据，本轮不以栅格单机有路冒充联合可解。原始状态、排除原因与路径检查均有封存收据。图只读取保存的场景与轨迹，最后生成。','']
    for scene,name in NAMES.items():
        rows=read(HERE/f'GEOMETRY_VALIDATION_{scene}.json');good=[r for r in rows if r['eligible']];count=collections.Counter(r['N'] for r in good)
        cfg=config(scene,good[0]['N'] if good else rows[0]['N']);text += [f'## {name}（{scene}）','',f'原始{len(rows)}个状态；有效{len(good)}个，剔除{len(rows)-len(good)}个。有效数量分布：'+ '、'.join(f'{n}机{k}个' for n,k in sorted(count.items()))+'。', '',f'半径{cfg.agent_radius}米，周期{cfg.dt}秒，速度界{cfg.max_speed:.15g}米/秒，最大{int(50/.15) if scene=="lit_crowded_random" else cfg.max_steps}步。容量给出显式安全点阵的可容纳数量下界；下界达到N仅证明能放置，不证明任意起终点联合可解。全部eligible状态通过膨胀墙栅格的逐机器人路径检查；联合可解未形式认证。','','构造调用：`states.config(scene,N)`、`states.walls(scene)`、`flow_scene.LiteratureEnv(state,config)`；原始状态在`FLOW_STATES_UNCHECKED.json`，有效集合在`FLOW_STATES_'+scene+'.json`，逐项检查在`GEOMETRY_VALIDATION_'+scene+'.json`。', '',f'![{name}](figures/{scene}.png)','']
        if scene=='lit_narrow_passage':text+=['无墙三机。两台已到达机器人保留为真实机器人，图8第一帧的穿行机器人位置声明为reset假设；不能恢复未公开的真正t0。零初始目标距离的进度扩展只解决原函数未定义处；非零距离仍用原进度定义。','']
        elif scene=='lit_crowded_random':text+=['官方GUI采样器原函数体保持；硬工作区边界使672组原样本不满足流程版几何条件。没有补抽、缩小半径或把无效状态算成死锁。有效状态仅2机、4机，不能对8机至14机作控制成功或学习迁移结论。工作区能放置的容量证据与这批起终点是否有效分别报告。','']
        elif scene=='lit_circle_five':text+=['外圆半径11米，流程版用外切正128边形硬线段墙，最大向外偏差0.00331401345米；原论文外圆为软约束。机器人中心采样拒绝半径>9的提案。流程版速度界由同一100组名义采样保持LQR代数预测的全100秒速度95分位确定，不另生成优化rollout。实际目标有路而粗规划网格终端cell贴墙时，只算对应controller的能力问题。','']
        else:text+=['参与计算的墙圆由两个紧包矩形包围：[.9,1.1]×[.3,1.1]与[.9,1.1]×[-1,-.2]；矩形填充相邻圆外缘凹口，门宽.5米。上下墙有限长、无外边界，外侧绕行宽度无上界。两个目标恰好相切；只在reset目标数据校验允许相切，真实占据状态和原swept碰撞判据未放宽。时限40.4秒。','']
    text+=['','## 用户决定的最终统计范围','','v1按用户决定收尾并转入v2。窄通道、二维拥挤随机、门口三个完成的流程版场景进入迁移统计；圆形流程版保留682条raw及完成的48000条单机筛选，8918条未落盘安全基线请求丢弃、不记unknown、不恢复。圆形几何图只表示已构建的几何，不表示迁移实验完成。不得重启v1协调器、恢复圆形claims或提交学习型试点。']
    (HERE/'SCENES.md').write_text('\n'.join(text)+'\n')
    ctls=read(HERE/'CONTROLLER_MANIFEST.json');types={r['type_code']:r['type'] for r in ctls}
    text=['# Controller移植','','48个实例、8个类型的参数逐项来自CONTROLLER_MANIFEST.json与历史ZOO_MANIFEST.csv。旧能力标记仅作来源记录，新场景能力以本轮单机器人>=95%为准。所有新类均在本轮目录；旧源码、超参数、冻结checkpoint与默认安全配置未修改。','','|类型|几何/机器人数适配|保留的算法|','|---|---|---|']
    descriptions={
    'cross_entropy':('将固定四机动作shape改为N，圆环代价改为本场景墙净间隙','原3段knots、采样数/迭代数/时域、精英10%、均值方差更新与返回样本'),
    'distributed_cross_entropy':('逐实际机器人读取其他者的coast预测与新墙代价','原逐机fold_in、3段knots、采样/精英/迭代参数'),
    'distributed_shooting':('任意N逐机rollout、实际几何与速度上限','原分布式shooting、名义/零候选、时域/样本数/碰撞权重'),
    'gradient_mpc':('实际N与墙代价；无配对时空集不取空均值','原3结点插值、Adam、迭代数/学习率、warm start、自定义范数导数、JAX float32'),
    'centralized_mppi':('联合动作维数2N与新墙代价','原样本数/时域/温度、softmax加权和warm-start平移'),
    'centralized_sampling':('联合动作维数2N与新墙代价','原shooting、名义/零候选、时域/样本数/碰撞权重'),
    'prioritized_astar':('固定环形边界替换为任意有限线段接口，实际N与物理速度作为输入','原分辨率、重规划间隔、优先顺序、700层搜索、跟踪规则'),
    'conflict_based_search':('任意N与墙段接口，位置/预约/负约束原样','原冲突树、预算、分辨率、单机搜索与跟踪规则')}
    for t,name in types.items():a,b=descriptions[t];text.append(f'|{name}（{t}）|{a}|{b}|')
    text+=['','精确缓存后端只缓存原始墙判断和同状态约束矩阵；取整目标无入边证书是原边谓词的必要条件，穷举返回值不变。不移动目标、不缩小机器人、不改变规划分辨率。GRID_CERTIFICATE_PROOF.json、STRICT_PROJECTION_GATE.json、STRICT_CONTROLLER_GATE.json与相关收据给出证明和全轨迹字节检查。规范执行身份保留，实际加速后端另写raw。','','|实例全称|原参数|新场景能力结果|','|---|---|---|']
    q={}
    for scene in NAMES:
        p=HERE/f"{('capability_v2_' if scene=='lit_narrow_passage' else 'capability_')+scene}_QUALIFICATION.json"
        if p.exists():
            for r in read(p):q[(scene,r['controller'])]=r
    for c in ctls:
        info=[]
        for s,name in NAMES.items():
            r=q.get((s,c['controller']));info.append(name+('待完成' if r is None else f"{'合格' if r['qualified'] else '不合格'} {100*r['rate_lower']:.2f}%"))
        text.append('|'+f"{c['type']}（{c['controller']}）|{c['parameters']}|"+'；'.join(info)+'|')
    text+=['','圆形场景另有精确零墙代价捷径：整批float64预测点在保守内圆时，原128个单位法向点积的正部平方都严格为0，直接返回同字节零数组；否则沿用原表达式。随机数、样本、参数、代价其余项、JAX分支与算法不变。16条已有五机轨迹覆盖8类，动作、位置、安全投影与终点逐字节一致；见CONTROLLER_INTERIOR_EQUIVALENCE和DEPLOYMENT收据。实际backend单列，原执行语义/DB身份和旧标签有效性保持。','', '确定性行为克隆策略与混合密度行为克隆策略仅完成第二优先级试点配方与接口准备，按用户决定未启动训练、能力门或迁移评估，没有试点权重与能力结果。原计划保留三层256 ReLU、五分量混合密度与训练规则，扩展机器人数量与几何输入字段，世界坐标观测对齐世界坐标专家速度；固定N联合网络的单机器人查询使用零编码槽位，物理中仅保留一台真实机器人。这些是预注册补充四与PILOT_CONTROLLER_MANIFEST中的计划，不是已完成实验。最终停止状态见PILOT_NOT_STARTED.md与PILOT_GAP文件，后续工作以指定v2文档为准。']
    (HERE/'CONTROLLER_PORTING.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':main()
