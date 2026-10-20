from common import *
import copy
assert not (HERE/'PREREGISTRATION.receipt.json').exists()
prompt=ROOT/'docs/prompt_round38_more_blackbox_controllers_rhr_menu_20261017.md';protocol=ROOT/'docs/generalization_axes_protocol_20261015.md'
cfg={'pilot_seed_start':202610380000,'pilot_split':'round38_blackbox_pilot','eval_seed_start':202610381000,'eval_split':'round38_blackbox_evaluation','probe_protocol':str(D/'controller_flow_blend_v1/stage_2/probes/protocol.json'),'H':700,'x64':False,'pstar':0.9681620884830543,'CEM':[[10,3,64],[10,5,64],[10,3,256],[10,5,256],[30,3,64],[30,5,256]],'CEM_penalty_weight':100,'CEM_knots':3,'CEM_elite_fraction':.1,'CEM_std_floor':1e-3,'ASTAR':[[.05,10,'index'],[.05,10,'far'],[.05,25,'index'],[.10,10,'index'],[.10,10,'far'],[.10,25,'far']],'ASTAR_max_time':700,'ASTAR_lookahead':.3,'ASTAR_speed':.52,'ASTAR_slowdown_distance':.3,'ASTAR_moves':'8 neighbors and wait; time quantum=max(dt, resolution/.52), segment swept validation, reserve padded robot discs; goal held until max_time; no path means zero nominal','bootstrap_seed':202610173800,'bootstrap_replicates':5000,'pilot_k':[2,4,8,16],'pilot_primary_k':4,'expanded_fit_seeds':[17,23,41],'expanded_steps':list(range(500,12001,500)),'expanded_optimizer':{'name':'AdamW','lr':.0008,'weight_decay':.0001,'clip':5.,'batch':128},'expanded_folds':'six deterministic controller-stratified folds (each contains one Gaussian and one decentralized MPC and round-robin Flow); earliest minimum average heldout NLL across folds/seeds; descriptive leave-one-family at selected full CV step','tie_rule':'first menu item; fair right-hand tune lower rate descending then speed,steps,mode(always,slow,progress)','menus':{'full':['F10','F13','S500','Sprog','RHR'],'original':['F10','F13','S500'],'expanded':['F10','F13','S500','RHR']},'MDN_components':5,'diffusion':{'steps':100,'beta_start':.0001,'beta_end':.02,'schedule':'linear','sampler_steps':10,'sampler':'deterministic DDIM eta0, indices linspace99..0, final alpha1','time_embedding':'sinusoidal32, concatenate observation/action/time, MLP3x256 ReLU','dev_noise_seed_offset':38000}}
train=[]
for kind in ['deterministic','mixture','diffusion']:
 for i,g in enumerate(read(D/'axis2_controller_v1/GAUSSIAN_CONFIGS.json')):
  x=copy.deepcopy(g);x.update(name=kind+str(i),family=kind,source_gaussian_name=g['name'],checkpoint_selection='earliest minimum fixed dev MSE' if kind=='deterministic' else 'earliest minimum fixed dev mixture NLL' if kind=='mixture' else 'earliest minimum fixed-noise fixed-time dev denoising MSE');train.append(x)
write(HERE/'DESIGN_CONFIG.json',cfg);write(HERE/'TRAINING_CONFIGS.json',train)
text='''完整继承本轮 prompt 与共同协议（文件哈希见收据），不调整已规定的参数、能力门、终点或标签冻结顺序。所有写入限本轮目录，唯一例外为官方数据库追加登记、默认 journal 和单进程 merger；不访问被禁止的并行研究目录，不安装包，不执行 git 写操作。历史执行代码精确复制，路径由 __file__ 自动定位；完整性基线已留存。任何新执行前通过两组各16条逐位复现门。某族受阻只记录该族缺口并继续。

DESIGN_CONFIG.json 与 TRAINING_CONFIGS.json 补全 prompt 未指定的实现细节并冻结。交叉熵方法使用原随机采样代价，避碰权重100、三动作结点、每轮ceil(10%样本)精英、对角标准差下限0.001；无额外局部优化。优先级时空搜索使用八邻格与等待，连续线段几何检查与机器人圆盘时空预约，时间量子max(dt,栅格/.52)，到达者预约至700个时间量子，无路时零动作，不调整几何。重规划每K个物理步（每次episode重置），纯追踪0.3m、0.52m/s、近目标线性减速。六个配置严格保持原表。

三种行为克隆使用原六配方数据清单、train/dev划分、坐标、归一化、训练seed、步数、batch256及Adam3e-4；dev原validation_seed抽2048条，每250步及step0验证，最早最小损失；不打开test数据。混合密度五分量，logstd[-5,1.5]；扩散线性beta[.0001,.02]100步，正弦时间编码32维，固定dev噪声与时间，10步确定性去噪。动作反归一化、clip[-1,1]、原速度界、对应坐标转世界坐标，再限速；训练前验证坐标互逆和旋转性质。

pilot16与评估32使用固定新种子区间，和全DB状态及彼此按uid/content/物理字节去重，不静默重抽；旧物理状态不可恢复时披露可见性限制。调参复用历史十六状态，不计作评估；探测复用冻结十六状态而非新状态。能力门在pilot全程修正η10和按比例切换时间表seed0，两者均低15%或均高98%排除，unknown用上下界判门，模糊则不作合格；少于4个合格实例的族不作确认性族推断。

固定参数持续右手规则在任何本轮标签前独立按三组等权下界平均重算；公平调参每个合格实例全部20组×16状态，按族等权平均下界选一个规则。少量标签按pilot study_index前k条，在按比例切换时间表与固定参数持续右手规则间选择，平局前者。原选择模型、扩展选择模型、少量标签选择、公平调参选择全部封存后才登记/提交主评估。评估八臂含两基线、五菜单臂、公平调参持续右手规则，种子0/1。原选择模型与不看控制器选择模型、探测规则均复用历史冻结模型；原选择模型菜单限制为三臂。扩展模型复用历史冻结预处理和主成分，架构64/64/16 SiLU，四动作onehot；仅用原26控制器十六调参状态seed0的三时间表臂与本轮固定规则对应的历史持续右手规则标签训练。unknown不入训练似然，不重复运行。六折按控制器留出，选步500..12000每500，三fit seed均值dev NLL最早最小，再全量重训。留一族仅描述性。

统计按控制器再状态簇5000次配对bootstrap（双seed配对），oracle按原state index奇偶交叉拟合并另报样本内值，每次bootstrap重新选oracle。unknown保留下/上界，不声称已知失败，不重跑；timeout不称deadlock。主终点完整继承prompt八节。所有受阻族保留表格行，不能用缺失标签冒充零成功率。无视频。
'''
(HERE/'PREREGISTRATION.md').write_text('# 更多黑盒控制器与持续右手规则菜单：预注册\n\nUTC '+utc()+'\n\n'+text)
write(HERE/'PREREGISTRATION.receipt.json',dict(utc=utc(),sha256=sha(HERE/'PREREGISTRATION.md'),files={str(p):sha(p) for p in [prompt,protocol,HERE/'DESIGN_CONFIG.json',HERE/'TRAINING_CONFIGS.json']},before_new_results_and_rollouts=True))
# Independent fixed-rule computation; no new labels opened.
a=read(D/'blackbox_addendum_v1/RIGHT_HAND_GRID_RESULTS.json')['pools'];table=[]
for i in range(20):
 v=[a[k][i] for k in ['Flow','Gaussian','MPC']];assert len({json.dumps(x['rule'],sort_keys=True) for x in v})==1
 table.append(dict(grid_index=i,rule=v[0]['rule'],equal_pool_lower=float(np.mean([x['lower'] for x in v])),equal_pool_upper=float(np.mean([x['upper'] for x in v]))))
best=min(table,key=lambda x:(-x['equal_pool_lower'],x['rule']['speed'],x['rule']['steps'],['always','slow','progress'].index(x['rule']['mode'])))
write(HERE/'FIXED_RIGHT_HAND_RECOMPUTATION.json',dict(table=table,selected=best,source_sha256=sha(D/'blackbox_addendum_v1/RIGHT_HAND_GRID_RESULTS.json')))
seal('FIXED_RIGHT_HAND_FROZEN.receipt.json',[HERE/'FIXED_RIGHT_HAND_RECOMPUTATION.json',HERE/'trigger.py'],before_any_new_labels=True)
print(best)
