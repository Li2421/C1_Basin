from support import *
from registration import normalized_state,connect,canonical,uid
from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
import itertools
G=guard()
def signature(v):
 try:
  if 'physical' in v:v=v['physical']
  arr=[]
  for keys in [('positions','initial_positions'),('velocities','initial_velocities'),('goals',)]:
   a=np.asarray(next(v[k] for k in keys if k in v),np.float64).copy();assert a.ndim==2 and a.shape[1]==2;a[a==0]=0;arr.append((a.shape,a.tobytes().hex()))
  return uid('physical',dict(arrays=arr,timestep=float(v.get('timestep',v.get('step',0)))))
 except (TypeError,KeyError,StopIteration,AssertionError):return None
def main():
 assert (HERE/'INTEGRITY_BEGIN.json').exists();old=D/'decentralized_fix_selection_v1';z=np.load(old/'LABELS.npz');x=np.load(old/'DIAGNOSTIC_FEATURES.npz')['features'];rate=(z['labels']==1).mean((2,3));fam=z['families'];grid=[];best=None
 for A in GRID_A:
  for B in GRID_B:
   for R in [3,4,5]:
    rr=dict(A=A,B=B,right_hand_arm=ARMS[R]);score=source_score(select(x,rr),rate,fam);rr['old_five_family_mean_success']=score;grid.append(rr)
    if best is None or score>best['old_five_family_mean_success']+1e-14:best=rr
 sg=[];secondary=None
 for C in GRID_A:
  rr=dict(best,C=C);rr['old_five_family_mean_success']=source_score(select(x,rr,True),rate,fam);sg.append(rr)
  if secondary is None or rr['old_five_family_mean_success']>secondary['old_five_family_mean_success']+1e-14:secondary=rr
 write(HERE/'FROZEN_RULES.json',dict(primary=best,secondary=secondary,secondary_A_B_fixed_to_primary=True,source_label_sha256=sha(old/'LABELS.npz'),source_diagnostic_sha256=sha(old/'DIAGNOSTIC_FEATURES.npz'),unknown_policy='success lower bound',new_data_read=False));write(HERE/'OLD_PRIMARY_GRID.json',grid);write(HERE/'OLD_SECONDARY_GRID.json',sg)
 e=read(D/'decentralized_blackbox_planners_v1/CONTROLLERS.json');p=dict(e,profiles={},controller_order=[],experiment_uid=EXP+'_main');r14=read(D/'ring_independent_flow_confirmation_v2/protocol.json');checkpoints=[]
 for i,n in enumerate(r14['controller_order']):
  name=f'TD{i:02}';v=dict(r14['base_profiles'][n]['payload'],controller_class='TD',source_name=n);assert sha(v['flow_checkpoint_path'])==v['flow_checkpoint_sha256'];p['profiles'][name]=dict(payload=v);p['controller_order'].append(name);checkpoints.append(dict(name=name,path=v['flow_checkpoint_path'],sha256=v['flow_checkpoint_sha256']))
 f=read(D/'teacher_student_macflow_v1f/DISTILLATION_GATE.json')['controllers'];fmap={(r['variant'],r['index']):r for r in f}
 for variant in ['V1','V3']:
  for i in range(14):
   r=fmap[variant,i];assert sha(r['checkpoint'])==r['checkpoint_sha256'];name=f'ST{variant}_{i:02}';v=dict(read(D/'decentralized_pipeline_v1/main_protocol.json')['base_profiles'][read(HERE/'TEACHERS.json')[i]['name']]['payload'],controller_class='ST',source_index=i,variant=variant,flow_checkpoint_path=r['checkpoint'],flow_checkpoint_sha256=r['checkpoint_sha256']);p['profiles'][name]=dict(payload=v);p['controller_order'].append(name);checkpoints.append(dict(name=name,path=r['checkpoint'],sha256=r['checkpoint_sha256']))
 used=read(D/'axis2_controller_v1/DESIGN_CONFIG.json')['DM'];unused=[list(v) for v in itertools.product([10,30],[64,256],[20,100]) if list(v) not in used];assert len(unused)==2
 newdm=unused+[[20,128,50],[20,128,150],[10,128,50],[30,128,150]];newas=[[.075,k,obs] for k in [20,30] for obs in [False,True]]+[[.1,10,False],[.1,50,False]]
 oldas=[e['profiles'][f'AS{i}']['payload']['nominal_parameters'] for i in range(6)];assert all(v not in used for v in newdm) and all(v not in oldas for v in newas)
 for kind,values in [('DM',newdm),('AS',newas)]:
  for i,vv in enumerate(values):
   n=f'{kind}{i}';v=dict(e['profiles'][f'{kind}0']['payload'],nominal_parameters=vv);p['profiles'][n]=dict(payload=v);p['controller_order'].append(n)
 recipes=['base_u_v2_macflow','base_u_v4_macflow','base_u_plus_w_v4_macflow','base_u_v9_local_macflow'];oldg=read(D/'axis2_controller_v1/GAUSSIAN_CONFIGS.json');assert not set(recipes)&{r['recipe'] for r in oldg};base=read(D/'controller_universal_eta_schedule_v1/protocol.json')['base_profiles'];g=[]
 for i,n in enumerate(recipes):
  v=base[n]['payload'];cfg=read(D/'ring_exchange_stage1'/n/'config.json');r=dict(oldg[0],name=f'G{i}',recipe=n,seed=2026101701+i,validation_seed=2026102701+i,steps=cfg['steps'],frame=v['observation_frame'],manifest=v['adapter_manifest_path'],manifest_sha256=sha(v['adapter_manifest_path']),flow_checkpoint=v['flow_checkpoint_path'],flow_checkpoint_sha256=sha(v['flow_checkpoint_path']));g.append(r);p['controller_order'].append(f'G{i}')
 write(HERE/'GAUSSIAN_CONFIGS.json',g);write(HERE/'CONTROLLERS_PENDING_GAUSSIAN.json',p);write(HERE/'FROZEN_EXISTING_CHECKPOINTS.json',checkpoints)
 native=dict(environment=read(D/'decentralized_pipeline_v1/base_protocol.json')['environments']['ring_exchange']);seen=set();ids=set();contents=set();unresolved=[]
 with connect(True) as db:
  rows=db.execute('SELECT state_uid,content_hash,physical_state_json FROM state').fetchall()
 for r in rows:
  ids.add(r['state_uid']);contents.add(r['content_hash'])
  try:sig=signature(json.loads(r['physical_state_json']))
  except (TypeError,json.JSONDecodeError):sig=None
  if sig:seen.add(sig)
  else:unresolved.append(r['state_uid'])
 states=[];reject=[]
 for seed in range(2026101720000,2026101721000):
  s=generate('round30i_confirm',1,seed,native)[0];sig=signature(s['physical'])
  if s['state_uid'] in ids or s['content_hash'] in contents or sig in seen:reject.append(seed);continue
  seen.add(sig);s['study_index']=len(states);states.append(normalized_state(s))
  if len(states)==32:break
 assert len(states)==32;write(HERE/'EVAL_STATES.json',states);write(HERE/'STATE_DEDUP_AUDIT.json',dict(utc=utc(),DB_states=len(rows),physical_resolved=len(rows)-len(unresolved),unresolved_legacy_state_ids=unresolved,all_DB_uid_content_checked=True,physical_comparison_all_resolvable_states=True,selected_seeds=[s['metadata']['draw_seed'] for s in states],candidate_interval=[2026101720000,2026101720999],rejected_seeds=reject,selected=32,passed=True))
 text='''# 新实例与新状态的修法选择前瞻确认（CONFIRMATORY）

本预注册在任何本轮训练、仿真和新探测之前封存。主规则只用round30h的46个旧controller、全部五族旧标签与旧诊断，以原A/B网格、原三组右手规则、各族等权下界平均成功率最大化；按A、B、甲/乙/丙循环顺序保留最早最优（浮点tie tolerance1e-14）。拟合结果如下，数值收据先于所有新探测。

FROZEN_RULES_JSON

探索性两级规则保持主规则A/B不动，仅穷举C与原A相同网格[0,.01,.025,.05,.1,.2,.35,.5,.75,1]，freeze>C选右手规则甲，否则右手规则乙；目标、平局与旧源族相同。不得再拟合。

58实例：14个round14A独立seed教师，28个round30f最终round3/step20000学生（闭环重标注14与共享随机种子14），4个全新配方高斯行为克隆，6新模型预测控制，6新A*。全部清单、checkpoint哈希与原父身份写入CONTROLLERS_PENDING_GAUSSIAN/FROZEN_EXISTING_CHECKPOINTS。两学生变体仍属于同一个分布式学生族（28实例）；各族并非独立模型谱系，结论只限规定样本。既有学生在旧状态上曾被评估，本轮前瞻性来自冻结规则、相对round30h的新实例与全新状态，不声称这些checkpoint从无任何历史闭环结果。

Gaussian_configs_JSON

Gaussian逐字复用round33训练函数，MLP256×3 ReLU，8维联合mean与logstd[-5,1.5]，Adam3e-4，batch256；对应原Flow步数、同一train数据与normalization；固定dev8×256，每250步加step0，最低NLL且最早tie选checkpoint，禁止任务标签选权重。只训练上述4个，GPU至多2 shard，不重训既有模型。

MPC_JSON

ASTAR_JSON

32个新评估状态：原round28 generate/LocalFrameConfig生成器，候选区间[2026101720000,2026101720999]，按顺序接受前32个，排除DB所有uid/content与可解析物理签名（包括signed zero归一），保留不可解析历史记录列表；每状态seed0、1，RNG保持原key_for(uid,seed,physical_step)。STATE_DEDUP_AUDIT先于提交封存。菜单与round30h相同：只加安全层不修正、全程η13、前500步η13后η10、右手规则甲progress20/.35、乙neighbor_right/.8/.10、丙progress40/.25。原物理、事件、安全、修正和右手规则核心逐字复制，不改变任何成功或终止标准。

三道复现门：round33与round30e各16条，round28分布式安全层16条；保存位置/速度/动作dtype、shape、bytes、步数、终点完全一致。复现是用户明确要求的例外；不增补重复任务标签或journal。高斯训练冻结之后，58×16旧R5 probe状态、seed0、安全层300步、关闭终止。诊断函数逐字复用round30h，7维均值定义完全相同：unfinished速度<.01机器人步/1200；当前speed>=.01且40步目标距离减少<.05的机器人秒（主定义不加unfinished，另保留有该条件敏感性）；初始team距离减100/200/300距离；每机器人步执行与nominal之差L2均值；累计角位移unwrap±.15rad的mixed时刻比例。16探测均值作controller输入，不读任务summary/termination。任何数值缺口保留不重跑，不填主特征，受影响部分记缺口。

旧冻结通用critic只使用其原菜单F10/F13/S300/S500。复用14新教师的round14A既有R5（原联合流协议，明确与分布式协议区别）以及28学生round30f既有R5；新Gaussian/MPC/A*共16实例缺少对应R5，按原16×{F10,F13}×200协议仅补512条无标签probe，零任务DB行。原critic权重、source transform不变，不调参；用32新state的已知几何特征作原模型输入，绝不读任务标签。全部诊断与R5先整体封存，再封存所有主、次要、旧critic选择。若旧critic选择菜单外原动作F10或S300，只为该选择补对应64条任务，以评估该对照，不加入主六臂菜单或oracle。这是原菜单对照的必要标签缺口；不重复其他原菜单全臂评估。

全局时序：本预注册与两规则收据→GPU训练4Gaussian并冻结全部58实例→全部诊断与R5冻结→所有新controller主/次要/旧critic选择冻结→design与官方preflight封存→只缺失任务→默认官方journal→ps/squeue确认无其他merger→单进程官方merger→postflight→打开新标签分析。复现门可与GPU训练并行，但全部门完成后才开始新实例诊断。所有入口先加载官方shared_rollout_db再PoC路径、全部导入后断言包和默认journals路径、恢复x64=False。DB登记只INSERT OR IGNORE、逐旧行hash与rollout计数前后相同；未知不重跑，不作失败。timeout不是deadlock。

确认一：主规则减6臂中最佳单一固定臂，5000次配对分层controller→state bootstrap，每次重新取最佳固定臂，95%百分位CI下界>0才PASS。确认二：同样主规则减全体500步时间表。seed2026101710；每族内重抽controller保持28/14/4/6/6，所抽controller内重抽32状态，2个seed成簇；所有策略共享抽样。主要合并58controller等权，另报五族等权。unknown下界0上界1，点值按下界，另给保守差值包络，不以unknown判失败。每族/每controller事后最优都只含六臂，平局按菜单顺序，类型为不修正、时间表（含全程η13）、右手规则；类型准确率与允许并列最优类型的敏感性均报。灾难为某controller相对本族最佳固定臂族平均低严格>10pp，另报unknown可能范围。

确认三描述每族成功率与最佳固定臂差、类型准确率、灾难数。两级规则仅探索性：总体两级−主、MPC族两级−主与配对CI；“明显更好”预先定义其MPC差CI下界>0，不升级为本轮主终点。每controller遗憾相对其六臂事后最优，配对CI条件于冻结选择，不重新拟合。原通用critic仅描述性对照，协议差异单列，不能据其差异重选主规则。

结论映射严格照任务：确认一和二均通过才主方法前瞻确认；只二通过则仅优于默认；一不通过且某族错则分析冻结特征阈值关系；次要MPC通过建议下一轮将两级作为主规则确认，不将探索性改称确认。既有右手菜单曾用族结果调参，历史菜单族信息不是本轮全新发现。

完整性BEGIN覆盖PoC全树、所有列出的历史及依赖全树、全部非并行Basin已有Python源码与两个仓库只读git快照。并行目录按开始squeue/最近24h/untracked规则识别、封存；本轮不导入或写它们，变化独立归因，不通过改外部文件来消除审计差异。所有新代码/缓存/日志/图只写本目录；官方DB流程为唯一例外。最多18 CPU、2 GPU shard，提交前计入running和pending预留；视频不要求。
'''
 text=text.replace('FROZEN_RULES_JSON',json.dumps(read(HERE/'FROZEN_RULES.json'),ensure_ascii=False,indent=2)).replace('Gaussian_configs_JSON',json.dumps(g,ensure_ascii=False,indent=2)).replace('MPC_JSON','新模型预测控制参数：'+json.dumps(newdm)).replace('ASTAR_JSON','新A*参数：'+json.dumps(newas));(HERE/'PREREGISTRATION.md').write_text(text);write(HERE/'PREREGISTRATION.receipt.json',dict(utc=utc(),sha256=sha(HERE/'PREREGISTRATION.md'),files={str(HERE/n):sha(HERE/n) for n in ['FROZEN_RULES.json','OLD_PRIMARY_GRID.json','OLD_SECONDARY_GRID.json','GAUSSIAN_CONFIGS.json','FROZEN_EXISTING_CHECKPOINTS.json','CONTROLLERS_PENDING_GAUSSIAN.json','EVAL_STATES.json','STATE_DEDUP_AUDIT.json']},before_new_training_and_probes=True,before_new_labels=True,guard=guard()));source_inventory('imports_preregister.json');print('FROZEN RULES',best,secondary,'58 controllers',flush=True)
if __name__=='__main__':main()
