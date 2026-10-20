from core import *
from guard import guard
from audit_utils import source_inventory
from shared_rollout_db.src.rollout_db import connect
from ring_exchange.environment import sample_initial_state
import jax
G=guard()
def signature(p):
 vals=[]
 for keys in [('positions','initial_positions'),('velocities','initial_velocities'),('goals',)]:
  try:v=np.asarray(next(p[k] for k in keys if k in p),np.float64).copy()
  except (StopIteration,TypeError,ValueError):return None
  if v.shape!=(4,2):return None
  v[v==0]=0;vals.append(v.tobytes().hex())
 return hashlib.sha256(json.dumps(vals).encode()).hexdigest()
def main():
 assert not (HERE/'PREREGISTRATION.receipt.json').exists()
 ts=read(HERE/'TEACHERS.json'); checks=[]
 for i,r in enumerate(ts):
  assert r==read(D/'teacher_student_macflow_v1/TEACHERS.json')[i]
  for pathkey,hashkey in [('checkpoint','checkpoint_sha256'),('train_config_path','train_config_sha256'),('dataset_manifest_path','dataset_manifest_sha256')]:assert sha(r[pathkey])==r[hashkey]
  s=read(HERE/'STUDENTS.json')[str(i)];assert sha(s['checkpoint'])==s['checkpoint_sha256']
  dp=D/'teacher_student_macflow_v1/data'/r['name'];ready=read(dp/'READY.json')
  for n,h in ready['files'].items():assert sha(dp/n)==h,(i,n)
  checks.append(dict(index=i,name=r['name'],teacher=r,student=s,dataset_ready_sha256=sha(dp/'READY.json'),dataset_files=ready['files']))
 with connect(True) as db:
  dbstates=[dict(r) for r in db.execute('SELECT state_uid,content_hash,physical_state_json FROM state')]
  controllers=[dict(r) for r in db.execute('SELECT controller_uid,config_json FROM controller_config')]
 seen={signature(json.loads(r['physical_state_json'])) for r in dbstates if r['physical_state_json']};unresolved=None in seen;seen.discard(None)
 for p in [HERE/'EVAL_STATES.json',HERE/'PILOT_STATES.json',D/'teacher_student_macflow_v1c/TUNING_STATES.json',D/'decentralized_pipeline_v1/PILOT_STATES.json']:
  if p.exists():seen.update(signature(s['physical']) for s in read(p))
 selected=[];reject=[]
 for rd in range(3):
  states=[]
  for seed in range(2026101606000+rd*10000,2026101606000+rd*10000+10000):
   instance=sample_initial_state('train',seed);p={k:getattr(instance,k).tolist() for k in ['positions','velocities','goals']};sig=signature(p)
   if sig in seen:reject.append(seed);continue
   seen.add(sig);states.append(dict(uid=f'round30f_train_r{rd+1}_seed{seed}',seed=seed,split='train',round=rd+1,physical=p,collection_arm='nominal only' if len(states)<32 else 'safety baseline',physical_signature=sig))
   if len(states)==64:break
  assert len(states)==64;selected.append(states)
 write(HERE/'TRAIN_STATES.json',selected)
 write(HERE/'HISTORY_AUDIT.json',dict(utc=utc(),checks=checks,DB_state_rows=len(dbstates),DB_controller_rows=len(controllers),DB_physical_unresolved=unresolved,lexical_new_variant_matches=[r['controller_uid'] for r in controllers if 'teacher_student_macflow_v1f' in r['config_json']],training_state_rejected_seeds=reject,reuse={'original_student_labels':['teacher_student_macflow_v1/main_DB_ROWS.json','teacher_student_macflow_v1b/main_DB_ROWS.json','teacher_student_macflow_v1c/main_DB_ROWS.json','teacher_student_macflow_v1d/fair_DB_ROWS.json'],'original_choices':['teacher_student_macflow_v1/FROZEN_CHOICES.json','teacher_student_macflow_v1b/FROZEN_CHOICES.json','teacher_student_macflow_v1c/FROZEN_CHOICES.json'],'teacher_labels':'decentralized_pipeline_v1 + teacher_student_macflow_v1b/c teacher supplement'},gaps=['No closed-loop relabeled checkpoints or shared-noise student checkpoints in searched rounds. These change policy execution and need new labels after all freezes. Original student and teacher labels reused.'],guard=G))
 prompt=ROOT/'docs/prompt_round30f_student_competence_dagger_shared_noise_20261016.md'
 text='''# 闭环重标注与共享随机种子学生预注册（round30f）

完整继承任务文档（以下原文是本预注册的一部分）。原学生（V0）只复用；闭环重标注学生（V1）从原学生继续；闭环重标注加共享随机种子学生（V3）离线初训后继续。所有最终版本固定在第三轮第20,000步，不以闭环结果挑选。

补充明确的执行参数：每个变体共享 TRAIN_STATES.json 中3轮各64个train状态；种子候选区间分别[2026101606000,2026101615999]、[2026101616000,2026101625999]、[2026101626000,2026101635999]；前32局只跑学生，后32局只加安全层；seed=0。每个物理动作之前t%5==0标注，包括t=0，不标注终止后的状态。教师标注键使用 key_for(training_state,0,t)，共享随机种子学生使用 fold_in(key,3006)，联合噪声标签3006。训练noise来自相同教师trace第0步；离线数据每次更新抽新教师键，重标注样本保留实际教师键和逐位noise。合并后的数据按transition均匀抽样；每个状态输出4个agent row。保持原train normalization，避免改变原学生checkpoint函数；Adam学习率3e-4、batch=256、宽度512×4；每轮重新初始化Adam槽、继续当前网络参数，训练20,000步，固定PRNGKey(student_seed+100000*variant+round)。共享随机种子学生初训80,000步、每500步dev、8×256固定dev，dev_batch_seed=2026103031、最低dev误差且平局最早；离线网络noise输入维数8，原学生维数2。局部教师索引[8,10,12,13]标签保持local，其余转换到own local。

联合噪声转换以物理向量的等变旋转定义单元测试：global噪声向量随场景共同旋转，local噪声分量不随世界旋转。另报告固定公共键下global噪声不旋转时的输入变化，不能声称固定种子的逐样本旋转不变。global转换使用自身位置给出的方向，其他机器人方向仅由自身radius与相对位置重建；不得读取其他机器人目标。所有联合噪声一次抽normal(key,(1,8))，与教师trace第0步逐位比对。

蒸馏门固定旧2048个dev indices与keys（2026103033/2026103034）；三条件按任务全部执行。能力门沿用round30：两臂均<15%或均>98%排除，否则合格，unknown按下界。未通过门的变体不重训不替换。全局冻结全部最终checkpoint和蒸馏门，再跑能力门；之后全局冻结探测与选择；再打开新主评估标签。R5=16probe×F10/F13×200步，不读取任务标签；critic沿用原A3/A0/G_E权重与预处理。

公平调参右手规则（RHR-fair）冻结progress、40步(2秒)、0.25m/s，不重新调参。评估32state×seed0/1，七臂按任务；eta值复制round28原16菜单零起始索引10/13。bootstrap=5000，seed2026103032，先teacher配对controller层后state层成簇，保留两个seed及所有arm/variant配对；unknown下界/上界及保守差区间均报告。通过能力门的配对交集才进入对应主比较，同时单列门导致的样本流失。时间表/固定臂的事后最佳在每次bootstrap中重新选择。

只有本目录可写；DB只按官方追加流程；不修改现有源码，不改包环境，不使用git变更命令。总资源18CPU shard/2GPU shard。复现16条round28允许按任务强制重放，但明确其缓存已存在，不登记重复评估标签。所有新训练/评估必须先有阶段计划；评估另需官方preflight。源码保护或DB断言失败全局停止，其余阶段失败记缺口继续独立部分。视频不要求。

---

'''
 (HERE/'PREREGISTRATION.md').write_text(text+prompt.read_text())
 seal('PREREGISTRATION.receipt.json',[HERE/'PREREGISTRATION.md',HERE/'TRAIN_STATES.json',HERE/'TEACHERS.json',HERE/'STUDENTS.json',HERE/'TRAINING_CONFIG.json',prompt],before_new_training=True,before_new_rollouts=True)
 source_inventory('imports_preregister.json');print('preregistered',len(checks),[len(x) for x in selected],flush=True)
if __name__=='__main__':main()
