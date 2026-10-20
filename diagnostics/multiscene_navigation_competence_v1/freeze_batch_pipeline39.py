from common import *
import ast
for n in ['collect39.py','batch_fields39.py','batch_collect39.py','policies39.py','submit_training39.py']:ast.parse((HERE/n).read_text())
for n in ['BATCH_FIELDS39_TEST.json','BATCH_NATIVE39_TEST.json','OBSERVED_POLICY39_TEST.json']:assert read(HERE/n)['passed']
text='''示范采集之前采用独立单机器人回合的批量静态场查询以减少Python调度开销。每组最多32个同几何回合，目标场各自固定、随机键各自按状态和步数折叠、观测不含其他回合的机器人，所有物理step与终止逐个使用原单机器人环境。三个族的跟随点、39维观测、专家动作与逐回合参考逐位一致；独立撞墙、到达与超时的完整轨迹、动作、标签、余量和终止逐位一致；两类策略使用预先计算观测与原观测入口的动作逐位一致。当前运行的专家门worker和源码身份保持不变。尚未采集任何本轮训练示范、没有策略训练或导航留出评估。训练超参数、种子、标签选择与三轮纠偏规则不变；旧标量参考保留。采集身份纳入批量代码及原网络定义哈希，所有回合仍走官方逐条身份与journal，不把独立回合当成多机器人实验。GPU单任务错误写缺口，继续其余已冻结训练实例，不替换或调参。'''
(HERE/'PREREGISTRATION_BATCH_COLLECTION.md').write_text(text+'\n')
old=read(HERE/'TRAINING_PIPELINE39_V6.receipt.json');names=list(old['files'])+[str(HERE/n) for n in ['PREREGISTRATION_BATCH_COLLECTION.md','batch_fields39.py','batch_collect39.py','test_batch_fields39.py','test_batch_native39.py','test_observed_policy39.py','BATCH_FIELDS39_TEST.json','BATCH_NATIVE39_TEST.json','OBSERVED_POLICY39_TEST.json']]
write(HERE/'TRAINING_PIPELINE39_V7.receipt.json',dict(utc=utc(),before_demonstration_collection=True,before_training=True,expert_gate_worker_unchanged=True,files={n:sha(n) for n in names}))
write(HERE/'TRAINING_PIPELINE39_CURRENT.json',dict(receipt='TRAINING_PIPELINE39_V7.receipt.json',sha256=sha(HERE/'TRAINING_PIPELINE39_V7.receipt.json')))
from freeze_guard39 import verify
verify()
for n in ['REPORT.md','HANDOFF.md','deliver_failed_expert_pilot.py']:
 p=HERE/n;p.write_text(p.read_text().replace('TRAINING_PIPELINE39_V6.receipt.json','TRAINING_PIPELINE39_V7.receipt.json'))
print('independent batched data pipeline frozen before demonstration collection')
