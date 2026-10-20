from common import *
import csv
from deliver import PARAM_DESCRIPTION,dname,verdict,fmt,rate

def table(file,header,rows,lead=''):
 (HERE/(file+'.md')).write_text(lead+'\n\n| '+' | '.join(header)+' |\n|'+'|'.join(['---']*len(header))+'|\n'+'\n'.join('| '+' | '.join(str(x).replace('None','无合格源').replace('Frozen capability threshold not passed','冻结的三臂能力门未通过').replace('No budget-eligible instance passed the frozen capability gate','没有预算内实例通过冻结能力门').replace('FAIL_CONTROLLER_CAPABILITY','三臂能力门未通过').replace('GAP','缺口').replace('SUPPORTED','满足预注册标准').replace('REJECTED','未满足预注册标准').replace('INCONCLUSIVE','不确定').replace('|','／')for x in row)+' |'for row in rows)+'\n')
 with (HERE/(file+'.csv')).open('w')as f:w=csv.writer(f);w.writerow(header);w.writerows(rows)
def main():
 d=read(HERE/'D_RESULTS.json');sel=read(HERE/'RIGHT_HAND_SELECTED.json');grid=read(HERE/'RIGHT_HAND_GRID_RESULTS.json')['pools'];rows=[]
 for key,full in [('Flow','Ring Flow控制器'),('Gaussian','高斯行为克隆策略'),('MPC','分散式模型预测控制控制器')]:
  for q in grid[key]:r=q['rule'];rows.append([full,{'always':'持续触发','slow':'低速累计触发','progress':'窗口进度不足触发'}[r['mode']],r['steps'],r['speed'],100*q['lower'],100*q['upper'],q['unknown'],'已选择'if q['grid_index']==sel[key]['grid_index']else''])
 table('RIGHT_HAND_TUNING_TABLE',['控制器组','触发模式','持续步数','切向速度','成功率下界%','成功率上界%','未知数','冻结选择'],rows,'三组独立调参，各使用16个共享新状态、每实例一个随机未来。选择遵守成功率下界、速度、持续步数和触发模式的冻结顺序。模式字段为原规则名。')
 rows=[]
 for key,q in d.items():rows.append([dname(key),rate(q['methods']['S500 class']),rate(q['methods']['continuing right-hand']),fmt(q['paired']['schedule_minus_right_hand']['CI_pp']),verdict(q['paired']['schedule_minus_right_hand'],q.get('inferential_endpoint_eligible',True)),fmt(q['paired']['right_hand_minus_safety']['CI_pp']),str({{'both':'两者都救回','schedule_only':'仅时间表救回','right_hand_only':'仅持续右手规则救回','neither':'两者均未救回','unknown_comparator':'比较方法结果未知'}[k]:v for k,v in q['recovery_counts'].items()})])
 table('RIGHT_HAND_RESULTS',['子项','时间表成功率与95%置信区间','持续右手规则成功率与95%置信区间','时间表减持续右手规则95%置信区间（百分点）','结论','持续右手规则减安全投影基线95%置信区间（百分点）','安全基线已知失败中的救回数'],rows,'四个扰动半径未通过能力标准，其比较保留为描述性。完整逐条交叉表与未知分类单独交付。')
 e=read(HERE/'E_RESULTS.json');cap=read(HERE/'MPC_CAPABILITY_GATE.json');timing=read(HERE/'MPC_TIMING_REPORT.json');rows=[]
 for n,q in timing.items():
  for k,v in q['instances'].items():rows.append([n,PARAM_DESCRIPTION(k),v['two_state_warmed_CPU_step_seconds'],v['predicted_main_CPU_hours'],q['predicted_total_CPU_hours'],432,'可执行'if q['eligible']else q['reason']])
 table('MPC_TIMING_REPORT',['机器人数量','固定参数实例','两个状态的每步CPU秒均值','实例主评估预计CPU小时','两实例合计预计CPU小时','预算CPU小时','预算判断'],rows,'测时只作计算预算，不作为任何成功率结果。前20步为预热，之后取两状态均值的较大者；精确前缀证明未通过时使用九个全长轨迹的预算。')
 rows=[]
 for n,q in cap.items():
  for k,v in q.items():rows.append([n,PARAM_DESCRIPTION(k),str(v.get('max_success_bounds','无标签')),str({{'safety_baseline':'只加安全投影','F10':'全程固定η10修正','S13t2':'前七分之五时域η13后η10'}[a]:b for a,b in v.get('arms',{}).items()}),{'PASS':'通过','FAIL_CONTROLLER_CAPABILITY':'控制器能力未通过','GAP':'缺口','INCONCLUSIVE':'不确定'}[v['status']],v.get('reason','')])
 table('MPC_CAPABILITY_TABLE',['机器人数量','固定参数实例','三臂最大成功率下上界','各臂已知下上界','冻结门结果','缺口原因'],rows,'每实例16个独立能力门状态，三臂为只加安全投影、全程固定η10修正、前七分之五时域η13后η10时间表；已知下界达到20%才通过。原始臂名仅在逐臂机器记录中保留。')
 rows=[]
 for n,q in e['units'].items():
  t=q.get('clean_same_controller_transfer',{});source=n==str(e['source_N']);rows.append([e['source_N'],n,'、'.join(PARAM_DESCRIPTION(k)for k in t.get('instances',[])),fmt(t['gain_CI_pp'])if'gain_CI_pp'in t else'缺口',str(t.get('mean_regret_pp','缺口')),'源数量内交叉拟合参照'if source and 'gain_CI_pp'in t else t.get('transfer_status','缺口'),t.get('correction_status',q.get('reason',t.get('reason','缺口')))])
 table('MPC_TRANSFER_TABLE',['源机器人数量','目标机器人数量','源目标均通过门的同一参数实例','冻结选择减安全投影基线95%置信区间（百分点）','相对完整菜单的平均遗憾下上界（百分点）','选择迁移判断','修正可用性或缺口'],rows,'源数量内交叉拟合只作参照，不算跨数量迁移。旧Ring冻结选择模型没有可变数量Gap接口，保持不适用；这里的选择只用源数量数据按状态奇偶分半冻结。')
 rows=[]
 for n,instances in read(HERE/'E_PER_INSTANCE_RESULTS.json').items():
  for k,q in instances.items():
   t=q.get('clean_transfer',{});rows.append([e['source_N'],n,PARAM_DESCRIPTION(k),fmt(t['gain_CI_pp'])if'gain_CI_pp'in t else'缺口',str(t.get('mean_regret_pp','缺口')),'源数量内交叉拟合参照'if q.get('source_reference_only')else t.get('transfer_status','缺口'),t.get('correction_status',q.get('reason',t.get('reason','缺口')))])
 table('MPC_TRANSFER_PER_INSTANCE_TABLE',['源机器人数量','目标机器人数量','固定参数实例','冻结选择减安全投影基线95%置信区间（百分点）','平均遗憾下上界（百分点）','选择迁移判断','修正可用性或缺口'],rows,'逐实例检验只重抽样状态，不能外推到控制器总体；源数量内结果不构成跨数量迁移。')
 rows=[]
 for q in read(HERE/'MPC_FIVE_LAYER_EVIDENCE.json')['layers']:
  for k,v in q['controller_capability'].items():rows.append([q['N'],PARAM_DESCRIPTION(k),str(q['geometry_valid_states'])+'/32',str(q['single_robot_path_witness_states'])+'/32',str(q['joint_reference']['successes'])+'/32',{'PASS':'通过','FAIL_CONTROLLER_CAPABILITY':'控制器能力未通过','GAP':'缺口','INCONCLUSIVE':'不确定'}[v['status']],'未建立正式死锁解除证据'])
 table('MPC_FIVE_LAYER_TABLE',['机器人数量','参数实例','几何有效状态','单机器人有路见证','联合可解专家见证','控制器能力','死锁解除'],rows,'前三层复用32个主评估状态在原场景、地图、机器人半径和动力学下的既有见证。能力失败与计算预算缺口分开；正式死锁解除未建立，不能把缺口或超时称为死锁。')
 rows=[];labels=['只执行控制器动作','只加安全投影','全程固定η10','全程固定η13','全程固定η15','前七分之三时域η13后η10','前七分之五时域η13后η10','前七分之三时域η15后η10','前七分之五时域η15后η10']
 for n,q in e['units'].items():
  for k in ['DM1','DM3']:
   if k not in q.get('winning_arms',{}):rows.append([n,PARAM_DESCRIPTION(k),'缺口','缺口','缺口',cap[n][k].get('reason',cap[n][k]['status'])])
  for k,w in q.get('winning_arms',{}).items():rows.append([n,PARAM_DESCRIPTION(k),labels[w['fixed']],labels[w['schedule']],labels[w['full']],'估计界'+fmt(q['paired']['best_schedule_minus_fixed']['estimate_pp'])+'，95%置信区间'+fmt(q['paired']['best_schedule_minus_fixed']['CI_pp'])])
 table('MPC_ACTION_FAMILY_TABLE',['机器人数量','参数实例','样本内最佳固定动作','样本内最佳时间表','样本内完整菜单最佳','交叉拟合最佳时间表减固定修正'],rows,'最佳动作族为描述性结果；样本内最佳值与按状态交叉拟合的效果分别保存，未知保留上下界。')
if __name__=='__main__':main()
