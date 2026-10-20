from common import *
import csv
G=guard()
FULL={'DM':'分布式模型预测控制（DM）','AS':'A*规划控制器（AS）','G':'高斯行为克隆策略（G）'}
def csvout(name,rows,fields):
 with (HERE/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 cap=read(HERE/'CAPABILITY_GATE.json');rows=[]
 for n,rr in cap['rates'].items():
  status='能力不足' if n in cap['excluded_low'] else '在16个先导状态的两臂均100%，无需修正' if n in cap['excluded_high'] else '进入主评估';a=rr['safety baseline'];b=rr['S500'];rows.append(dict(family=FULL[n.rstrip('0123456789')],name=n,safety_lower=a['lower'],safety_upper=a['upper'],schedule_lower=b['lower'],schedule_upper=b['upper'],status=status))
 fields=list(rows[0]);csvout('CAPABILITY_GATE.csv',rows,fields)
 for name,ids,title in [('CAPABILITY_EXCLUDED_LOW',cap['excluded_low'],'两个臂均低于15%的能力不足实例'),('CAPABILITY_EXCLUDED_HIGH',cap['excluded_high'],'两个臂均高于98%的无需修正实例（仅指16个先导状态）')]:
  rr=[r for r in rows if r['name'] in ids];csvout(name+'.csv',rr,fields);lines=['# '+title,'','| 控制器全名 | 实例 | 只加安全层成功率 | 前500步修正时间表成功率 |','|---|---|---:|---:|']
  for r in rr:lines.append(f"| {r['family']} | {r['name']} | {100*r['safety_lower']:.2f}% | {100*r['schedule_lower']:.2f}% |")
  if not rr:lines+=['','无。']
  (HERE/(name+'.md')).write_text('\n'.join(lines)+'\n')
 if (HERE/'RHR_GRID_RESULTS.json').exists():
  rr=[];lines=['# 逐族公平调参右手规则的完整24候选','', '触发窗口越大越晚；邻居阈值越小越晚。完整平局顺序已预注册。','', '| 控制器全名 | 候选 | 触发全名 | 窗口步数 | 邻居阈值（米） | 速度（米/秒） | 成功数/条数 | unknown | 成功率未知界 |','|---|---:|---|---:|---:|---:|---:|---:|---:|'];mode={'always':'一直开启的中心顺时针切向速度','slow':'自身低速触发的中心顺时针切向速度','progress':'自身无进展触发的中心顺时针切向速度','neighbor_right':'邻居接近触发的相对邻居右手速度'}
  for r in read(HERE/'RHR_GRID_RESULTS.json'):
   v=r['rule'];row=dict(family=FULL[r['family']],candidate_index=r['index'],trigger=mode[v['mode']],window_steps=v['steps'],neighbor_distance=v.get('distance'),speed=v['speed'],success=r['success'],n=r['n'],unknown=r['unknown'],lower=r['lower'],upper=r['upper']);rr.append(row);lines.append(f"| {row['family']} | {r['index']} | {row['trigger']} | {v['steps']} | {v.get('distance','—')} | {v['speed']} | {r['success']}/{r['n']} | {r['unknown']} | {100*r['lower']:.2f}%–{100*r['upper']:.2f}% |")
  csvout('RHR_GRID_TABLE.csv',rr,list(rr[0]));(HERE/'RHR_GRID_TABLE.md').write_text('\n'.join(lines)+'\n')
 for family in ['V0','ALL']:
  path=HERE/f'PAPER_TABLE_{family}.md'
  if path.exists():
   note='新三族使用24个右手规则候选；原学生按要求复用历史20候选的冻结规则与标签，未补跑新增4候选。合并分析仅DEVELOPMENT。'
   content=path.read_text()
   if note not in content:path.write_text(content+'\n'+note+'\n')
 print('EXTRA TABLES PASS')
if __name__=='__main__':main()
