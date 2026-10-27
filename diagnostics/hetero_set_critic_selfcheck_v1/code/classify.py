"""Scene classes vs right-hand rule. 3 original controllers on all 48 scenes (train eval states 8-15, val/test 8-23);
eta reference = best of 49 menu eta chosen on states 0-7 (same rule as step5 own_best). Pipeline only on val/test."""
import json,collections,math,csv
import numpy as np
from run import KEYS
S={s['name']:s for s in json.load(open('scenes_all.json'))};Z=json.load(open('FINAL_FROZEN.json'))
VAL,TEST=Z['val'],Z['test'];TR=[s for s in S if s not in VAL and not s.endswith('_te')]
L={}
def add(sc,ci,si,k,s):L[(sc,ci,si,k)]=s
for f in ['design.jsonl','extra.jsonl','fresh.jsonl','test_gate.jsonl','test_menu.jsonl']:
 for d in map(json.loads,open(f)):
  k=d.get('key') or d['action'];add(d['scene'],d['controller'],d['state'],k[5:] if k.startswith('menu:') else k,d['success'])
for f in ['rhr_goal.jsonl','train_rhr.jsonl','val8.jsonl','test_rhr.jsonl','ens_val.jsonl','test_props.jsonl','test8.jsonl']:
 for d in map(json.loads,open(f)):k=d['key'];add(d['scene'],d['controller'],d['state'],k[5:] if k.startswith('menu:') else k,d['success'])
def rate(sc,k,cs,ss):
 x=[L[(sc,c,s,k)] for c in cs for s in ss];return 100*np.mean(x)
rows=[]
for sc in TR+VAL+TEST:
 split='train' if sc in TR else 'val' if sc in VAL else 'test';ev=range(8,16) if split=='train' else range(8,24)
 m=np.array([[np.mean([L[(sc,c,s,k)] for c in range(3)]) for k in KEYS] for s in range(8)]).mean(0);own=KEYS[int(m.argmax())]
 s=S[sc];fam=s['cluster'];n=s['config']['num_agents'];r=s['config']['agent_radius'];p=s['params']
 if fam=='box':crowd=n*math.pi*r*r/5.72;geo=f"随机起终点，机器人半径 {r}"
 elif fam=='plus':crowd=None;geo=f"路宽 {2*p['w']:.2f}"
 else:R=p['R'];crowd=None;geo=f"圈半径 {R}，相邻间距 {2*math.pi*R/n:.2f}"
 row=dict(scene=sc,split=split,family=fam,N=n,radius=r,geometry=geo,crowding=crowd,own_key=own,
  eta_ref=rate(sc,own,range(3),ev),best_fixed=rate(sc,'sE22',range(3),ev),rhr=rate(sc,'rhr',range(3),ev),rhr_goal=rate(sc,'rhr_goal',range(3),ev),safety=rate(sc,'safety',range(3),ev))
 if split!='train':
  pk=f"p{Z['choices'][sc]['ens16']}";U=[0,1,2,4,5,6,7]
  row.update(pipe3=rate(sc,pk,range(3),ev),pipe7=rate(sc,pk,U,ev),rhr7=rate(sc,'rhr',U,ev),rhr_goal7=rate(sc,'rhr_goal',U,ev),fixed7=rate(sc,'sE22',U,ev),safety7=rate(sc,'safety',U,ev))
 rows.append(row)
json.dump(rows,open('CLASSIFY_ROWS.json','w'),indent=1)

def cls(r):
 if r['family']=='box':return '1 随机起终点·拥挤方形场地'
 if r['family']=='plus':return '2 对心对穿·窄十字路口（路宽≤0.50）' if 2*S[r['scene']]['params']['w']<=0.5+1e-9 else '3 对心对穿·宽十字路口（路宽≥0.56）'
 return '4 对心对穿·中央静态障碍' if r['family']=='obst' else '5 对心对穿·无障碍圆圈'
G=collections.defaultdict(list)
for r in rows:G[cls(r)].append(r)
print('A. 3 个原 controller，全部 48 个场景（训练场景用状态 8-15，验证/测试用 8-23）')
print('| 场景类 | 场景数 | 每场景最优 η（状态0-7选） | 最好固定 η | 右手规则·绕中心 | 右手规则·按目标方向 | 只加安全投影 | 最优η 胜 绕中心右手 | 最优η − 绕中心右手（最小/中位/最大） |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---|')
for k in sorted(G):
 g=G[k];m=lambda f:np.mean([r[f] for r in g]);d=sorted(r['eta_ref']-r['rhr'] for r in g)
 print(f"| {k[2:]} | {len(g)} | {m('eta_ref'):.1f} | {m('best_fixed'):.1f} | {m('rhr'):.1f} | {m('rhr_goal'):.1f} | {m('safety'):.1f} | {sum(x>0 for x in d)}/{len(d)}（平 {sum(x==0 for x in d)}） | {d[0]:+.1f} / {np.median(d):+.1f} / {d[-1]:+.1f} |")
print()
print('B. 流水线（16 个提案）只在验证+测试 20 个场景上有，7 个合格 controller')
print('| 场景类 | 场景数 | 流水线 | 最好固定 η | 右手规则·绕中心 | 右手规则·按目标方向 | 只加安全投影 | 流水线 − 绕中心右手（逐场景） |')
print('|---|---:|---:|---:|---:|---:|---:|---|')
for k in sorted(G):
 g=[r for r in G[k] if 'pipe7' in r];m=lambda f:np.mean([r[f] for r in g])
 print(f"| {k[2:]} | {len(g)} | {m('pipe7'):.1f} | {m('fixed7'):.1f} | {m('rhr7'):.1f} | {m('rhr_goal7'):.1f} | {m('safety7'):.1f} | {', '.join(f'{r[chr(112)+chr(105)+chr(112)+chr(101)+chr(55)]-r[chr(114)+chr(104)+chr(114)+chr(55)]:+.1f}' for r in g)} |")
w=csv.writer(open('/home/zhihan/research/Basin_C1/diagnostics/hetero_set_critic_selfcheck_v1/SCENE_CLASSES_VS_RIGHT_HAND.csv','w'))
keys=['scene','split','family','N','radius','geometry','own_key','eta_ref','best_fixed','rhr','rhr_goal','safety','pipe7','fixed7','rhr7','rhr_goal7','safety7']
w.writerow(['class']+keys)
for k in sorted(G):
 for r in sorted(G[k],key=lambda r:r['scene']):w.writerow([k[2:]]+[r.get(x,'') if not isinstance(r.get(x),float) else round(r[x],2) for x in keys])
coll=collections.Counter((cls(dict(family=S[d['scene']]['cluster'],scene=d['scene']))) for d in map(json.loads,open('rhr_goal.jsonl')) if d['outcome']=='collision')
print('goal-relative collisions by class:',dict(coll))
