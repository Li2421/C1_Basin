"""Nine user-picked scenes: per scene pick the controller (13 capable ones; gate failures excluded) with the largest
pipeline-minus-safety-only gain (ties -> learned first), then its first state 8-23 where safety-only fails and the
pipeline succeeds.  Writes VID9_PLAN.json."""
import os,sys,json
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));ROOT=os.path.dirname(HERE)
src=open(os.path.join(HERE,'all32_tables.py')).read();cut=src.index("G2 = [('经典', CL), ('学习型', LE)]");exec(compile(src[:cut],'all32_head','exec'))
PICK=['xsc_hs_box_n09_r013_f25_te','xsc_hs_box_n10_r013_f26_te','xsc_hs_box_n11_r014_f27_te','xsc_hs_plus_w020_te','xsc_hs_plus_w030_te',
      'hs_obst_n08_R140_r005','hs_obst_n08_R138_r048_te','xsc_hs_swap_n08_te','xsc_hs_swap_n10_te']
EXCL={'xsc_hs_box_n09_r013_f25_te':{13},'xsc_hs_box_n10_r013_f26_te':{13},'xsc_hs_box_n11_r014_f27_te':{8,13}}
NAMES={0:'集中式随机采样模型预测控制',1:'梯度下降模型预测控制',2:'分布式模型预测控制（打靶）',4:'交叉熵方法模型预测控制',5:'分布式交叉熵方法模型预测控制',
       6:'冲突搜索多机规划',7:'优先级时空 A* 规划',8:'多层感知机行为克隆·种子 0',9:'多层感知机行为克隆·种子 1',10:'扩散策略·种子 0',
       11:'扩散策略·种子 1',12:'Transformer 规划器·种子 0',13:'Transformer 规划器·种子 1'}
plan=[]
for i,v in enumerate(PICK,1):
 best=None
 for c in sorted(CL+LE,key=lambda c:(c<8,c)):
  if c in EXCL.get(v,()):continue
  gain=r([v],'pipe',[c])-r([v],'safety',[c])
  st=[s for s in range(8,24) if R[(v,'safety',c,s)]==0 and R[(v,key_of(v,'pipe'),c,s)]==1]
  if st and (best is None or gain>best[0]+1e-9):best=(gain,c,st[0])
 gain,c,s=best;k=CH[v]['ens16']
 plan.append(dict(order=i,scene=v,plain=nice(v),family=FN[S[v]['cluster']],controller=c,controller_name=NAMES[c],state=s,
   pipeline_key=f'p{k}',pipeline_eta=CH[v]['proposals'][k],gain=round(gain,1),safety_rate=round(r([v],'safety',[c]),1),pipeline_rate=round(r([v],'pipe',[c]),1)))
 print(i,FN[S[v]['cluster']],nice(v),'|',NAMES[c],'| state',s,'| pipeline',plan[-1]['pipeline_rate'],'safety',plan[-1]['safety_rate'])
json.dump(plan,open(os.path.join(HERE,'VID9_PLAN.json'),'w'),ensure_ascii=False,indent=1)
