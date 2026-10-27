"""Right-hand-rule limitation scenes among the 32 frozen evaluation scenes (current pipeline):
crowded random start/goal boxes (8) and narrow crossroads (road width <= 0.50 m, 4).  Plain names, classical vs learned."""
import os,sys,json,collections,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));ROOT=os.path.dirname(HERE)
src=open(os.path.join(HERE,'all32_tables.py')).read();cut=src.index("G2 = [('经典', CL), ('学习型', LE)]");exec(compile(src[:cut],'all32_head','exec'))
MM=['pipe','sE22','rhr_goal','rhr','safety'];MMN=['流水线','最好固定 η','右手规则·按目标方向（文献的定义）','右手规则·绕中心（我们的环岛版本）','只加安全投影']
box=byfam['box'];narrow=[v for v in byfam['plus'] if 2*S[v]['params']['w']<=0.5+1e-9]
G2=[('经典',CL),('学习型',LE)]
for title,vs in [('拥挤方形场地（文献：拥挤、机器人多时右手规则会引出新的死锁）',box),('窄十字路口（文献：窄通道里右手规则会卡住或来回绕）',narrow)]:
 print(f'\n### {title}\n| 场景 | 组 | '+' | '.join(MMN)+' |\n|---|---|'+'---:|'*len(MM))
 for v in vs:
  for i,(g,cs) in enumerate(G2):print(f"| {nice(v) if i==0 else ''} | {g} | "+' | '.join(f'{r([v],m,cs):.1f}' for m in MM)+' |')
 for i,(g,cs) in enumerate(G2):print(f"| {'**平均**' if i==0 else ''} | {g} | "+' | '.join(f'**{r(vs,m,cs):.1f}**' for m in MM)+' |')
