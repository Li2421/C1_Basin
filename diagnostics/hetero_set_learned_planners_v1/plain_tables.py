"""Same per-scene tables as group_tables.py, with plain-language scene names (no code names)."""
import os,sys,json,collections,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));ROOT=os.path.dirname(HERE)
src=open(os.path.join(HERE,'balance.py')).read();cut=src.index('# ---------- (1) similarity');exec(compile(src[:cut],'balance_head','exec'))
HEAD={'box':'拥挤方形场地：2.4 m × 2.4 m 的方框里，每台机器人的起点和终点都是随机的，机器人比较大（自建场景）',
 'plus':'十字路口：4 台机器人从四个路口同时出发，穿过路口去对面（自建场景）',
 'obst':'带障碍的圆圈对穿：8 台机器人站在圆周上，同时走到正对面，圆周内侧有 4 个 V 形障碍（障碍取自 HJB-GNN 论文）',
 'swap':'无障碍圆圈对穿：机器人站在圆周上，同时走到正对面，没有障碍（8 台、圆半径 1 m 的是 HJB-GNN 论文的设置）'}
def nice(v):
 p=S[v]['params'];f=S[v]['cluster'];tag='验证' if v in VAL else '测试';paper=''
 if f=='box':name=f"{p['N']} 台机器人，半径 {p['r']:.2f} m"
 elif f=='plus':name=f"路宽 {2*p['w']:.2f} m"+('（窄）' if 2*p['w']<=0.5 else '')
 elif f=='obst':
  name=f"起点圆半径 {p['R']:.2f} m，机器人半径 {p['r']:g} m";paper='，论文原设置' if abs(p['R']-1.4)<1e-9 and abs(p['r']-0.05)<1e-9 else ''
 else:
  name=f"{p['N']} 台机器人，圆半径 {p['R']:g} m";paper='，论文原设置' if p['N']==8 else ''
 return f'{name}（{tag}{paper}）'
def r(vs,m,cs):return 100*np.mean([R[(v,key_of(v,m),c,s)] for v in vs for c in cs for s in range(8,24)])
MM=M[:-1];MMN=MN[:-1]
for f in ['box','plus','obst','swap']:
 print(f'\n### {HEAD[f]}\n| 场景 | 组 | '+' | '.join(MMN)+' |\n|---|---|'+'---:|'*len(MM))
 for v in byfam[f]:
  for i,(g,cs) in enumerate([('经典',CL),('学习型',LE)]):print(f"| {nice(v) if i==0 else ''} | {g} | "+' | '.join(f'{r([v],m,cs):.1f}' for m in MM)+' |')
 for i,(g,cs) in enumerate([('经典',CL),('学习型',LE)]):print(f"| {'**5 个场景平均**' if i==0 else ''} | {g} | "+' | '.join(f'**{r(byfam[f],m,cs):.1f}**' for m in MM)+' |')
if '--codes' in sys.argv:
 print('\n文件里的代号对照：'+'；'.join(f'{nice(v)} = {v}' for f in ['box','plus','obst','swap'] for v in byfam[f]))
