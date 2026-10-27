"""Per-scene tables with two rows per scene: classical (7 capable controllers) and learned (all learned instances with
complete records); 20 frozen validation + test scenes, states 8-23."""
import os,sys,json,collections,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));ROOT=os.path.dirname(HERE)
src=open(os.path.join(HERE,'balance.py')).read();cut=src.index('# ---------- (1) similarity');exec(compile(src[:cut],'balance_head','exec'))
SRC={'box':'自建（2.4 m 方框内随机起终点，机器人半径 0.12–0.16 m）；文献里相近的是 IMPC-DR 论文（Chen、Guo、Li，TAC 2024）的二维拥挤随机场景，但边界、尺寸、采样都不同','plus':'自建（4 机对心穿过十字路口）',
 'obst':'障碍几何取自 HJB-GNN 论文 lit_swap8_obstacles（Wang 等，RSS 2026，图 7(k)–(o)）；起点圆半径与机器人半径为我们改动','swap':'N=8、R=1.0 与 HJB-GNN 论文 lit_swap8_free 相同（图 1、图 7(a)–(e)）；其余 N 按 R=N/8 缩放'}
def r(vs,m,cs):return 100*np.mean([R[(v,key_of(v,m),c,s)] for v in vs for c in cs for s in range(8,24)])
note=lambda v:{'hs_obst_n08_R140_r005':'（= 论文原参数 R 1.4、半径 0.05）','xsc_hs_swap_n08':'（= 论文原参数 8 机、R 1.0）','xsc_hs_swap_n08_te':'（= 论文原参数 8 机、R 1.0）'}.get(v,'')
for f in ['box','plus','obst','swap']:
 print(f'\n### {FAM[f]}　来源：{SRC[f]}\n| 场景 | 组 | '+' | '.join(MN)+' |\n|---|---|'+'---:|'*len(M))
 for v in byfam[f]:
  tag=('验证' if v in VAL else '测试')
  for g,cs in [('经典',CL),('学习型',LE)]:print(f'| {v}（{tag}）{note(v)} | {g} | '+' | '.join(f'{r([v],m,cs):.1f}' for m in M)+' |')
 for g,cs in [('经典',CL),('学习型',LE)]:print(f'| **本族 5 个场景平均** | {g} | '+' | '.join(f'**{r(byfam[f],m,cs):.1f}**' for m in M)+' |')
