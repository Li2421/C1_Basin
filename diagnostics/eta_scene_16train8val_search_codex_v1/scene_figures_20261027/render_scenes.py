"""Read archived geometry only; no simulation, sampling, training, or DB access."""
import os
from pathlib import Path
H = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(H / 'mplconfig')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

R = H.parents[2]
font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False,
                     'font.size': 11, 'pdf.fonttype': 42, 'svg.fonttype': 'path'})
sources = [R/'diagnostics/hetero_scene_pipeline_v1/SCENES.json',
           R/'diagnostics/critic_pool_supplement_v1/SCENES.json']
scenes, hashes = {}, {}
for p in sources:
    raw = p.read_bytes()
    hashes[str(p)] = hashlib.sha256(raw).hexdigest()
    for s in json.loads(raw):
        scenes[s['name']] = s
panels = [
 ('lit_swap8_obstacles', 'A  V 形静态障碍 · 八机对径交换',
  'N=8 ｜机器人半径 0.05 m｜起点圆半径 1.40 m', 1.75),
 ('xsc_tr_swap_n09', 'B  开放场地 · 九机对径交换',
  'N=9 ｜机器人半径 0.05 m｜起点圆半径 1.125 m', 1.45),
 ('xsc_tr_plus_w016', 'C  窄十字路口 · 四机交汇',
  'N=4 ｜机器人半径 0.05 m｜通道全宽 0.32 m', 1.75),
 ('lit_circle_five', 'D  圆形围场 · 五机随机任务',
  'N=5 ｜机器人半径 2.00 m｜围场内切半径 11 m', 12.6),
]
colors = ['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00',
          '#6B5FB5','#4D7785','#A67C52','#333333']

def draw(ax, s, title, subtitle, lim):
    state = s['states'][0]  # fixed first stored state; never select by outcome
    p, g = np.array(state['positions']), np.array(state['goals'])
    radius = s['config']['agent_radius']
    if 'bottleneck_config' in state:
        for x0,y0,x1,y1 in state['bottleneck_config'].get('extra_rectangles',[]):
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor='#dfe3e8',edgecolor='none'))
    for wall in s['walls']:
        w = np.array(wall)
        ax.plot(w[:,0],w[:,1],color='#414956',lw=2,solid_capstyle='round',zorder=2)
    for i,(start,goal) in enumerate(zip(p,g)):
        c = colors[i]
        ax.plot([start[0],goal[0]],[start[1],goal[1]],'--',color=c,lw=1,alpha=.38,zorder=1)
        a, b = start+.34*(goal-start), start+.46*(goal-start)
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color=c,lw=1.4),zorder=3)
        ax.add_patch(Circle(start,radius,facecolor=c,edgecolor='white',lw=.7,alpha=.86,zorder=4))
        ax.scatter(*goal,marker='*',s=70,facecolors='none',edgecolors=c,linewidths=1.3,zorder=5)
        # Labels are displaced radially; start and goal coordinates stay exact.
        direction = start / max(float(np.linalg.norm(start)),1e-12)
        q = start + direction*(radius+.055*lim)
        ax.text(*q,str(i+1),ha='center',va='center',color=c,fontsize=10,weight='bold',zorder=6)
    ax.set(xlim=(-lim,lim),ylim=(-lim,lim),aspect='equal',xlabel='x (m)',ylabel='y (m)')
    ax.set_title(title+'\n'+subtitle,loc='left',fontsize=12,pad=14,linespacing=1.7)
    ax.text(.98,.97,'权重 25%',ha='right',va='top',transform=ax.transAxes,
            fontsize=11,color='#164B76',bbox=dict(boxstyle='round,pad=.35',fc='#eaf2fa',ec='none'))
    ax.text(.02,.025,f"存档 state {state['index']}",transform=ax.transAxes,fontsize=9,color='#626975',
            bbox=dict(fc='white',ec='none',alpha=.85,pad=2))
    ax.grid(alpha=.12)
    for spine in ax.spines.values(): spine.set_color('#c8ced6')
    return dict(scene=s['name'],state_index=state['index'],weight=.25,
                config=s['config'],positions=state['positions'],goals=state['goals'],walls=s['walls'])

fig,axs = plt.subplots(2,2,figsize=(13,12))
records=[]
for ax,(name,title,subtitle,lim) in zip(axs.ravel(),panels):
    records.append(draw(ax,scenes[name],title,subtitle,lim))
fig.suptitle('当前四个等权锚点场景',fontsize=22,weight='bold',y=.98)
fig.text(.5,.941,'读取已有存档配置；每图展示一个固定代表状态，各图坐标尺度不同。',ha='center',fontsize=12,color='#52606e')
handles=[Line2D([],[],marker='o',color='none',markerfacecolor='#0072B2',markersize=10,label='实心圆：起点与实际机器人尺寸'),
         Line2D([],[],marker='*',color='none',markerfacecolor='white',markeredgecolor='#0072B2',markersize=12,label='同色星：目标'),
         Line2D([],[],linestyle='--',color='#0072B2',label='虚线 / 箭头：任务方向，非实际轨迹'),
         Line2D([],[],color='#414956',lw=2,label='深灰线：存档墙段')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.025),ncol=2,frameon=False,fontsize=11)
fig.text(.5,.011,'对径交换中部分目标与其他机器人的起点重合。未生成新状态或 rollout。',ha='center',fontsize=10,color='#626975')
fig.subplots_adjust(top=.87,bottom=.13,hspace=.38,wspace=.27)
for ext in ['png','pdf','svg']:
    fig.savefig(H/f'equal_weight_anchor_scenes.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
plt.close(fig)
for name,title,subtitle,lim in panels:
    fig,ax=plt.subplots(figsize=(6.3,6.3))
    draw(ax,scenes[name],title,subtitle,lim)
    fig.tight_layout()
    fig.savefig(H/f'{name}.png',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)
(H/'FIGURE_PROVENANCE.json').write_text(json.dumps(dict(
    scope='Four historical post-selected equal-weight anchors, not newly confirmed 16+8 scenes',
    new_rollouts=0,new_states=0,new_training=0,source_sha256=hashes,
    state_selection='first stored state, independent of outcomes',scenes=records,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),ensure_ascii=False,indent=2)+'\n')
print(str(H/'equal_weight_anchor_scenes.png'))
