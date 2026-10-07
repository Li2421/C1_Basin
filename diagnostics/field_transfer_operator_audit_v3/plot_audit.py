"""Standalone publication-exportable figures; no new simulation."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HERE=Path(__file__).resolve().parent
def read(p):return json.loads((HERE/p).read_text())

def main():
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,
                        'pdf.fonttype':42,'ps.fonttype':42})
    fig,axs=plt.subplots(3,2,figsize=(12,12),layout='constrained')
    scale=read('scale_probe_summary.json');colors=['#167a90','#bc5136']
    for ax,sc,name in zip(axs[0],['toy_give_way','ring_exchange'],['Toy','Ring']):
        xs=np.array([.01,.05,.1,.25,.5,1])
        for mode,col in zip(['flow','FF'],colors):
            q=np.array([scale[f'{sc}:{mode}:{a}']['relative_error']['quantiles'] for a in xs])
            ax.plot(xs,q[:,2],'-o',color=col,label='无投影 Flow' if mode=='flow' else 'FF')
            ax.fill_between(xs,q[:,1],q[:,3],color=col,alpha=.12)
        ax.set(xscale='log',yscale='log',xlabel='实际候选修正的比例 α',ylabel='一阶预测相对误差',title=f'{name}：零修正点的雅可比随幅度失效')
        ax.axhline(.25,color='#666',ls=':',lw=1);ax.legend(frameon=False);ax.grid(alpha=.2)
    geom=read('geometry.json');xx=np.arange(3);width=.32
    for ax,kind,ylabel in zip(axs[1],['gain','angle'],['局部方向增益 ‖MBj‖ / ‖Bj‖','方向夹角（度）']):
        for sc,shift,col,name in zip(['toy_give_way','ring_exchange'],[-width/2,width/2],colors,['Toy','Ring']):
            vals=[geom[sc+':FF'][f'basis{j}_{kind}']['quantiles'][2] for j in range(3)]
            ax.bar(xx+shift,vals,width,color=col,label=name)
        ax.set_xticks(xx,['目标方向 B1','Flow 正交残差 B2','相对位置 B3'])
        ax.set(ylabel=ylabel,title='FF：实际候选点处的方向变化中位数')
        ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    state=read('fixed_noise_state_dependence.json');ax=axs[2,0]
    for sc,shift,col,name in zip(['toy_give_way','ring_exchange'],[-width/2,width/2],colors,['Toy','Ring']):
        ax.bar(xx+shift,[state[sc+':'+ch]['relative_Frobenius_dispersion'] for ch in ['flow','TF','FF']],width,color=col,label=name)
    ax.set_xticks(xx,['无投影 Flow','TF','FF']);ax.set(ylabel='跨状态相对 Frobenius 离散度',title='固定物理 δ=0 和相同实际 Flow 噪声');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    ax=axs[2,1];integ=read('path_integral_checks.json')
    for sc,col,name in zip(['toy_give_way','ring_exchange'],colors,['Toy','Ring']):
        rows=[r for r in integ if r['scene']==sc]
        ax.scatter([r['single_relative_error'] for r in rows],[r['quadrature_errors']['33']['relative_error'] for r in rows],color=col,label=name,s=50)
    ax.plot([1e-4,10],[1e-4,10],':',color='#666')
    ax.set(xscale='log',yscale='log',xlabel='单个零修正雅可比的相对误差',ylabel='沿修正路径积分的相对误差',title='8 个固定索引案例：局部导数累积恢复有限响应')
    ax.legend(frameon=False);ax.grid(alpha=.2)
    fig.suptitle('场级修正的局部传递与有限幅度边界',fontsize=17)
    (HERE/'figures').mkdir(exist_ok=True)
    fig.savefig(HERE/'figures/mechanism_audit.png',dpi=180)
    fig.savefig(HERE/'figures/mechanism_audit.pdf')
    fig.savefig(HERE/'figures/mechanism_audit.svg')
    plt.close(fig)

if __name__=='__main__':main()
