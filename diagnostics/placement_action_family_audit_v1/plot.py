"""Publication-style static scientific evidence, entirely from saved audit results."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from analyze import HERE,read,DISPLAY

def main():
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'font.size':10,'axes.unicode_minus':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    g=read(HERE/'geometry.json');pairs=read(HERE/'paired_geometry.json');families=read(HERE/'sampled_families.json')
    basin=read(HERE/'basin_groups.json');seq=read(HERE/'selected_sequence_examples.json')
    scenes=['toy_give_way','ring_exchange'];names=['Toy','Ring'];colors=['#285f8f','#e38635','#4d8b68','#8055a5']
    fig,axs=plt.subplots(2,3,figsize=(15.6,9.2),layout='constrained')
    ax=axs[0,0];values=np.array([[g[f'{sc}:initial:{ch}'][f'basis{i}_gain']['quantiles'][2] for i in range(3)] for sc in scenes for ch in DISPLAY])
    im=ax.imshow(values,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    for (y,x),v in np.ndenumerate(values):ax.text(x,y,f'{v:.3f}',ha='center',va='center',color='white' if v>.6 else '#172633')
    ax.set_xticks(range(3),['目标方向','侧向基','相对位置基']);ax.set_yticks(range(8),[s+' / '+ch for s in names for ch in DISPLAY])
    ax.axhline(3.5,color='white',lw=2);ax.set_title('(a) 局部基方向增益：中位数')
    fig.colorbar(im,ax=ax,fraction=.03,pad=.02)
    ax=axs[0,1]
    for i,(sc,ch) in enumerate((sc,ch) for sc in scenes for ch in DISPLAY):
        rec=g[f'{sc}:initial:{ch}'];left=0
        for rank,col in [(1,'#d9dce2'),(2,'#91b9d4'),(3,'#285f8f')]:
            v=rec['rank_resolved_counts'].get(str(rank),0)/rec['reliable']
            ax.barh(i,v,left=left,color=col,label=f'秩 {rank}' if i==0 else None);left+=v
    ax.set_yticks(range(8),[s+' / '+ch for s in names for ch in DISPLAY]);ax.invert_yaxis();ax.set_xlim(0,1)
    ax.set_xlabel('可靠局部估计中的比例');ax.set_title('(b) J_eta 的可分辨秩');ax.legend(ncol=3,fontsize=9,loc='lower left')
    ax=axs[0,2]
    for i,ch in enumerate(DISPLAY):
        val=[families['individual'][sc+':'+ch]['diameter']['quantiles'][2] for sc in scenes]
        ax.bar(np.arange(2)+(i-1.5)*.18,val,width=.17,color=colors[i],label=ch)
    ax.set_xticks([0,1],names);ax.set_ylabel('联合物理动作的欧氏距离');ax.set_title('(c) 16 个候选动作族的直径：中位数');ax.legend(ncol=4,fontsize=9)
    ax=axs[1,0]
    contrasts=['TT-TF','TT-FT','TT-FF']
    for si,sc in enumerate(scenes):
        yy=np.array([pairs[f'{sc}:initial:{ch}']['principal_max']['quantiles'][[1,2,3][j]] for ch in contrasts for j in range(3)]).reshape(3,3)
        ax.errorbar(np.arange(3)+(si-.5)*.13,yy[:,1],yerr=np.stack([yy[:,1]-yy[:,0],yy[:,2]-yy[:,1]]),fmt='o',capsize=3,
                    color=['#285f8f','#d47625'][si],label=names[si])
    ax.set_xticks(range(3),[c.replace('-',' → ') for c in contrasts]);ax.set_ylim(-3,93)
    ax.set_ylabel('最大主角（度）');ax.set_title('(d) 干预子空间：中位数与 10–90%');ax.legend()
    ax.text(.02,.02,'不同维数须结合秩表解读',transform=ax.transAxes,fontsize=9,color='#555')
    ax=axs[1,1]
    for i,(sc,ch) in enumerate((sc,ch) for sc in scenes for ch in contrasts):
        rec=basin[sc+':'+ch]['counts'];ax.bar(i,rec.get('rescue',0),color='#4b8767',label='新增成员' if i==0 else None)
        ax.bar(i,-rec.get('loss',0),color='#b85348',label='丢失成员' if i==0 else None)
        ax.text(i,rec.get('rescue',0)+2,str(rec.get('rescue',0)),ha='center',fontsize=9)
        ax.text(i,-rec.get('loss',0)-3,str(rec.get('loss',0)),ha='center',va='top',fontsize=9)
    ax.axhline(0,color='#333',lw=.6);ax.set_xticks(range(6),[s+'\n'+ch.replace('-','→') for s in names for ch in contrasts],fontsize=9)
    ax.set_ylim(-135,115);ax.set_ylabel('state–eta cells');ax.set_title('(e) 15/16 稳健成功盆的成员变化');ax.legend(fontsize=9,loc='lower right')
    ax.text(.02,.02,'Ring 涉 FT 另有 10 个未决 cell',transform=ax.transAxes,fontsize=8,color='#555')
    ax=axs[1,2]
    z=np.load(HERE/'selected_sequence_arrays.npz');r=seq[13];dist=z['example13_action_difference']
    ax.semilogy(np.arange(len(dist))*.05,np.maximum(dist,1e-17),color='#8055a5')
    ax.axvline(.45,color='#555',lw=.8,ls='--');ax.set_ylim(1e-17,1)
    ax.set_xlabel('共同执行前缀的物理时间（秒）');ax.set_ylabel('||u_FT − u_TT||')
    ax.set_title('(f) 初始族相同，闭环结果不同')
    ax.text(.28,.5,'Toy 820002 / eta[4] / seed 2\n第 9 步开始明显分离\nTT：死锁；FT：成功',transform=ax.transAxes,va='top',fontsize=10)
    fig.suptitle('固定控制器下，placement 改变有限动作族及其闭环展开',fontsize=16)
    out=HERE/'figures';out.mkdir(exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'action_family_audit.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)
    (HERE/'figure_caption.txt').write_text('四 placement 审计。a–d 为 Toy/Ring 各 24 个初态、16 个 eta、16 个配对噪声。a、b、d 仅统计通过差分稳定性检查的局部估计；b 的秩阈值同时考虑三档差分差异；d 中零夹角不能排除维数损失。c 是实际 16 个候选的离散像直径，不是连续族的体积。e 是缓存 Q16 标签，Ring 涉 FT 的 10 个未决 cell 保留。f 从既有完整轨迹读取，曲线只画两条轨迹均未终止的共同前缀；案例条件性选择用于反例展示。所有控制器、eta、实际 noise key 均保持原注册设置，未生成新 rollout。\n')
if __name__=='__main__':main()
