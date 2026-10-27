"""Standalone figures rendered solely from frozen scene and result artifacts."""
import guard
from guard import HERE
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import font_manager

def main():
    fonts=[p for p in font_manager.findSystemFonts()if any(k in p.lower()for k in('notosanscjk','wqy','droidsansfallback'))]
    if fonts:plt.rcParams['font.family']=['DejaVu Sans',font_manager.FontProperties(fname=fonts[0]).get_name()]
    plt.rcParams['svg.fonttype']='none'
    frozen=json.loads((HERE/'SCENES_FROZEN.json').read_text());fig,axes=plt.subplots(5,4,figsize=(15,18),constrained_layout=True)
    for ax,e in zip(axes.flat,frozen['entries']):
        ax.set_title(e['title'],fontsize=10)
        if 'scene'not in e:
            gap='纵队几何待确认\n未纳入正式实验'if e['type_id']==2 else'指定几何不可实现\n未运行'
            ax.text(.5,.5,gap,ha='center',va='center',transform=ax.transAxes);ax.axis('off');continue
        s=e['scene'];state=s['states'][0];p=np.asarray(state['positions']);g=np.asarray(state['goals']);r=s['config']['agent_radius'];colors=plt.cm.tab20(np.linspace(0,1,len(p)))
        for a,b in np.asarray(s['walls']).reshape(-1,2,2):ax.plot([a[0],b[0]],[a[1],b[1]],color='black',linewidth=1.4)
        for i,(a,b)in enumerate(zip(p,g)):
            ax.add_patch(Circle(a,r,facecolor=colors[i],edgecolor='black',linewidth=.4))
            ax.scatter(*b,color=colors[i],marker='x',s=22)
            path=np.asarray(s['initial_paths'][0][i]);ax.plot(path[:,0],path[:,1],color=colors[i],alpha=.35,linewidth=.7)
        bounds=s['bounds'];span=np.vstack([p,g]);lo=np.minimum(span.min(0)-.2,bounds[:2]);hi=np.maximum(span.max(0)+.2,bounds[2:])
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.tick_params(labelsize=7)
        ax.set_xlabel(f"N={len(p)}, r={r}m, route={s['route_guidance']}",fontsize=8)
    for ax in list(axes.flat)[len(frozen['entries']):]:ax.axis('off')
    fig.savefig(HERE/'scene_overview.svg');fig.savefig(HERE/'scene_overview.png',dpi=160);plt.close(fig)
    if(HERE/'SCENE_DESCRIPTORS.json').exists():
        data=json.loads((HERE/'SCENE_DESCRIPTORS.json').read_text());matrix=np.array([[np.nan if x is None else x for x in row]for row in data['distance_matrix']])
        fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(matrix,cmap='viridis');fig.colorbar(im,ax=ax,label='Standardized feature distance');ax.set_xticks(range(17),range(1,18));ax.set_yticks(range(17),range(1,18));ax.set_title('Geometry and traffic descriptor distances');fig.tight_layout();fig.savefig(HERE/'scene_distances.svg');plt.close(fig)
    if(HERE/'VERDICT.json').exists():
        verdict=json.loads((HERE/'VERDICT.json').read_text());families=verdict['families'];ci=json.loads((HERE/'CONFIDENCE_INTERVALS.json').read_text());y=np.arange(len(families))
        fig,ax=plt.subplots(figsize=(15,11))
        for mode,offset,color,label in [('oracle',-.16,'#31688e','冻结参数池 Oracle@16'),('fixed',.16,'#35b779','固定参数')]:
            rates=np.array([f[mode]['success_rate']for f in families]);intervals=np.array([ci[f"family_{f['index']}_{mode}"]['bootstrap95']for f in families]);errors=np.maximum(0,np.array([rates-intervals[:,0],intervals[:,1]-rates]))
            ax.barh(y+offset,rates,height=.28,color=color,label=label,xerr=errors,error_kw=dict(ecolor='#222222',capsize=2,linewidth=.8))
            for yy,rate,f in zip(y+offset,rates,families):ax.text(min(1.06,rate+.035),yy,f"{f[mode]['success']}/{f[mode]['n']}",va='center',fontsize=8)
        ax.set_yticks(y,[' +\n'.join(f['column_names'])for f in families],fontsize=8);ax.invert_yaxis();ax.set_xlim(0,1.18);ax.set_xticks(np.linspace(0,1,6));ax.set_xlabel('任务成功率（每族120案例，60个独立物理初态）');ax.set_title('冻结修正族：独立留出；误差线为地图内物理初态分块95%区间');ax.legend(loc='lower right');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);fig.subplots_adjust(left=.38,right=.98,bottom=.09,top=.94);fig.savefig(HERE/'basis_dimension_results.svg');fig.savefig(HERE/'basis_dimension_results.png',dpi=160);plt.close(fig)
    if(HERE/'PAPER_RESULTS.json').exists():
        results=json.loads((HERE/'PAPER_RESULTS.json').read_text());registry=json.loads((HERE.parents[1]/'diagnostics/paper_baselines_20261028/PAPER_REGISTRY.json').read_text());names={x['id']:x['name']for x in registry['allowed']};methods=list(results);yy=np.arange(len(methods));left=np.zeros(len(methods))
        fig,ax=plt.subplots(figsize=(15,8))
        for field,color,label in [('success','#35b779','成功'),('collision','#d55e00','碰撞'),('timeout','#e6ab02','超时'),('numerical_unknown','#9e9e9e','求解/规划/接口unknown')]:
            values=np.array([results[m]['overall'][field]/max(1,results[m]['overall']['n'])for m in methods]);ax.barh(yy,values,left=left,color=color,label=label);left+=values
        labels=[names[m.split('__')[0]]+('\n本轮开发参数'if'__development_selected'in m else'\n上轮冻结参数')for m in methods]
        ax.set_yticks(yy,labels,fontsize=8);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel('60个共同物理留出初态中的比例（每个profile）');ax.set_title('论文完整机制端口：共同SI环境；原生控制/安全，无C1外部路线引导');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.09),ncol=4,fontsize=8);fig.subplots_adjust(left=.38,right=.98,bottom=.16,top=.91);fig.savefig(HERE/'paper_baseline_results.svg');fig.savefig(HERE/'paper_baseline_results.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
