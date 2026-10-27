"""Final plots read sealed numeric evidence; no simulator/controller calls."""
from experiment import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import font_manager
from states import walls,config

def main():
    receipt=read(HERE/'FINAL_NUMERIC_RESULTS.receipt.json');assert all(sha(p)==h for p,h in receipt['files'].items())
    installed={f.name for f in font_manager.fontManager.ttflist};chosen=next((f for f in ['Noto Sans CJK SC','Noto Sans CJK JP','WenQuanYi Zen Hei','SimHei'] if f in installed),None)
    if chosen:plt.rcParams['font.family']=chosen
    plt.rcParams['axes.unicode_minus']=False;folder=HERE/'figures';folder.mkdir(exist_ok=True);outputs=[]
    english={'lit_narrow_passage':'Narrow passage scene','lit_crowded_random':'Crowded random scene in two dimensions','lit_circle_five':'Five-robot random circular arena scene','lit_doorway':'Doorway scene'}
    for scene,name in NAMES.items():
        s=read(HERE/f'FLOW_STATES_{scene}.json')[0];c=config(scene,s['N']);fig,ax=plt.subplots(figsize=(6.4,5.6));ww=walls(scene)
        for wall in ww:ax.plot(wall[:,0],wall[:,1],color='black',lw=2)
        for i,(p,g) in enumerate(zip(s['positions'],s['goals'])):
            color=plt.cm.tab10(i%10);ax.add_patch(Circle(p,c.agent_radius,color=color,alpha=.35));ax.scatter(*p,color=color,s=14);ax.scatter(*g,color=color,s=85,marker='*');ax.text(p[0],p[1],str(i),ha='center',va='center',fontsize=8)
        if scene=='lit_circle_five':ax.set(xlim=(-11.5,11.5),ylim=(-11.5,11.5))
        elif scene=='lit_crowded_random':ax.set(xlim=(-.25,2.25),ylim=(-.25,2.25))
        elif scene=='lit_doorway':ax.set(xlim=(-1.4,2.4),ylim=(-1.6,1.6))
        else:ax.set(xlim=(-1.3,.65),ylim=(-.75,.75))
        title=name if chosen else english[scene]
        if scene=='lit_circle_five':title+='（仅几何；流程版按用户决定停止）' if chosen else ' (geometry only; flow study stopped by user)'
        ax.set_title(f'{title}\nN={s["N"]}, state index={s["index"]}');ax.set(xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal');ax.grid(alpha=.15);fig.tight_layout();path=folder/f'{scene}.png';fig.savefig(path,dpi=200);plt.close(fig);outputs.append(path)
    results=read(HERE/'MIGRATION_RESULTS.json');palette=['#44546a','#2f75b5','#c0504d','#e3a03b','#549b6a'];method_names=['Safety projection only','Fixed correction eta10','Fixed correction eta13','Human progress switch','Scene-trained correction']
    for population in ('fixed_full_inventory','learned_evaluation'):
        present=[(s,d[population]) for s,d in results['per_scene'].items() if d[population]]
        if not present:continue
        fig,ax=plt.subplots(figsize=(11,5));width=.16;methods=present[0][1]['methods']
        for j,m in enumerate(methods):
            for k,(scene,data) in enumerate(present):
                r=next(r for r in data['statistics'] if r['type']=='type_equal' and r['method']==m);x=k+(j-(len(methods)-1)/2)*width;point=100*r['success_rate'];lo,hi=100*np.asarray(r['ci95'])
                ax.bar(x,point,width*.9,color=palette[j],alpha=.85,label=method_names[j] if k==0 else None);ax.vlines(x,lo,hi,color='black',lw=1);ax.plot([x-.025,x+.025],[lo,lo],color='black',lw=1);ax.plot([x-.025,x+.025],[hi,hi],color='black',lw=1)
        labels=[NAMES[s] if chosen else english[s].replace(' scene','').replace(' in two dimensions','\nin two dimensions').replace(' random circular',' random\ncircular') for s,_ in present]
        ax.set_xticks(range(len(present)),labels);ax.set(ylim=(0,105),ylabel='Collision-free team success (%)',title='Type-equal, qualified instances; 95% clustered bootstrap intervals');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncols=3,frameon=False);ax.grid(axis='y',alpha=.15);fig.tight_layout();path=folder/f'{population}.png';fig.savefig(path,dpi=200,bbox_inches='tight');plt.close(fig);outputs.append(path)
    seal('FIGURES.receipt.json',[*outputs,HERE/'plot_results.py',HERE/'FINAL_NUMERIC_RESULTS.receipt.json'],simulation_evidence_only=True,controllers_physics_events_success_unchanged=True)

if __name__=='__main__':main()
