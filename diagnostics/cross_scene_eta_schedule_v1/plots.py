from core import *
sys.path.insert(0,str(R9/'vendor'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    r=read(HERE/'results.json');rates=np.asarray(r['scene_rates_lower'])*100;cis=np.asarray(r['scene_rate_CI95'])*100;fig,axs=plt.subplots(2,2,figsize=(14,9),sharey=True);labels=['F10','F13','F15','13→10 t1','13→10 t2','15→10 t1','15→10 t2']
    for i,ax in enumerate(axs.flat):
        ax.bar(np.arange(7),rates[i],color=['#4979a7']*3+['#4b967b']*2+['#bb985a']*2);ax.vlines(np.arange(7),cis[i,:,0],cis[i,:,1],color='black');ax.plot(np.arange(7),cis[i,:,0],'_',color='black');ax.plot(np.arange(7),cis[i,:,1],'_',color='black');ax.set_title(SCENES[i]);ax.set_xticks(np.arange(7),labels,rotation=20);ax.set_ylim(0,104);ax.set_ylabel('Success lower point (%)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Prospective E8 Flow-reference programs: 3 controllers, 40 states/scene, Q4\n95% state-cluster bootstrap intervals; unknown upper bounds stored separately');fig.tight_layout();fig.savefig(HERE/'four_scene_seven_arms.png',dpi=180);fig.savefig(HERE/'four_scene_seven_arms.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(4);width=.35
    for j,method in enumerate(['fixed','expanded']):
        rows=[a for a in r['X2']['scenes'] if a['method']==method];y=np.array([a['regret'] for a in rows])*100;lo=np.array([a['CI95'][0] for a in rows])*100;hi=np.array([a['CI95'][1] for a in rows])*100;xx=x+(j-.5)*width;ax.bar(xx,y,width,label=method,color=['#4979a7','#4b967b'][j]);ax.vlines(xx,lo,hi,color='black');ax.plot(xx,lo,'_',color='black');ax.plot(xx,hi,'_',color='black')
    ax.axhline(5,color='gray',linestyle='--',label='5pp maximum-regret gate');ax.set_xticks(x,['Toy','DB','Four','Ring']);ax.set_ylabel('Held-scene regret (pp)');ax.set_title('LOSO source-average argmax: source choice reselected in bootstrap');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(HERE/'X2_loso_regrets.png',dpi=180);fig.savefig(HERE/'X2_loso_regrets.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5));colors=['#4979a7','#bf6574','#4b967b','#bb985a']
    for ax,a in zip(axs,[13,15]):
        markers=['o','s','^'];sizes=[50,80,110]
        for i,sc in enumerate(SCENES):
            rows=[q for q in r['X3']['effects'] if q['scene']==sc and q['early_eta']==a]
            for q in rows:
                j=NAMES.index(q['controller']);ax.scatter(100*q['E'],100*q['L'],edgecolors=colors[i],facecolors='none',marker=markers[j],s=sizes[j],linewidths=1.5)
        ax.axhline(10,color='gray',linestyle='--');ax.axvline(10,color='gray',linestyle='--');ax.set(xlabel='E = S − F10 (pp)',ylabel=f'L = S − F{a} (pp)',title=f'Early eta{a}, switch at t2');ax.grid(alpha=.2)
    from matplotlib.lines import Line2D
    scene_handles=[Line2D([0],[0],marker='o',color=colors[i],linestyle='none',label=label) for i,label in enumerate(['Toy','DB','Four','Ring'])]
    family_handles=[Line2D([0],[0],marker=marker,color='black',markerfacecolor='none',linestyle='none',label=label) for marker,label in zip(['o','s','^'],['isotropic','anisotropic','coupled'])]
    axs[0].legend(handles=family_handles,fontsize=8,loc='upper right',title='Operator');axs[1].legend(handles=scene_handles,fontsize=8,loc='center right',title='Scene');fig.tight_layout();fig.savefig(HERE/'X3_phase_effects.png',dpi=180);fig.savefig(HERE/'X3_phase_effects.pdf');plt.close(fig)
if __name__=='__main__':main()
