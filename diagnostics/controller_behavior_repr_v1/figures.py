"""Standalone scientific PNG/PDF figures and CSV tables from saved results."""
from core import *
from data import Representations
import csv
sys.path.insert(0,str(R4/'plot_vendor'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def short(n):return 'base_u' if n=='base_u_macflow' else n.replace('base_u_plus_w_','plus_w_').replace('base_u_','').replace('_macflow','').replace('blend_','')
def table(path,headers,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:w=csv.writer(f);w.writerow(headers);w.writerows(rows)
def save(fig,key):
    folder=HERE/'figures';folder.mkdir(exist_ok=True)
    for suffix in ('png','pdf'):fig.savefig(folder/(key+'.'+suffix),bbox_inches='tight',dpi=180)
    plt.close(fig)
def info():
    s=read(HERE/'STAGE1.json')['results'];r=Representations();truth=read(HERE/'TRAIN_TRUTH.json');_,delta,_=r.behavior(r.names[:14]);ds=np.array([v['delta_success'] for v in truth]);colors=['#2563eb']*14+['#ea580c']*12
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for ax,(label,ids) in zip(axes,[('14 real Flow',list(range(14))),('12 blends',list(range(14,26))),('All 26',list(range(26)))]):
        ax.scatter(delta[ids,0],ds[ids],c=[colors[i] for i in ids],s=40)
        for i in ids:ax.annotate(short(r.names[i]),(delta[i,0],ds[i]),fontsize=6,xytext=(3,3),textcoords='offset points')
        ax.axhline(0,color='.65',lw=.7);ax.axvline(0,color='.65',lw=.7);ax.set(xlabel='Mean probe progress(eta13) - progress(eta10)',ylabel='TRAIN S(eta13) - S(eta10)',title=label)
    fig.suptitle('DEVELOPMENT: fixed 200-step behavior contrast');fig.tight_layout();save(fig,'I1_delta_progress_vs_delta_success')
    table(HERE/'tables/controller_truth_and_delta.csv',['controller','regime','best_K16','delta_success']+STATS,[[n,truth[i]['regime'],truth[i]['best_K16'],ds[i]]+delta[i].tolist() for i,n in enumerate(r.names)])
    for mode,results in s.items():
        rows=[x for x in results['I3'] if x['classifier']!='source_majority'];fig,axes=plt.subplots(7,2,figsize=(8,22))
        for ax,x in zip(axes.ravel(),rows):
            cm=np.array(x['confusion']);ax.imshow(cm,vmin=0,vmax=9,cmap='Blues');ax.set_xticks(range(4),x['classes']);ax.set_yticks(range(4),x['classes']);ax.set(title=f"{x['representation']} {x['classifier']} acc={x['accuracy']:.3f}",xlabel='Predicted regime',ylabel='TRAIN regime')
            for i in range(4):
                for j in range(4):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>4 else 'black')
        fig.suptitle('14-Flow LOO: DEVELOPMENT / '+mode);fig.tight_layout();save(fig,'I3_confusion_'+mode)
        table(HERE/f'tables/I2_{mode}.csv',['representation','bed','spearman','sign_accuracy'],[[x[k] for k in ('representation','bed','spearman','sign_accuracy')] for x in results['I2']])
        table(HERE/f'tables/I3_{mode}.csv',['representation','classifier','accuracy','delta_vs_majority'],[[x[k] for k in ('representation','classifier','accuracy','delta_vs_fixed_majority')] for x in results['I3']])
        table(HERE/f'tables/I1_{mode}.csv',['population','statistic','rho','ci_low','ci_high','bootstrap_degenerate'],[[x['population'],x['statistic'],x['rho'],*x['ci95'],x['bootstrap_degenerate']] for x in results['I1']])
    I4=s['main']['I4'];fig,axes=plt.subplots(1,4,figsize=(19,5))
    for ax,rep in zip(axes,['R3','R3-PCA2','R5-PCA2','R5-Dprog']):
        x=I4[rep];v=np.array(x['distances']);im=ax.imshow(v,cmap='viridis');labels=[short(n) for n in x['controllers']];ax.set_xticks(range(7),labels,rotation=65,ha='right',fontsize=7);ax.set_yticks(range(7),labels,fontsize=7);ax.set_title(rep);fig.colorbar(im,ax=ax,shrink=.65)
        table(HERE/f'tables/I4_distances_{rep}.csv',['controller']+x['controllers'],[[n]+row for n,row in zip(x['controllers'],x['distances'])]);table(HERE/f'tables/I4_neighbors_{rep}.csv',['controller']+[f'neighbor{i}' for i in range(1,7)],[[n]+x['nearest_ranking'][n] for n in x['controllers']])
    fig.suptitle('Descriptive distances: normalization/PCA fit to 14 real Flow');fig.tight_layout();save(fig,'I4_panel_distances')

STATS=['delta_progress','delta_projection','delta_minimum_distance','delta_mean_speed','delta_stalled_steps']
def endpoints(bed,sens=False):
    tag=bed+('_imputed' if sens else '');res=read(HERE/(tag+'_results.json'));ss=[s for s in res['summaries'] if s['primary']];covs=list(dict.fromkeys(s['coverage'] for s in ss));fig,axes=plt.subplots(1,3,figsize=(17,11),sharey=True);labels=[s['coverage']+' | '+s['representation']+' | '+s['policy'] for s in ss]
    y=np.arange(len(ss));color=['#2563eb' if s['coverage']==covs[0] else '#ea580c' for s in ss]
    for j,metric in enumerate(['E1','E2','E3']):
        ax=axes[j]
        for i,s in enumerate(ss):
            if metric=='E1':point=s['E1']['mean_lower']*100;ci=np.array(s['E1']['ci95_simultaneous'])*100;ax.plot(ci,[i,i],color=color[i]);ax.scatter(point,i,c=color[i],s=25)
            elif metric=='E2':
                for k,base in enumerate(('vs_src','vs_R0')):
                    point=s['E2'][base]['lower']*100;ci=np.array(s['E2'][base]['ci95_family_simultaneous'])*100;yy=i+(-.13 if k==0 else .13);ax.plot(ci,[yy,yy],color=color[i],alpha=1 if k==0 else .55);ax.scatter(point,yy,c=color[i],s=22,marker='o' if k==0 else 'x')
            else:
                ax.barh(i,s['E3']['possible_count'],color=color[i],alpha=.3);ax.barh(i,s['E3']['certain_count'],height=.35,color=color[i])
        if metric!='E3':ax.axvline(0,color='.6',lw=.8)
        if metric=='E1':ax.axvline(-2,color='red',linestyle='--',lw=.9)
        ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(y,labels,fontsize=7);axes[0].invert_yaxis();axes[0].set_title('E1 vs R0: global simultaneous CI (pp)');axes[1].set_title('E2 necessary: family simultaneous CI (pp)\no prior / x R0');axes[2].set_title('E3 catastrophes: possible / certain');fig.suptitle('DEVELOPMENT | '+tag);fig.tight_layout();save(fig,tag+'_factorial_E1_E2_E3')
    rr=[r for r in res['rows'] if r['primary']];targets=list(dict.fromkeys(r['controller'] for r in rr));fig,axes=plt.subplots(len(targets),1,figsize=(12,max(10,3.1*len(targets))),squeeze=False)
    for ax,n in zip(axes.ravel(),targets):
        rows=[r for r in rr if r['controller']==n];v=np.array([r['choice_histogram'] for r in rows]);ax.imshow(v,aspect='auto',vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(16));ax.set_yticks(range(len(rows)),[r['coverage']+' '+r['representation']+' '+r['policy'] for r in rows],fontsize=5);ax.set_title(short(n),fontsize=10);ax.set_xlabel('Selected eta index')
    fig.suptitle('Full per-target selection histograms | '+tag);fig.tight_layout();save(fig,tag+'_eta_histograms_all_targets')
    fields=['key','controller','necessary','source_prior_eta','success_lower','success_upper','gain_R0_lower','gain_R0_upper','gain_src_lower','gain_src_upper','catastrophe_possible','catastrophe_certain'];table(HERE/f'tables/{tag}_per_target.csv',fields+['eta'+str(j) for j in range(16)],[[r[f] for f in fields]+r['choice_histogram'] for r in res['rows']])
if __name__=='__main__':
    if sys.argv[1]=='info':info()
    else:endpoints(sys.argv[1],len(sys.argv)>2 and sys.argv[2]=='imputed')
