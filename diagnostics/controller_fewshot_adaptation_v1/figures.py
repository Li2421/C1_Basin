from core import *
import csv
sys.path.insert(0,str(R4/'plot_vendor'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def short(n):return 'base_u' if n=='base_u_macflow' else n.replace('base_u_plus_w_','plus_w_').replace('base_u_','').replace('_macflow','')
def save(fig,name):
    d=HERE/'figures';d.mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(d/(name+'.'+ext),dpi=170,bbox_inches='tight')
    plt.close(fig)
def csvtable(name,headers,rows):
    d=HERE/'tables';d.mkdir(exist_ok=True)
    with (d/name).open('w') as f:w=csv.writer(f);w.writerow(headers);w.writerows(rows)

def main():
    gate();res=[read(HERE/'endpoints'/f'main_B{B}_with_E3_CI.json') for B in BUDGETS];eff=read(HERE/'SAMPLE_EFFICIENCY.json');names=read(HERE/'PROTOCOL.json')['controller_order'];x=np.arange(7);colors=plt.cm.tab10(np.arange(10));methodcolors={m:colors[i//2%10] for i,m in enumerate(METHODS[1:])}
    for safe in (False,True):
        methods=[m for m in METHODS[1:] if m.endswith('-safe')==safe];fig,axes=plt.subplots(6,3,figsize=(17,22),sharex=True)
        for st,strategy in enumerate(STRATEGIES):
            for m in methods:
                key=strategy+'|'+m;point=[next(s for s in r['summaries'] if s['key']==key) for r in res];rows=[[q for q in r['rows'] if q['key']==key] for r in res];color=methodcolors[m]
                axes[0,st].plot(x,[np.mean([q['success_lower'] for q in rr if q['target'] in (1,4)])*100 for rr in rows],marker='o',color=color,label=m)
                axes[1,st].plot(x,[np.mean([q['success_lower'] for q in rr if q['target'] not in (1,4)])*100 for rr in rows],marker='o',color=color,label=m)
                for gi,group in enumerate([(1,4),(0,2,3,5,6)]):
                    axes[2+gi,st].plot(x,[np.mean([q['possible_catastrophe_rate'] for q in rr if q['target'] in group])*100 for rr in rows],marker='o',color=color,label=m)
                    axes[4+gi,st].plot(x,[np.mean([q['any_possible_catastrophe'] for q in rr if q['target'] in group])*100 for rr in rows],marker='o',color=color,label=m)
            p0=[r for r in res[0]['rows'] if r['key']=='P0']
            axes[0,st].axhline(np.mean([q['success_lower'] for q in p0 if q['target'] in (1,4)])*100,color='black',ls=':',label='P0')
            axes[1,st].axhline(np.mean([q['success_lower'] for q in p0 if q['target'] not in (1,4)])*100,color='black',ls=':')
            for row in range(6):axes[row,st].grid(alpha=.2);axes[row,st].set_xticks(x,BUDGETS);axes[row,st].set_xlabel('Target label budget B')
            axes[0,st].set_title('Acquisition '+strategy);axes[0,st].legend(fontsize=7,ncol=2)
        for ax,label in zip(axes[:,0],['Necessary-target success lower (%)','Nonnecessary-target success lower (%)','Necessary-target mean possible catastrophe (%)','Nonnecessary-target mean possible catastrophe (%)','Necessary targets with any sampled catastrophe (%)','Nonnecessary targets with any sampled catastrophe (%)']):ax.set_ylabel(label)
        fig.suptitle('DEVELOPMENT | '+('safe' if safe else 'raw')+' adaptation | means over 200 acquisitions; CI separately reported');fig.tight_layout();save(fig,'learning_curves_'+('safe' if safe else 'raw'))
    # Exact simultaneous effect bands for the primary feasible safe strategy.
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for m in ['E-safe','H-safe','C-R0-safe','C-F2-safe','C-ctl-R0-safe','C-ctl-F2-safe']:
        ss=[next(s for s in r['summaries'] if s['key']=='R|'+m) for r in res]
        for ax,metric in zip(axes,['E1','E2']):
            pt=np.array([s[metric]['lower'] for s in ss])*100;ci=np.array([s[metric]['ci95_simultaneous'] for s in ss])*100;ax.plot(x,pt,marker='o',label=m);ax.fill_between(x,ci[:,0],ci[:,1],alpha=.08);ax.set_xticks(x,BUDGETS);ax.grid(alpha=.2);ax.axhline(0,color='black',lw=.7);ax.set_xlabel('B');ax.set_ylabel('Gain relative to source prior (pp)');ax.set_title(metric)
    axes[0].axhline(-2,color='red',ls='--',lw=.8);axes[0].legend(fontsize=7);fig.suptitle('R acquisition: budget-family simultaneous intervals');fig.tight_layout();save(fig,'R_safe_simultaneous_effects')
    # Point budget minima, with right-censored cells retained as >128.
    fig,ax=plt.subplots(figsize=(12,8));methods=METHODS[1:];values=np.full((len(methods),3),np.nan)
    for i,m in enumerate(methods):
        for j,st in enumerate(STRATEGIES):
            v=next(v for v in eff['budgets'] if v['key']==st+'|'+m)['B_SPT'];values[i,j]=v if isinstance(v,int) else 160
    ax.imshow(values,aspect='auto',cmap='YlGnBu',vmin=0,vmax=160);ax.set_xticks(range(3),STRATEGIES);ax.set_yticks(range(len(methods)),methods)
    for i in range(len(methods)):
        for j in range(3):ax.text(j,i,str(int(values[i,j])) if values[i,j]<=128 else '>128',ha='center',va='center',color='white' if values[i,j]>100 else 'black')
    ax.set_title('B_SPT: registered grid, right censoring preserved');fig.tight_layout();save(fig,'B_SPT_budget_map')
    fields=['budget','key','controller','target','success_lower','success_upper','gain_lower','gain_upper','possible_catastrophe_rate','certain_catastrophe_rate','any_possible_catastrophe']
    csvtable('main_per_target_curves.csv',fields,[[q[k] for k in fields] for r in res for q in r['rows']]);csvtable('sample_efficiency.csv',['key','B_SPT','B_CT'],[[v['key'],v['B_SPT'],v['B_CT']] for v in eff['budgets']]);csvtable('target_B2pp.csv',['key','controller','necessary','B_2pp','B_2pp_certified'],[[v[k] for k in ['key','controller','necessary','B_2pp','B_2pp_certified']] for v in eff['target_B2pp']])
    for bed in ['main','stageB']:
        rows=[]
        for B in BUDGETS:
            r=read(HERE/'endpoints'/f'{bed}_B{B}.json')
            for q in r['inferences']:rows.append([B,q['key'],q['metric'],q['lower'],q['upper'],*q['ci95_simultaneous']])
        csvtable(bed+'_simultaneous_contrasts.csv',['B','key','metric','gain_lower','gain_upper','CI_lower','CI_upper'],rows)
    desc=[read(HERE/'endpoints'/f'stageB_B{B}.json') for B in BUDGETS];fig,axes=plt.subplots(1,3,figsize=(15,5))
    for st,strategy in enumerate(STRATEGIES):
        for m in ['E','E-safe','H','H-safe']:
            ss=[next(s for s in r['summaries'] if s['key']==strategy+'|'+m) for r in desc];axes[st].plot(x,[s['success_lower']*100 for s in ss],marker='o',label=m)
        axes[st].set_xticks(x,BUDGETS);axes[st].set_title(strategy+' | StageB P1 evaluation');axes[st].set(xlabel='B',ylabel='Success lower (%)');axes[st].grid(alpha=.2);axes[st].legend(fontsize=8)
    fig.suptitle('DEVELOPMENT descriptive only: StageB target TRAIN pool; source validation shares P1');fig.tight_layout();save(fig,'StageB_descriptive_curves')
if __name__=='__main__':main()
