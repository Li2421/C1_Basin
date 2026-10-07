"""Publication figures from the audited snapshot; no interpolation or new labels."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
FIG=HERE/'figures'
COLORS=['#bdc3c9','#d46a19','#2469bb','#28886b']
MARKERS=['o','v','^','s']
NAMES=['Both fail B15','TT only (break)','Field only (rescue)','Both succeed B15']
SCENES=('toy_give_way','ring_exchange')
LABELS={'toy_give_way':'Toy GiveWay','ring_exchange':'Ring Exchange'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})


def read(p):return json.loads(Path(p).read_text())


def save(fig,name):
    for ext in ('png','pdf','svg'):
        fig.savefig(FIG/(name+'.'+ext),dpi=190,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    z=np.load(HERE/'paired_snapshot.npz');eta=z['eta'];o=z['primary']
    s=o[...,0].sum(-1);u=o[...,1].sum(-1)
    b=np.where(s>=15,1,np.where(s+u<15,0,-1));assert (b>=0).all()
    states=read(HERE/'inputs/primary_states.json')
    chosen=read(HERE/'figure_selection.json')['states']
    sizes=28+105*eta[:,2]/.75
    legend=[Line2D([],[],marker=m,linestyle='',markerfacecolor=c,markeredgecolor='white',
                   markersize=9,label=n) for m,c,n in zip(MARKERS,COLORS,NAMES)]

    def category(i):return b[0,i]+2*b[1,i]

    def axes(ax,small=False):
        ax.set(xlim=(.46,1.29),ylim=(-.54,.54),xticks=[.5,.75,1,1.25],yticks=[-.5,0,.5])
        ax.set_xlabel(r'$\eta_1$  goal',fontsize=8 if small else 10)
        ax.set_ylabel(r'$\eta_2$  flow-perpendicular',fontsize=8 if small else 10)
        ax.grid(alpha=.14);ax.set_axisbelow(True)

    def transition(ax,i,annotate=False):
        cat=category(i)
        for k in range(4):
            mask=cat==k
            ax.scatter(eta[mask,0],eta[mask,1],s=sizes[mask],marker=MARKERS[k],c=COLORS[k],
                       edgecolors='white',linewidths=.7,zorder=3)
        if annotate:
            for j,(x,y,_) in enumerate(eta):ax.annotate(str(j+1),(x,y),xytext=(5,3),
                                                      textcoords='offset points',fontsize=7,color='#303740')

    fig,axs=plt.subplots(2,3,figsize=(13,8.2))
    selections=[]
    for row,sc in enumerate(SCENES):
        i=next(j for j,st in enumerate(states) if st['uid']==chosen[sc]);selections.append(i)
        for ci in (0,1):
            ax=axs[row,ci];axes(ax)
            ax.scatter(eta[:,0],eta[:,1],s=sizes,c=np.where(b[ci,i], '#665298' if ci==0 else '#28886b',COLORS[0]),
                       edgecolors='white',linewidths=.7,zorder=3)
            for j,(x,y,_) in enumerate(eta):ax.annotate(str(j+1),(x,y),xytext=(5,3),
                                                      textcoords='offset points',fontsize=7,color='#303740')
            ax.set_title(f"{LABELS[sc]} · {'TT' if ci==0 else 'Field (FF)'}\n{b[ci,i].sum()}/16 robust samples",loc='left',fontsize=11)
        ax=axs[row,2];axes(ax);transition(ax,i,True)
        cat=category(i);r=int((cat==2).sum());br=int((cat==1).sum())
        ax.set_title(f'Paired membership changes\n{r} rescues · {br} breaks',loc='left',fontsize=11)
    fig.suptitle('Identical intervention samples reveal changes in robust success membership',fontsize=15,y=.985)
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.035),ncol=4,frameon=False)
    fig.text(.5,.005,'Fixed 2D projection; marker area encodes '+r'$\eta_3$ (relation). Labels are shared sample IDs. No interpolated boundaries.'+
             '\nStates chosen by a fixed outcome-blind hash rule; all 32 states appear in the atlas.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.88,bottom=.16,hspace=.47,wspace=.27)
    save(fig,'paired_basin_map')

    fig=plt.figure(figsize=(12,5.5))
    for panel,i in enumerate(selections):
        ax=fig.add_subplot(1,2,panel+1,projection='3d');cat=category(i)
        for k in range(4):
            mask=cat==k
            ax.scatter(*eta[mask].T,s=70,c=COLORS[k],marker=MARKERS[k],depthshade=False,
                       edgecolors='white',linewidths=.5)
        for j,e in enumerate(eta):ax.text(*e,str(j+1),fontsize=7)
        ax.set(xlim=(.5,1.25),ylim=(-.5,.5),zlim=(0,.75),xlabel=r'$\eta_1$ goal',
               ylabel=r'$\eta_2$ flow-perp.',zlabel=r'$\eta_3$ relation',title=LABELS[states[i]['scenario']])
        ax.view_init(elev=24,azim=-60)
    fig.suptitle('Paired membership in the full 3D intervention space',fontsize=14)
    fig.legend(handles=legend,loc='lower center',ncol=4,frameon=False)
    fig.subplots_adjust(bottom=.12,top=.86,wspace=.1)
    save(fig,'paired_basin_map_3d')

    order=sorted(range(len(states)),key=lambda i:(states[i]['scenario'],states[i]['uid']))
    fig,axs=plt.subplots(8,4,figsize=(13,21))
    plot_index=[]
    for panel,(ax,i) in enumerate(zip(axs.flat,order)):
        sc=states[i]['scenario'];short=('T' if sc=='toy_give_way' else 'R')+f'{sum(states[j]["scenario"]==sc for j in order[:panel+1]):02d}'
        axes(ax,True);transition(ax,i)
        cat=category(i);r=int((cat==2).sum());br=int((cat==1).sum())
        ax.set_title(f'{short}: TT {b[0,i].sum()} → Field {b[1,i].sum()}\nrescue {r}, break {br}',loc='left',fontsize=9)
        ax.tick_params(labelsize=8)
        plot_index.append(dict(plot_id=short,state_uid=states[i]['uid'],scenario=sc,
                               state_index=i,representative=states[i]['uid']==chosen[sc]))
    fig.suptitle('Every primary TEST state · identical 16-point Sobol bank',fontsize=15,y=.998)
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.002),ncol=4,frameon=False)
    fig.text(.5,.022,'Fixed '+r'$\eta_1$–$\eta_2$ projection; marker area encodes $\eta_3$. '
             'No state omitted. Point labels and exact coordinates are in cell_membership.csv.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.038,1,.98),h_pad=1.6,w_pad=1.2)
    save(fig,'all_states_atlas')
    with (HERE/'figure_state_index.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(plot_index[0]));w.writeheader();w.writerows(plot_index)

    fig,axs=plt.subplots(1,2,figsize=(11,8),sharex=True)
    for ax,sc in zip(axs,SCENES):
        ii=sorted([i for i,s in enumerate(states) if s['scenario']==sc],key=lambda i:states[i]['uid'])
        for y,i in enumerate(ii):
            aa,bb=b[:,i].mean(1);color='#28886b' if bb>aa else '#d46a19' if bb<aa else '#bdc3c9'
            ax.plot([aa,bb],[y,y],c=color,lw=2.4,zorder=1)
            ax.scatter(aa,y,c='#665298',s=45,marker='o',zorder=2)
            ax.scatter(bb,y,c='#28886b',s=45,marker='s',zorder=3)
            ax.text(max(aa,bb)+.026,y,f'{(bb-aa)*100:+.1f} pp',va='center',fontsize=8)
        prefix='T' if sc=='toy_give_way' else 'R'
        ax.set(yticks=range(16),yticklabels=[prefix+f'{k+1:02d}' for k in range(16)],
               xlim=(-.02,1.15),xticks=[0,.25,.5,.75,1],xlabel='Empirical robust basin fraction',title=LABELS[sc])
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Volume direction is evaluated per state',fontsize=15,y=.99)
    handles=[Line2D([],[],marker='o',color='#665298',linestyle='',label='TT'),
             Line2D([],[],marker='s',color='#28886b',linestyle='',label='Field (FF)')]
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.05,1,.95))
    save(fig,'per_state_volume')
    print(json.dumps({'figures':4,'formats':['png','pdf','svg'],'representative_states':chosen}))


if __name__=='__main__':main()
