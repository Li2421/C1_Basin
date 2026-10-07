"""Lightweight deterministic SBMA figures from sealed tables and traces."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from diagnostics.success_basin_multimodality.setup import HERE

def read(name):return json.loads((HERE/name).read_text())
def save(fig,name):
    fig.savefig(HERE/'figures'/name,dpi=150,bbox_inches='tight');plt.close(fig)

def main():
    (HERE/'figures').mkdir(exist_ok=True);p=read('protocol.json');m=read('success_map.json');cells=m['phase_a_cells'];states=[x['state_id'] for x in p['primary_states']]
    axes=p['phase_a_design']['axes'];goals=axes['goal'];safes=axes['safe'];rels=axes['relative']
    for outcome in ['success','deadlock','timeout']:
        fig,axs=plt.subplots(3,5,figsize=(14,8),sharex=True,sharey=True)
        for i,sid in enumerate(states):
            for j,safe in enumerate(safes):
                a=axs[i,j];v=np.zeros((len(rels),len(goals)))
                for c in cells:
                    if c['state_id']==sid and c['eta'][1]==safe:v[rels.index(c['eta'][2]),goals.index(c['eta'][0])]=c['probabilities'][outcome]
                im=a.imshow(v,origin='lower',vmin=0,vmax=1,cmap='viridis',extent=[.375,1.375,-.125,.875],aspect='auto')
                if i==0:a.set_title(f'safe={safe:g}')
                if j==0:a.set_ylabel(f'{sid}\nrelative')
                if i==2:a.set_xlabel('goal')
        fig.suptitle(f'Phase-A Q_{outcome[0].upper()} empirical slices; 16 fresh seeds/cell')
        fig.subplots_adjust(right=.90,top=.91,wspace=.25,hspace=.25);fig.colorbar(im,cax=fig.add_axes([.92,.16,.015,.68]))
        save(fig,f'Q_{outcome}_slices.png')
    cc=read('connected_components.json');fig=plt.figure(figsize=(13,4))
    colors={'SUCCESS_CELL':'#1b9e77','FAILURE_CELL':'#d95f02','UNKNOWN_CELL':'#777777'}
    for i,sid in enumerate(states):
        ax=fig.add_subplot(1,3,i+1,projection='3d')
        for label,color in colors.items():
            q=np.asarray([c['eta'] for c in cells if c['state_id']==sid and c['classification']==label])
            if len(q):ax.scatter(q[:,0],q[:,1],q[:,2],c=color,label=label.replace('_CELL',''),s=24)
        ax.set(xlabel='goal',ylabel='safe',zlabel='relative',title=f"{sid}: {cc['states'][sid]['component_count']} success component")
        if i==0:ax.legend(fontsize=7)
    fig.suptitle('3-D empirical success graph (axis-neighbor adjacency)');save(fig,'success_connected_components.png')
    path=read('path_connectivity.json');fig=plt.figure(figsize=(6,5));ax=fig.add_subplot(111,projection='3d')
    common=np.asarray(cc['common_success_cells_all_three_states']);ax.scatter(common[:,0],common[:,1],common[:,2],c='#cccccc',s=25,label='common Phase-A success')
    line=np.asarray([(1-l)*np.asarray(path['eta_A'])+l*np.asarray(path['eta_B']) for l in path['lambda']]);ax.plot(line[:,0],line[:,1],line[:,2],'o-',c='#2b6cb0',lw=2,label='fresh-validated straight path')
    ax.set(xlabel='goal',ylabel='safe',zlabel='relative',title='Candidate success path: 1056/1056 success');ax.legend();save(fig,'candidate_success_path.png')
    inter=read('interpolation_results.json');fig,axs=plt.subplots(1,3,figsize=(12,3.5),sharey=True)
    for ax,sid in zip(axs,states):
        rows=sorted([c for c in inter['cells'] if c['state_id']==sid],key=lambda c:c['lambda'])
        for e,color in [('success','#1b9e77'),('deadlock','#d95f02'),('timeout','#e6ab02')]:ax.plot([c['lambda'] for c in rows],[c['probabilities'][e] for c in rows],'o-',label=e,color=color)
        ax.set(title=sid,xlabel='lambda',ylim=(-.03,1.03));
    axs[0].set_ylabel('outcome probability');axs[-1].legend(fontsize=8);fig.suptitle('Straight interpolation outcomes: 32 fresh seeds at every point');save(fig,'interpolation_outcomes.png')
    margin=read('basin_margin.json');names=['+goal','+safe','+relative','-goal','-safe','-relative'];fig,axs=plt.subplots(2,3,figsize=(11,6),sharex=True,sharey=True)
    for ax,name in zip(axs.ravel(),names):
        for sid in states:
            row=next(x for x in margin['directions'] if x['state_id']==sid and x['direction']==name)
            ax.plot([c['delta'] for c in row['tested_cells']],[c['probabilities']['success'] for c in row['tested_cells']],'o-',label=sid)
        ax.axhline(.8,color='k',ls='--',lw=.8);ax.set(title=name,xlabel='delta',ylabel='Q_S',ylim=(-.03,1.03))
    axs[0,0].legend(fontsize=7);fig.suptitle('Local empirical success margin around eta=(1,0,.25)');fig.tight_layout();save(fig,'local_success_margin.png')
    jobs=read('interpolation_jobs.json');records=json.loads((HERE/'raw/straight_interpolation/manifest.json').read_text())['records'];fig,axs=plt.subplots(1,3,figsize=(12,3.7))
    for ax,sid in zip(axs,states):
        for lam,color in [(0.,'#2b6cb0'),(1.,'#c53030')]:
            r=next(x for x in records if x['state_id']==sid and x['seed']==95103001 and x['lambda']==lam)
            with np.load(HERE/r['file']) as d:
                for agent,style in [(0,'-'),(1,'--')]:ax.plot(d['positions_after'][:,agent,0],d['positions_after'][:,agent,1],style,color=color,label=f'lambda={lam:g}, agent{agent}')
        ax.set(title=sid,xlabel='x',ylabel='y',aspect='equal');
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,fontsize=7);fig.suptitle('Representative successful endpoint trajectories',y=1.05);save(fig,'representative_trajectories.png')

if __name__=='__main__':main()
