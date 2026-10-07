"""Conservative standalone scientific SBGA figures; missing outcomes stay visible."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm

HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/name).read_text())
def save(fig,name):
    fig.tight_layout();fig.savefig(HERE/'figures'/name,dpi=150,bbox_inches='tight');plt.close(fig)

def main():
    (HERE/'figures').mkdir(exist_ok=True)
    p=read('protocol.json');mapping=read('outcome_basin_map.json');cs=mapping['coarse_cells'];axis=p['main_grid']['goal'];sids=p['primary_states']
    for outcome in ['success','deadlock','timeout']:
        fig,axes=plt.subplots(1,3,figsize=(12,4),sharex=True,sharey=True)
        for ax,sid in zip(axes,sids):
            vals=np.full((9,9),np.nan)
            for c in cs:
                if c['state_id']!=sid:continue
                ix=axis.index(c['phi'][0]);iy=axis.index(c['phi'][2]);vals[iy,ix]=c['counts'].get(outcome,0)/c['n']
                if c['unresolved_solver_failures']:ax.plot(c['phi'][0],c['phi'][2],'x',color='white',ms=5,mew=.8)
            im=ax.imshow(vals,origin='lower',extent=[-1.125,1.125,-1.125,1.125],vmin=0,vmax=1,cmap='viridis',aspect='equal')
            ax.set(title=sid,xlabel='goal gain',ylabel='relative gain')
        fig.suptitle(f'{outcome.upper()} map; white x = unresolved solver executions (color is lower outcome bound)')
        fig.subplots_adjust(top=.82,right=.86,wspace=.38)
        fig.colorbar(im,cax=fig.add_axes([.90,.18,.018,.57]),label=f'Observed {outcome} count / scheduled trials')
        fig.savefig(HERE/'figures'/f'Q_{outcome}.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    names=['SUCCESS','DEADLOCK','TIMEOUT','MIXED'];cmap=ListedColormap(['#269b70','#c4423d','#d9ae3a','#888888'])
    for ax,sid in zip(axes,sids):
        vals=np.full((9,9),3)
        for c in cs:
            if c['state_id']==sid:vals[axis.index(c['phi'][2]),axis.index(c['phi'][0])]=names.index(c['dominant_outcome'])
        ax.imshow(vals,origin='lower',extent=[-1.125,1.125,-1.125,1.125],cmap=cmap,vmin=-.5,vmax=3.5)
        ax.set(title=sid,xlabel='goal gain',ylabel='relative gain')
    fig.suptitle('Observed outcome dominance >= 75% of all scheduled trials; gray = mixed/unresolved')
    save(fig,'dominant_outcome.png')
    cc=read('connected_components.json')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,sid in zip(axes,sids):
        ax.scatter(*np.meshgrid(axis,axis),color='#dddddd',s=18)
        for i,comp in enumerate(cc['states'][sid]['components']):
            pts=np.array(comp);ax.scatter(pts[:,0],pts[:,2],s=100,label=f'component {i+1}')
        ax.set(title=sid,xlabel='goal gain',ylabel='relative gain',xlim=(-1.2,1.2),ylim=(-1.2,1.2))
        if cc['states'][sid]['components']:ax.legend(fontsize=7)
    fig.suptitle('Coarse success graph: one-sided 95% lower bound >= 0.75; topology provisional')
    save(fig,'success_components.png')
    interp=read('interpolation_test.json');fig,ax=plt.subplots(figsize=(6.5,4))
    if interp['cells']:
        rows=sorted(interp['cells'],key=lambda c:c['phi'][2]);x=[c['phi'][2] for c in rows]
        for event in ['success','deadlock','timeout']:ax.plot(x,[c['counts'].get(event,0)/c['n'] for c in rows],'o-',label=event)
        ax.legend();ax.set(xlabel='interpolated relative gain (fixed goal gain)',ylabel='Observed outcome fraction',ylim=(-.03,1.03))
    else:ax.text(.5,.5,'Not applicable: no two independently\nvalidated success components',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off()
    save(fig,'interpolation.png')
    minimum=read('minimum_success_correction.json');fig,ax=plt.subplots(figsize=(7,4))
    for i,s in enumerate(minimum['states']):
        c=s['minimum_observed_validated']
        if c:ax.bar(i,c['phi_norm'],color='#269b70');ax.text(i,c['phi_norm'],str(tuple(c['phi'])),fontsize=7,ha='center',va='bottom')
        else:ax.text(i,.05,'unresolved',ha='center')
    ax.set_xticks(range(3),sids);ax.set(ylabel='Smallest validated observed parameter norm',title='Finite sampled minimum; no continuous optimality claim')
    save(fig,'minimum_correction.png')
    temporal=read('temporal_basin_evolution.json');fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,s in zip(axes,temporal['states']):
        for c in s['cells']:
            if c['phi'][1]!=0:continue
            ax.scatter(c['phi'][0],c['phi'][2],c=[c['counts'].get('success',0)/c['n']],vmin=0,vmax=1,cmap='viridis',s=50)
        ax.set(title=f"same source: {s['offset_seconds']}s before D",xlabel='goal gain',ylabel='relative gain',xlim=(-.2,1.2),ylim=(-.7,1.2))
    fig.suptitle('Same-trajectory success slices; color = observed success / trials')
    save(fig,'temporal_success.png')

if __name__=='__main__':main()
