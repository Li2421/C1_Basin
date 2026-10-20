"""Static scientific figures; use only stored simulation/choice evidence."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
def save(fig,name):
    folder=HERE/'figures';folder.mkdir(exist_ok=True);fig.tight_layout();fig.savefig(folder/(name+'.svg'));fig.savefig(folder/(name+'.png'),dpi=160);plt.close(fig)
def main():
    descriptions=read(HERE/'SELECTED_ETA_DESCRIPTIONS.json')['rows'];base=read(HERE/'BASE_PROTOCOL.json');eta13=np.asarray(base['eta'][13]);eta10=np.asarray(base['eta'][10]);kinds=('transformer_bc','gradient_mpc','conflict_based_search')
    chosen=read(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json')['states'];maps=read(HERE/'NEW_EVAL_METHOD_UIDS.json')
    for kind in kinds:
        rows=[r for r in descriptions if r['type']==kind]
        if not rows:continue
        early=np.asarray([r['eta_early'] for r in rows]);late=np.asarray([r['eta_late'] for r in rows]);fig,axes=plt.subplots(1,3,figsize=(11,3.4))
        for j,ax in enumerate(axes):
            ax.scatter(early[:,j],late[:,j],s=7,alpha=.25);ax.scatter([eta13[j]],[eta13[j]],c='red',marker='x',s=60,label='eta13');ax.scatter([eta10[j]],[eta10[j]],c='black',marker='+',s=60,label='eta10');ax.set(xlabel=f'Early eta[{j+1}]',ylabel=f'Late eta[{j+1}]');ax.legend(fontsize=7)
        save(fig,kind+'_eta_scatter');fig,axes=plt.subplots(1,3,figsize=(11,3.4))
        for j,ax in enumerate(axes):
            ax.hist(early[:,j],bins=20,alpha=.5,label='Early');ax.hist(late[:,j],bins=20,alpha=.5,label='Late');ax.axvline(eta13[j],c='red',ls='--',label='eta13');ax.axvline(eta10[j],c='black',ls=':',label='eta10');ax.set(xlabel=f'eta[{j+1}]',ylabel='Controller-state count');ax.legend(fontsize=7)
        save(fig,kind+'_eta_histograms');fig,ax=plt.subplots(figsize=(7,3.5))
        for phase in ('early','late'):
            for anchor in ('eta13','eta10'):ax.hist([r[f'distance_{phase}_{anchor}'] for r in rows],bins=20,alpha=.35,label=phase+' to '+anchor)
        ax.set(xlabel='Euclidean distance',ylabel='Count');ax.legend();save(fig,kind+'_eta_distances')
        fig,ax=plt.subplots(figsize=(6,3.4));cos=[r['direction_cosine'] for r in rows if r['direction_cosine'] is not None];ax.hist(cos,bins=np.linspace(-1,1,21));ax.set(xlabel='Cosine of late-minus-early with eta10-minus-eta13',ylabel='Count',title=f'{kind}; diagonal undefined: {len(rows)-len(cos)}');save(fig,kind+'_direction_cosine')
        fig,ax=plt.subplots(figsize=(6,3.4));ticks=['diagonal','200','300','400','500','600'];counts=[sum(r['diagonal'] for r in rows)]+[sum(r['switch_step']==t for r in rows) for t in (200,300,400,500,600)];ax.bar(ticks,counts);ax.set(xlabel='Effective switch step',ylabel='Controller-state count');save(fig,kind+'_switch_times')
        # Predetermined representative: first eligible instance, state0, seed0.
        record=next(r for r in maps if r['type']==kind and r['state_index']==0);colors=plt.cm.tab10(np.arange(4));fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax,method in zip(axes,('all_safety_baseline','all_S500','pipeline')):
            rid=record['method_uids'][method][0];raw=read(HERE/'new_evaluation/raw'/f'{rid}.json');assert sha(raw['trace_path'])==raw['trace_sha256']
            with np.load(raw['trace_path'],allow_pickle=False) as z:p=z['positions'];goals=z['goals']
            ax.add_patch(Circle((0,0),.9,fc='.8'));ax.add_patch(Circle((0,0),3.25,fill=False,color='.5'))
            for robot in range(4):ax.plot(p[:,robot,0],p[:,robot,1],c=colors[robot],lw=1.3);ax.scatter(*p[0,robot],c=[colors[robot]],s=16);ax.scatter(*goals[robot],c=[colors[robot]],marker='*',s=50)
            ax.set(xlim=(-3.4,3.4),ylim=(-3.4,3.4),title=method);ax.set_aspect('equal')
        save(fig,kind+'_representative_trajectories')
    fig,ax=plt.subplots(figsize=(8,4))
    for cfg in read(HERE/'TRANSFORMER_TRAINING_CONFIGS.json'):
        history=read(HERE/'policies'/cfg['name']/'TRAINING_HISTORY.json');ax.plot([r['step'] for r in history],[r['dev_loss'] for r in history],label=cfg['name'])
    ax.set(xlabel='Offline optimization step',ylabel='Fixed dev joint Gaussian NLL');ax.legend(fontsize=8);save(fig,'transformer_training_curves')
    features=read(HERE/'FEATURE_OUT_OF_RANGE.json')['rows'];fig,ax=plt.subplots(figsize=(9,4));valid=[r for r in features if r['out_of_range_fraction'] is not None];ax.bar([r['controller'] for r in valid],[r['out_of_range_fraction'] for r in valid]);ax.tick_params(axis='x',rotation=80);ax.set(ylabel='Fraction observed dimensions outside training range');save(fig,'unseen_feature_ranges')
    selected=read(HERE/'SELECTED_MODELS.json');folds=read(HERE/'DEVELOPMENT_OUTER_FOLDS.json');groups=[(r['held_type'],r['source_only_selection']) for r in folds]+[('all16',selected['all_types']['source_only_selection'])];fig,axes=plt.subplots(4,5,figsize=(16,11))
    for ax,(name,record) in zip(axes.flat,groups):
        for mode,branch in record['branches'].items():ax.plot([r['step'] for r in branch['history']],[r['mean_nll'] for r in branch['history']],label=mode);ax.axvline(branch['selected']['step'],ls=':',alpha=.3)
        ax.set_title(name,fontsize=8);ax.legend(fontsize=6)
    for ax in list(axes.flat)[len(groups):]:ax.axis('off')
    save(fig,'nested_source_checkpoint_curves')
    files=list((HERE/'figures').glob('*'));seal('FIGURES.receipt.json',files+[HERE/'plots.py',HERE/'CONFIRMATORY_RESULTS.receipt.json'],static_saved_evidence_only=True,representatives='First eligible instance, state0,seed0; Safety/S500/pipeline')
if __name__=='__main__':main()
