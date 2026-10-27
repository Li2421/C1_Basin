"""Main matched-evaluation figures and explicitly separated fixed references."""
from experiment import *
from registry import verify_receipt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    verify_receipt('MIGRATION_NUMBERS.receipt.json')
    data=read(HERE/'MIGRATION_RESULTS.json')['results'];files=[];folder=HERE/'figures';folder.mkdir(exist_ok=True)
    names={'lit_narrow_passage':'Narrow passage','lit_crowded_random':'Crowded random (no walls)','lit_circle_five':'Five robots in circular enclosure','lit_doorway':'Doorway'}
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,(scene,title) in zip(axes.ravel(),names.items()):
        summary=data[scene]['revised']['primary_evaluation'];selection=read(HERE/f'TRAIN_SELECTED_FIXED_20261022_{scene}.json');rows=summary['statistics']
        chosen=[next(r for r in rows if (r['type'],r['layer'],r['method'])==('type_equal','distributed',m)) for m in ('safety','train_selected_fixed','learned_scene')]
        y=np.asarray([r['success_rate'] for r in chosen])*100;ci=np.asarray([r['ci95'] for r in chosen]).T*100;upper=np.asarray([r['success_upper'] for r in chosen])*100
        ax.errorbar(np.arange(3),y,yerr=np.maximum(np.stack([y-ci[0],ci[1]-y]),0),fmt='o',capsize=3,color='#d77933',label='Success lower bound and95% CI')
        for index,(low,high) in enumerate(zip(y,upper)):
            if high>low:ax.plot([index,index],[low,high],color='#777777',lw=3);ax.plot(index,high,'o',mfc='none',mec='#777777')
        ax.set_title(title+f" (fixed menu {selection['selected_menu_index']},0-based)");ax.set_xticks(range(3),['Safety only','Train-selected fixed','Scene learned'],rotation=12);ax.set_ylim(-3,103);ax.set_ylabel('Success (%)');ax.grid(axis='y',alpha=.2)
    axes.ravel()[0].legend(fontsize=8);fig.suptitle('Primary: distributed revised semantics, matched held-out states; fixed action selected on training labels')
    for ext in ('png','pdf'):
        path=folder/f'primary_train_selected_fixed_vs_learned_20261022.{ext}';assert not path.exists();fig.savefig(path,dpi=180);files.append(path)
    plt.close(fig)
    for sem in ('original','revised'):
        fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
        for ax,(scene,title) in zip(axes.ravel(),names.items()):
            entry=data[scene] if sem=='original' else data[scene]['revised'];rows=entry['fixed_full']['statistics']
            for offset,layer,color in [(-.16,'central','#3974a8'),(.16,'distributed','#d77933')]:
                chosen=[next(r for r in rows if (r['type'],r['layer'],r['method'])==('type_equal',layer,m)) for m in ('safety','fixed_eta10','fixed_eta13','human_progress')]
                y=np.asarray([r['success_rate'] for r in chosen])*100;ci=np.asarray([r['ci95'] for r in chosen]).T*100
                ax.errorbar(np.arange(4)+offset,y,yerr=np.maximum(np.stack([y-ci[0],ci[1]-y]),0),fmt='o',capsize=3,label=layer,color=color)
            ax.set_title(title);ax.set_xticks(range(4),['Safety only','Fixed eta10','Fixed eta13','Progress switch'],rotation=15);ax.set_ylim(-3,103);ax.set_ylabel('Success (%)');ax.grid(axis='y',alpha=.2)
        axes.ravel()[0].legend();fig.suptitle('Appendix references, full main states: '+('original preregistered semantics' if sem=='original' else 'revised posthoc semantics'))
        for ext in ('png','pdf'):
            path=folder/f'appendix_fixed_central_vs_distributed_{sem}_20261022.{ext}';assert not path.exists();fig.savefig(path,dpi=180);files.append(path)
        plt.close(fig)
    seal('FIGURES.receipt.json',[*files,Path(__file__)],source_numbers_receipt_sha256=sha(HERE/'MIGRATION_NUMBERS.receipt.json'),images_never_used_to_modify_simulation=True,primary_training_selected_fixed=True,reference_figures_explicitly_marked_appendix=True)


if __name__=='__main__':main()
