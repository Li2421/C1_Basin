"""Figures read sealed numerical evidence only, using the plots environment."""
from experiment import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    proof=read(HERE/'MIGRATION_NUMBERS.receipt.json')
    for path,expected in proof['files'].items():assert sha(path)==expected,path
    data=read(HERE/'MIGRATION_RESULTS.json')['results'];fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    names={'lit_narrow_passage':'Narrow passage','lit_crowded_random':'Crowded random (no walls)','lit_circle_five':'Five robots in circular enclosure','lit_doorway':'Doorway'}
    methods=['safety','fixed_eta10','fixed_eta13','human_progress'];labels=['Safety only','Fixed eta10','Fixed eta13','Progress switch']
    for ax,(scene,title) in zip(axes.ravel(),names.items()):
        rows=data[scene]['fixed_full']['statistics']
        for offset,layer,color in [(-.16,'central','#3974a8'),(.16,'distributed','#d77933')]:
            selected=[next(r for r in rows if r['type']=='type_equal' and r['layer']==layer and r['method']==m) for m in methods]
            y=np.asarray([r['success_rate'] for r in selected])*100;ci=np.asarray([r['ci95'] for r in selected]).T*100
            ax.errorbar(np.arange(4)+offset,y,yerr=np.maximum(np.stack([y-ci[0],ci[1]-y]),0),fmt='o',capsize=3,label=layer,color=color)
        ax.set_title(title);ax.set_xticks(range(4),labels,rotation=15);ax.set_ylim(-3,103);ax.set_ylabel('Success (%)');ax.grid(axis='y',alpha=.2)
    axes.ravel()[0].legend();folder=HERE/'figures';folder.mkdir(exist_ok=True)
    files=[folder/'fixed_central_vs_distributed.png',folder/'fixed_central_vs_distributed.pdf']
    for path in files:fig.savefig(path,dpi=180)
    plt.close(fig)
    seal('FIGURES.receipt.json',[*files,HERE/'plot_results.py'],source_numbers_receipt_sha256=sha(HERE/'MIGRATION_NUMBERS.receipt.json'),images_never_used_to_modify_simulation=True)

if __name__=='__main__':main()
