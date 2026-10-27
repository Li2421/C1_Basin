"""Final static figures read sealed numerical evidence, never simulation inputs."""
from experiment import *
from registry import verify_receipt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
 verify_receipt('MIGRATION_NUMBERS.receipt.json');data=read(HERE/'MIGRATION_RESULTS.json')['results'];files=[];folder=HERE/'figures';folder.mkdir(exist_ok=True)
 names={'lit_narrow_passage':'Narrow passage','lit_crowded_random':'Crowded random (no walls)','lit_circle_five':'Five robots in circular enclosure','lit_doorway':'Doorway'}
 for sem in ('original','revised'):
  fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
  for ax,(scene,title) in zip(axes.ravel(),names.items()):
   entry=data[scene] if sem=='original' else data[scene]['revised'];rows=entry['fixed_full']['statistics']
   for offset,layer,color in [(-.16,'central','#3974a8'),(.16,'distributed','#d77933')]:
    chosen=[next(r for r in rows if (r['type'],r['layer'],r['method'])==('type_equal',layer,m)) for m in ('safety','fixed_eta10','fixed_eta13','human_progress')]
    y=np.asarray([r['success_rate'] for r in chosen])*100;ci=np.asarray([r['ci95'] for r in chosen]).T*100
    ax.errorbar(np.arange(4)+offset,y,yerr=np.maximum(np.stack([y-ci[0],ci[1]-y]),0),fmt='o',capsize=3,label=layer,color=color)
   ax.set_title(title);ax.set_xticks(range(4),['Safety only','Fixed eta10','Fixed eta13','Progress switch'],rotation=15);ax.set_ylim(-3,103);ax.set_ylabel('Success (%)');ax.grid(axis='y',alpha=.2)
  axes.ravel()[0].legend();fig.suptitle('Original preregistered semantics' if sem=='original' else 'Revised posthoc development evidence')
  for ext in ('png','pdf'):
   path=folder/f'fixed_central_vs_distributed_{sem}.{ext}';fig.savefig(path,dpi=180);files.append(path)
  plt.close(fig)
 seal('FIGURES.receipt.json',[*files,Path(__file__)],source_numbers_receipt_sha256=sha(HERE/'MIGRATION_NUMBERS.receipt.json'),images_never_used_to_modify_simulation=True)
if __name__=='__main__':main()
