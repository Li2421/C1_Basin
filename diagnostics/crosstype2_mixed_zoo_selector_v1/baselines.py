from core import *
from statistics_pipeline import best

LEARNED={'Flow','gaussian','deterministic','mixture','diffusion','chunking','recurrent','cvae','ring_ippo','ring_ippo_conservative'}
NONLEARNED={'distributed_shooting','centralized_sampling','cross_entropy','prioritized_astar'}
CLASSES=('no_correction','parameter_correction','continuing_right_hand')


def repair_class(index):
    return np.where(np.asarray(index)==10,0,np.where(np.asarray(index)>=13,2,1))


def source_fixed(data,source):
    # Every candidate has seed0; extra historical anchor seed1 cannot bias menus.
    lower=(data['labels'][...,0]==1).astype(float)
    type_scores=np.asarray([lower[data['types']==t].mean((0,1)) for t in sorted(source)])
    winners=best(type_scores);votes=np.bincount(winners,minlength=16);average=type_scores.mean(0)
    finalists=np.flatnonzero(votes==votes.max());top=finalists[np.isclose(average[finalists],average[finalists].max(),rtol=0,atol=1e-12)]
    chosen=12 if 12 in top else int(top.min())
    return chosen,int(best(average)),dict(source_types=sorted(source),type_winners=winners.tolist(),votes=votes.tolist(),type_equal_success_lower=average.tolist(),primary_definition='majority of source-type best candidates; tie source type-equal mean then progress anchor/menu order',common_seed=0)


def few_label(data,k):
    return best((data['labels'][:,:k,:,0]==1).mean(1))
