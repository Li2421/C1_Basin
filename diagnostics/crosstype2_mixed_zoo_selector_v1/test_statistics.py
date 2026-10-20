from core import *
from statistics_pipeline import *


def main():
    labels=np.zeros((2,2,17,2),np.int8);labels[:,:,0,0]=1;labels[0,:,12]=1;labels[1,:,13]=1;labels[:,:,16]=1
    data=dict(labels=labels,names=np.asarray(['a','b']),types=np.asarray(['A','B']),uid_class=np.broadcast_to(np.arange(17,dtype=np.int8)[None,None,:,None],labels.shape))
    for key in ('collision','timeout','unknown'):data[key]=np.zeros_like(labels)
    fixed={k:np.full((2,2),a,np.int32) for k,a in [('pipeline',0),('all_Sprog',12),('all_RHR',13),('source_fixed',10),('all_safety_baseline',10)]}
    cc,ss,w=point_indices(data['types'],2,False);_,_,contrasts,_=draw(data,fixed,cc,ss,w,[])
    assert np.array_equal(contrasts['pipeline__minus__best_fixed'],np.zeros((1,3)))
    cc=np.zeros((1,2),int);ss=np.broadcast_to(np.arange(2),(1,2,2));w=np.full((1,2),.5)
    _,_,contrast,_=draw(data,fixed,cc,ss,w,[]);assert np.array_equal(contrast['pipeline__minus__best_fixed'],np.full((1,3),-.5))
    labels[:,:,:16]=0;oracles=oracle_choices(labels[None]==1)
    assert all(not np.any(c==16) for c in oracles.values())
    labels[:,:,:16]=-1;data['unknown'][:,:,:16]=1;fixed['pipeline'][:]=12
    cc,ss,w=point_indices(data['types'],2,False);_,_,contrast,_=draw(data,fixed,cc,ss,w,[('pipeline','all_Sprog')])
    assert np.array_equal(contrast['pipeline__minus__all_Sprog'],np.zeros((1,3)))
    values=np.zeros((1,1,1,16,2));values[0,0,0,0,0]=1;values[0,0,0,1,1]=1
    choices=oracle_choices(values)['state_crossfit_oracle'];assert choices.tolist()==[[[[1,0]]]]
    write(HERE/'STATISTICS_TESTS.json',dict(utc=utc(),passed=True,nominal_excluded_from_oracles=True,opposite_seed_crossfit=True,best_fixed_reselected_per_draw=True,same_uid_unknowns_cancel=True,real_labels_used=False))
    print('Analytic statistics checks passed')


if __name__=='__main__':main()
