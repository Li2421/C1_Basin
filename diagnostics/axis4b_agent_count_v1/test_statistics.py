from statistics import *
lo=np.zeros((1,32,9,2));lo[0,::2,2]=1;lo[0,1::2,3]=1;source=dict(classes=['P2'],lower=lo,upper=lo.copy());choice=source_choices(source);assert choice['per_controller']['P2']['0']['point']==3 and choice['per_controller']['P2']['1']['point']==2
m,_=transfer(lo,lo,['P2'],source);assert m[0].mean()==0
# Source selection cannot access a target-optimal arm.
target=np.zeros_like(lo);target[:,:,8]=1;m,_=transfer(target,target,['P2'],source);assert m[0].mean()==0
# Complete-menu oracle includes the safety baseline, unlike source correction menu.
a=np.zeros_like(lo);a[:,:,1]=1;m,_=oracle(a,a,FULL);assert m[0].mean()==1;assert all(v['point']>=2 for v in source_choices(dict(classes=['P2'],lower=a,upper=a))['per_controller']['P2'].values())
# Bootstrap must retain original physical fold identities under reordering.
indices=np.arange(32)[::-1][None];m,_=oracle(lo[:,::-1],lo[:,::-1],CORRECTION,indices);assert m[0].mean()==0
# Unobserved source outcomes remain bounded; no optimistic deterministic choice.
a=np.zeros_like(lo);b=a.copy();b[:,:,2]=1;m,_=transfer(np.zeros_like(lo),np.ones_like(lo),['P2'],dict(classes=['P2'],lower=a,upper=b));assert np.all(m[0]==0) and np.all(m[1]==1)
print('Source/target isolation, all9 oracle, state-fold and unknown tests PASS')
