from statistics import *
lo=np.zeros((2,4,4,2));lo[:,:,2]=1;hi=lo.copy();m,s=evaluate(lo,hi,{'F10':2,'F13':3},[2,3],[2,3]);assert np.all(m['full menu oracle'][0]==1);assert np.all(s['best fixed oracle']==2)
# Unknown-only competing arm cannot be treated as a definite losing arm.
hi[:,:,3]=1;m,s=evaluate(lo,hi,{'F10':2,'F13':3},[2,3],[2,3]);assert np.all(m['full menu oracle'][0]==0) and np.all(m['full menu oracle'][1]==1)
# Cross-fitting must evaluate on the opposite states, not select each test outcome.
lo=np.zeros((1,4,2,1));lo[0,::2,0]=1;lo[0,1::2,1]=1;m,_=oracle(lo,lo,[0,1]);assert m[0].mean()==0
print('Oracle selection/unknown bounds/state crossfit tests PASS')
