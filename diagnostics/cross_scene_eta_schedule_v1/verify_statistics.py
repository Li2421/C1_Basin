from analyze import statistics
from core import *
Y=np.zeros((4,3,7,40,4));U=np.zeros_like(Y);Y[0,:,0]=1;Y[1:,:,2]=1;Y[:,:,4]=1
r,*_=statistics(Y,U);assert r['X1']['status']=='SUPPORTED';assert r['X2']['status']=='REJECTED';assert r['X2']['methods']['expanded']['choices'][0]=='F15','Frozen earliest-argmax tie rule'
Y[1:,:,2,:10]=0;r,*_=statistics(Y,U);assert r['X1']['status']=='SUPPORTED' and r['X2']['status']=='SUPPORTED' and r['X2']['methods']['expanded']['max_regret']==0
Y[:,:,4,:10]=0;U[:,:,4,:10]=1;r,*_=statistics(Y,U);assert r['X1']['status']=='INCONCLUSIVE','Unknown is not observed failure or imputed success'
write(HERE/'STATISTICS_VERIFICATION.json',dict(utc=utc(),passed=True,synthetic_only=True,final_outcomes_read=False,checks=['four-scene averaging','X1 best-fixed noninferiority','source-only expanded LOSO','earliest menu tie','unknown bounds','E-L=N'],analysis_sha256=sha(HERE/'analyze.py')))
print('Synthetic statistics PASS')
