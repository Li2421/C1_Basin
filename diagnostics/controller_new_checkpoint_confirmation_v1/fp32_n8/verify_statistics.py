from analyze import statistics
from core import *
names=['synthetic'+str(i) for i in range(8)];arms=['F10','F13','S300','S500','F2','F4'];Y=np.ones((8,6,64,4));U=np.zeros_like(Y)
# Six S500-selected controllers gain 12.5pp over F10; two F10 selections
# save 12.5pp vs S500, giving exactly two true low-E controllers.
Y[:6,0,:8]=0;Y[6:,3,:8]=0
selectors={n:dict(primary_action='S500' if i<6 else 'F10',secondary_action='S500' if i<6 else 'F10',primary_Ehat=.125 if i<6 else -.125,secondary_Ehat=.125 if i<6 else -.125,primary_source_mean_E=.3,secondary_source_mean_E=.3) for i,n in enumerate(names)}
r,*_=statistics(Y,U,names,arms,selectors);m=r['methods']['primary'];assert m['C1']=='SUPPORTED' and m['C1b']=='SUPPORTED' and m['C2']=='SUPPORTED' and m['headroom_count']==2 and m['mean_regret']==0
# All S500: oracle and shared selected-arm comparisons must cancel exactly.
for x in selectors.values():x['primary_action']='S500';x['secondary_action']='S500'
r,*_=statistics(Y,U,names,arms,selectors);m=r['methods']['primary'];assert m['gain_vs_source_average_CI95'].tolist()==[0,0] and m['C2']=='REJECTED'
# Remove low-E headroom: C2 must become INCONCLUSIVE, even if mean gain exists.
Y=np.ones_like(Y);Y[:,0,:8]=0;r,*_=statistics(Y,U,names,arms,selectors);assert r['methods']['primary']['C2']=='INCONCLUSIVE'
# Unknown-only distinction cannot be called confirmed benefit.
Y=np.ones_like(Y);Y[:,0,:8]=0;U[:,0,:8]=1;r,*_=statistics(Y,U,names,arms,selectors);assert r['methods']['primary']['C1']=='INCONCLUSIVE'
write(HERE/'STATISTICS_VERIFICATION.json',dict(utc=utc(),passed=True,synthetic_only=True,no_final_labels_read=True,checks=['n8 unchanged gates','paired S500 equality cancellation','trueE headroom2','noheadroom INCONCLUSIVE','unknown envelope blocks false confirmation'],analysis_sha256=sha(HERE/'analyze.py')))
print('Synthetic registered-gate checks PASS')
