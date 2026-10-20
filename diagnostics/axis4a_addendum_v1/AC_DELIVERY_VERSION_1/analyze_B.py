from common import *
from statistics import summarize,ci
from pipeline import lookup
OLD=D/'axis4a_geometry_horizon_v1'
def main():
 assert (HERE/'B_NUMERICAL_COMPLETE.receipt.json').exists();names=read(HERE/'HISTORICAL_INVENTORY.json')['controller_names'];choices0=read(HERE/'FROZEN_REFERENCE_CHOICES.json')['controllers'];parts={};data={};differences=[]
 for H in [500,700,1000]:
  with np.load(OLD/f'H{H}_LABELS.npz') as z:assert z['controller_names'].tolist()==names;oldlo=z['lower'];oldhi=z['upper'];arms=z['arms'].tolist();events={k:np.concatenate([z[k],np.zeros((14,32,1,2))],axis=2) for k in ['collision','timeout','unknown']}
  arms.append('Sprog');lo=np.concatenate([oldlo,np.zeros((14,32,1,2))],2);hi=np.concatenate([oldhi,np.zeros((14,32,1,2))],2);filled=np.zeros((14,32,2),bool)
  for c in read(HERE/'B_CASES.json'):
   if c['H']!=H:continue
   q=read(lookup(c,'B'));idx=(names.index(c['base_controller_name']),c['state_index'],c['seed']);assert not filled[idx];filled[idx]=True;full=(idx[0],idx[1],len(arms)-1,idx[2]);lo[full]=q['success'];hi[full]=q['success']+q['numerical_failure'];events['collision'][full]=q['collision'];events['timeout'][full]=q['timeout'];events['unknown'][full]=q['numerical_failure']
  assert filled.all();relative='S500' if H==700 else 'Srel';absolute='S500' if H==700 else 'F13' if H==500 else 'Sabs';choice={'nominal only':0,'safety baseline':1,'F10':arms.index('F10'),'F13':arms.index('F13'),'Srel':arms.index(relative),'Sabs':arms.index(absolute),'Sprog':arms.index('Sprog')}
  if H==700:choice['S500']=arms.index('S500')
  for m,old in choices0.items():choice[m]=[arms.index(relative if old[n]=='S500' else old[n]) for n in names]
  res=summarize(lo,hi,events,arms,choice,202610163900+H);gain=res['paired']['A3 minus safety baseline']['CI_pp'];reg=res['methods']['A3']['mean_regret_pp'];res.update(H=H,r=.9,arms=arms,grade='CONFIRMATORY（定义修订）; historical inputs keep originalsourcegrades',HZ4=dict(status='SUPPORTED' if gain[0]>0 and reg[1]<=3 else 'REJECTED' if gain[1]<=0 or reg[0]>3 else 'INCONCLUSIVE',A3_minus_safety_CI_pp=gain,regret_pp=reg,complete_correction_menu=True));parts[str(H)]=res;data[H]=(lo,hi);pr=arms.index('Sprog');rel=arms.index(relative);differences.append((lo[:,:,pr]-hi[:,:,rel],hi[:,:,pr]-lo[:,:,rel]));np.savez_compressed(HERE/f'B_H{H}_LABELS.npz',lower=lo,upper=hi,arms=arms,controller_names=names,**events)
 rng=np.random.default_rng(202610163999);pooled=[];individual={H:[] for H in [500,700,1000]}
 for _ in range(5000):
  cc=rng.integers(14,size=14);ss=rng.integers(32,size=(14,32));values=[]
  for H,(a,b) in zip([500,700,1000],differences):
   v=[a[cc[:,None],ss].mean(),b[cc[:,None],ss].mean()];individual[H].append(v);values.append(v)
  pooled.append(np.mean(values,axis=0))
 interval=ci(pooled);hz3=dict(status='BETTER' if interval[0]>0 else 'WORSE' if interval[1]<0 else 'UNDETECTED',estimate_pp=(np.mean([[a.mean(),b.mean()] for a,b in differences],axis=0)*100).tolist(),CI_pp=interval,perH={str(H):dict(CI_pp=ci(individual[H]),estimate_pp=[a.mean()*100,b.mean()*100]) for H,(a,b) in zip([500,700,1000],differences)},scope='Same controller/state bootstrap draws across all3H; bothfutures clustered. Undetected doesnotproveequivalence.');write(HERE/'B_RESULTS.json',dict(utc=utc(),decision=read(HERE/'PSTAR_DECISION.receipt.json'),HZ3=hz3,parts=parts));print('B analyzed',hz3,flush=True)
if __name__=='__main__':main()
