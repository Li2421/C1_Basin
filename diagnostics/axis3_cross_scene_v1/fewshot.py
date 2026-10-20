from common import *
from functools import lru_cache
from scipy.special import betaln,logsumexp
@lru_cache(None)
def beta_gt(s,n,t,m):
 a=s+1;b=n-s+1;c=t+1;d=m-t+1;i=np.arange(a);return float(np.exp(logsumexp(betaln(c+i,b+d)-np.log(b+i)-betaln(1+i,b)-betaln(c,d))))

def main():
 cfg=read(HERE/'DESIGN_CONFIG.json');results={};table=[]
 for j,sc in enumerate(SCENES):
  path=HERE/f'{sc}_LABELS.npz'
  if not path.exists():results[sc]=dict(status='GAP',reason='No prospective target labels');continue
  z=np.load(path);names=z['controller_names'].tolist();ids=z['state_uids'].tolist();states=read(HERE/'STATES.json')[sc]['eval'];original={s['state_uid']:s['original_state_uid'] for s in states};order=np.argsort([original[s] for s in ids]);cal=order[:16];ev=order[16:];Y=z['lower'][:,:,2:];U=z['unknown'][:,:,2:];unit=[];rng=np.random.default_rng(202610153401+j);weights={};evalrate=Y[:,ev].mean((1,3));evalupper=(Y[:,ev]+U[:,ev]).mean((1,3));baseline=MENU.index('S13_t2')
  for c,name in enumerate(names):
   pools=[Y[c,cal,a][U[c,cal,a]==0].flatten() for a in range(7)];rates=evalrate[c];upper=evalupper[c];oracle=rates.max();oraclehi=upper.max();stats={}
   for k in cfg['ks']:
    if min(map(len,pools))<k:stats[str(k)]=dict(status='GAP',reason='Insufficient known calibration episodes; unknown not rerun');continue
    wins=np.asarray([[int(pool[rng.permutation(len(pool))[:k]].sum()) for pool in pools] for _ in range(1000)]);emp=wins.argmax(1);emp[(wins==wins.max(1)[:,None]).sum(1)>1]=baseline;safe=np.full(1000,baseline,int)
    for b in range(1000):
     eligible=[a for a in range(7) if a!=baseline and beta_gt(int(wins[b,a]),k,int(wins[b,baseline]),k)>=.9]
     if eligible:safe[b]=max(eligible,key=lambda a:(wins[b,a],-a))
    stats[str(k)]={}
    for method,choice in [('D0',np.full(1000,baseline,int)),('E',emp),('S',safe)]:
     low=oracle-upper[choice];high=oraclehi-rates[choice];regret=oracle-rates[choice];stats[str(k)][method]=dict(mean_regret_pp=float(regret.mean()*100),identified_regret_pp=[float(low.mean()*100),float(high.mean()*100)],p90_regret_pp=float(np.quantile(regret,.9)*100),prob_regret_le3=float((regret<=.03+1e-12).mean()),success_pct=float(rates[choice].mean()*100),known_labels_per_arm=k,total_known_labels=7*k,choice_probabilities=np.bincount(choice,minlength=7)/1000);weights.setdefault((k,method),[]).append(np.bincount(choice,minlength=7)/1000);table.append(dict(scene=sc,controller=name,k=k,method=method,**{x:stats[str(k)][method][x] for x in ['mean_regret_pp','p90_regret_pp','prob_regret_le3','success_pct','total_known_labels']}))
   unit.append(dict(controller=name,stats=stats,calibration_known_counts=[len(x) for x in pools]))
  curves={};meanstar={};reliablestar={}
  for method in ['D0','E','S']:
   curves[method]={}
   for k in cfg['ks']:
    values=[u['stats'].get(str(k),{}).get(method) for u in unit]
    if any(v is None for v in values):continue
    curves[method][str(k)]={a:float(np.mean([v[a] for v in values])) for a in ['mean_regret_pp','p90_regret_pp','prob_regret_le3','success_pct']};curves[method][str(k)]['mean_regret_upper_pp']=float(np.mean([v['identified_regret_pp'][1] for v in values]))
   meanstar[method]=next((k for k in cfg['ks'] if str(k) in curves[method] and curves[method][str(k)]['mean_regret_upper_pp']<=3),None);reliablestar[method]=next((k for k in cfg['ks'] if all(u['stats'].get(str(k),{}).get(method,{}).get('prob_regret_le3',0)>=.9 for u in unit)),None)
  results[sc]=dict(evidence='Descriptive target fewshot simulation, disjoint calibration/evaluation states',controllers=unit,curves=curves,mean_kstar=meanstar,all_controller_reliability_kstar=reliablestar,calibration_original_uids=[original[ids[i]] for i in cal],evaluation_original_uids=[original[ids[i]] for i in ev],menu=MENU,per_arm_known_label_budget=True,unobserved_unknown_query_cost_not_zero=True,no_successful_grid_means_greater_than16_per_arm=True,oracle='In-sample7-arm oracle on disjoint evaluation16states; does not use calibration labels to evaluate')
 write(HERE/'FEWSHOT_RESULTS.json',results)
 import csv
 with (HERE/'FEWSHOT_CURVES.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 print('Fewshot complete',flush=True)
if __name__=='__main__':main()
