"""C0 exact Ring controller audit; database read-only, unknown outcomes retained."""
import sys,json,sqlite3,itertools,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from scipy.stats import spearmanr
from study import OUT,OLD,read,write,sha
from analyze_learning import ci
DB=OUT.parents[1]/'shared_rollout_db/rollout.sqlite'
def tensor_metrics(y):
 # C,S,E,K; unknown=-1. Independent halves and swap; bounds do not call unknown a failed label.
 c,s,e,k=y.shape;lo=(y==1).astype(float);un=(y<0).astype(float);records=[];energy=[]
 for flip in [0,1]:
  a=np.arange(k//2) if not flip else np.arange(k//2,k);b=np.arange(k//2,k) if not flip else np.arange(k//2)
  qa=lo[:,:,:,a].mean(-1);qb=lo[:,:,:,b].mean(-1);ub=un[:,:,:,b].mean(-1)
  fixed=qa.mean(1).argmax(1);choice=qa.argmax(2)
  for target in range(c):
   j=fixed[target];idx=np.arange(s);own=qb[target,idx,choice[target]];unknown=ub[target,idx,choice[target]];base=qb[target,:,j];bu=ub[target,:,j];dg=own-base
   rec=dict(half=flip,target=target,best_eta_index=int(j),fixed_success=float(base.mean()),state_oracle_success_lower=float(own.mean()),state_oracle_success_upper=float((own+unknown).mean()),state_gain=float(dg.mean()),state_gain_ci95=ci(dg),state_gain_worst_unknown_ci95=ci(dg-bu),cross_controller=[])
   for source in range(c):
    if source==target:continue
    reg=base-qb[target,:,fixed[source]];mm=own-qb[target,idx,choice[source]]
    rec['cross_controller'].append(dict(source=source,fixed_regret=float(reg.mean()),fixed_regret_ci95=ci(reg),matched_minus_mismatched_oracle=float(mm.mean()),matched_minus_mismatched_ci95=ci(mm),same_discovery_best=bool(fixed[source]==j)))
   records.append(rec)
  if c==2:
   da=qa[1]-qa[0];db=qb[1]-qb[0]
   A=da.mean(0)-da.mean();B=db.mean(0)-db.mean()
   ia=da-da.mean(0,keepdims=True)-da.mean(1,keepdims=True)+da.mean();ib=db-db.mean(0,keepdims=True)-db.mean(1,keepdims=True)+db.mean()
   energy.append(dict(level_difference=float(db.mean()),level_ci95=ci(db.mean(1)),controller_eta_energy=float(np.mean(A*B)),state_controller_eta_energy=float(np.mean(ia*ib))))
 return dict(n_controllers=c,states=s,etas=e,seeds=k,unknown=int((y<0).sum()),crossfit=records,independent_half_energies=energy,energy_scope='lower-bound endpoint contrasts; unbiased Bernoulli contrast energy only when no unknown; negative estimates retained',global_eta_spearman=float(spearmanr(lo[0].mean((0,2)),lo[1].mean((0,2))).statistic) if c==2 and np.std(lo[0].mean((0,2))) and np.std(lo[1].mean((0,2))) else None)
def main():
 db=sqlite3.connect(f'file:{DB}?mode=ro',uri=True);db.row_factory=sqlite3.Row
 rows=[dict(r) for r in db.execute("select c.* from controller_config c join scenario sc on sc.scenario_uid=c.scenario_uid where sc.name='ring_exchange'")]
 counts={r[0]:dict(rows=r[1],states=r[2],etas=r[3],unknown=r[4],quarantine=r[5]) for r in db.execute("select r.controller_uid,count(*),count(distinct state_uid),count(distinct eta_uid),sum(numerical_failure),sum(conflict_quarantined) from rollout r join controller_config c using(controller_uid) join scenario sc on sc.scenario_uid=c.scenario_uid where sc.name='ring_exchange' group by r.controller_uid")}
 catalog=[]
 for r in rows:
  q=json.loads(r['config_json']);r['payload']=q;r.update(counts.get(r['controller_uid'],{}));catalog.append(r)
 write(OUT/'C0_CONTROLLER_CATALOG.json',catalog)
 native=read(OLD/'protocol.json');cid=native['controller']['controller_uid'];scenario=native['controller']['payload']['scenario_uid'];controllers={r['controller_uid']:r for r in catalog if r['scenario_uid']==scenario and r['compatibility_quality']=='EXACT_PROFILE' and r.get('rows',0)}
 cells=defaultdict(lambda:defaultdict(dict));uids=defaultdict(dict);quarantined=Counter()
 for r in db.execute('SELECT rollout_uid,state_uid,eta_uid,controller_uid,seed_key,success,numerical_failure,conflict_quarantined FROM rollout WHERE controller_uid IN ('+','.join('?'*len(controllers))+')',list(controllers)):
  ru,s,e,c,k,suc,nu,qu=r
  if qu:quarantined[c]+=1;continue
  cells[c][s,e][k]=-1 if nu else int(suc);uids[c][s,e,k]=ru
 eta_table={r['eta_uid']:r for r in db.execute('select * from eta')};print('eta_columns',list(next(iter(eta_table.values())).keys()),flush=True)
 # Canonical K16 identity comes from archived cases, keeping exact float hashes.
 cases=read(OLD/'cases.json');emap={int(r['eta_index']):r['eta_uid'] for r in cases};K=[emap[j] for j in range(16)];strong=K[13]
 results=[];coverage=[];allused=set();pd=OUT/'controller_panels';pd.mkdir(exist_ok=True)
 for ca,cb in itertools.combinations(controllers,2):
  aa,bb=controllers[ca],controllers[cb]
  if any(aa[k]!=bb[k] for k in ['horizon','dt','safety_config_hash','success_semantics_version','rng_semantics_version']):continue
  common=set(cells[ca])&set(cells[cb]);shared={se:set(cells[ca][se])&set(cells[cb][se]) for se in common}
  for minq in [16,8,4,2]:
   bys=defaultdict(set)
   for (s,e),kk in shared.items():
    if len(kk)>=minq and e in K:bys[s].add(e)
   if not bys:continue
   patterns={tuple(K[j] for j in range(16) if K[j] in es) for es in bys.values()}
   patterns.add(tuple(K));patterns.add(tuple(K[j] for j in [0,5,10,15]))
   best=None
   for es in patterns:
    if len(es)<4:continue
    ss=[s for s,ee in bys.items() if set(es)<=ee and len(set.intersection(*(shared[s,e] for e in es)))>=minq]
    if len(ss)<8:continue
    score=(int(strong in es),len(ss)*len(es))
    if best is None or score>best[0]:best=(score,sorted(ss),list(es))
   if best is None:continue
   _,ss,ee=best;z=np.full((2,len(ss),len(ee),minq),-1,np.int8);ru=[];seedlist=[]
   for i,s in enumerate(ss):
    keys=sorted(set.intersection(*(shared[s,e] for e in ee)),key=lambda k:(json.loads(k).get('future_index',0),k))[:minq];seedlist.append(keys)
    for h,c in enumerate([ca,cb]):
     for j,e in enumerate(ee):
      for l,key in enumerate(keys):z[h,i,j,l]=cells[c][s,e][key];ru.append(uids[c][s,e,key])
   pid=hashlib.sha256((ca+cb+str(minq)).encode()).hexdigest()[:16];np.savez_compressed(pd/(pid+'.npz'),Y=z,state_uids=np.array(ss),eta_uids=np.array(ee),controller_uids=np.array([ca,cb]),seed_keys=np.array(seedlist),rollout_uids=np.array(ru));allused.update(ru)
   family=[controllers[c]['payload'].get('response_family',controllers[c]['payload'].get('chain','native')) for c in [ca,cb]]
   res=dict(panel=pid,controllers=[ca,cb],families=family,contains_eta13=strong in ee,eta_bank_indices=[K.index(e) for e in ee],conditioning=[controllers[c]['conditioning_version'] for c in [ca,cb]],file=str(pd/(pid+'.npz')),**tensor_metrics(z));results.append(res)
   # Highest available precision per pair; record lower-q wider coverage separately in coverage index only.
   break
 write(OUT/'C0_RESULTS.json',dict(status='COMPLETED',ring_controllers=len(catalog),exact_native_geometry_controllers=len(controllers),catalog_path=str(OUT/'C0_CONTROLLER_CATALOG.json'),pair_panels=results,unique_rollouts=len(allused),quarantined=dict(quarantined),new_rollouts=0,selection='support-only panel choice, prefer inclusion of eta13 then area; highest seed precision first; not strongest outcome panel',limitations=['Many comparisons exploratory and dependent; no multiplicity-free positive gate claim.','Exact same state_uid and seed_key required.','All native geometry controllers retained, including analytic targets; interpretation must condition on controller type.']))
 write(OUT/'C0_ROLLOUT_INDEX.json',sorted(allused));print('C0',len(catalog),len(results),len(allused),flush=True)
if __name__=='__main__':main()
