"""Analyze full C1 only; all original K16 retained, disjoint seed halves."""
import sys,json,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from collections import Counter
import numpy as np
from diagnostics.ring_state_selection_transfer_v1.study import read,write,sha
from diagnostics.ring_state_selection_transfer_v1.analyze_learning import ci
from shared_rollout_db.src.rollout_db import connect
HERE=Path(__file__).resolve().parent

def main():
 p=read(HERE/'protocol.json');cases=read(HERE/'cases.json');names=list(p['kappa']);states=[s['state_uid'] for s in p['states']];y=np.full((6,32,16,8),-2,np.int8);length=np.zeros(y.shape,int);provenance=[];unknown=Counter();dur=[];rawnew=0
 with connect(True) as db:
  for c in cases:
   r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert r is not None,('MISSING',c['rollout_uid']);assert not r['conflict_quarantined']
   i=names.index(c['controller_name']);s=states.index(c['state_uid']);j=c['eta_index'];k=c['seed'];assert y[i,s,j,k]==-2
   y[i,s,j,k]=-1 if r['numerical_failure'] else r['success'];length[i,s,j,k]=r['episode_length'];provenance.append(c['rollout_uid'])
   if r['numerical_failure']:unknown[names[i]]+=1
 assert (y!=-2).all();np.savez_compressed(HERE/'labels.npz',Y=y,episode_length=length,controller_names=names,state_uids=states,eta=np.array(p['eta']),rollout_uids=provenance)
 lo=(y==1).astype(float);un=(y<0).astype(float);scores={};choice_records=[]
 for tie in ['discovery_global_prior','bank_order']:
  per=[]
  for flip in [0,1]:
   a=slice(0,4) if flip==0 else slice(4,8);b=slice(4,8) if flip==0 else slice(0,4);q=lo[:,:,:,a].mean(-1);v=lo[:,:,:,b].mean(-1);uv=un[:,:,:,b].mean(-1);fixed=q.mean(1).argmax(-1);state=[]
   for c in range(6):
    state.append(np.array([np.lexsort((np.arange(16),-q[c].mean(0),-row))[0] for row in q[c]]) if tie=='discovery_global_prior' else q[c].argmax(-1))
   per.append(dict(q=q,v=v,u=uv,fixed=fixed,state=state))
   choice_records.append(dict(tie=tie,half=flip,fixed=fixed.tolist(),state=np.array(state).tolist()))
  for c,name in enumerate(names):
   own=[];upper=[];baseline=[];bupper=[]
   for h in per:
    idx=np.arange(32);own.append(h['v'][c,idx,h['state'][c]]);upper.append(h['v'][c,idx,h['state'][c]]+h['u'][c,idx,h['state'][c]]);baseline.append(h['v'][c,:,h['fixed'][c]]);bupper.append(h['v'][c,:,h['fixed'][c]]+h['u'][c,:,h['fixed'][c]])
   own,upper,baseline,bupper=map(np.array,[own,upper,baseline,bupper]);gain=own-baseline;dg=gain.mean(0);worst=(own-bupper).mean(0);globalq=lo[c].mean((0,2));best=int(globalq.argmax())
   scores[tie,name]=dict(controller=name,kappa=p['kappa'][name],discovery_fixed_eta=[int(h['fixed'][c]) for h in per],hindsight_fixed_eta=best,fixed_crossfit_success=float(baseline.mean()),hindsight_best_fixed_success=float(globalq.max()),oracle_crossfit_success_lower=float(own.mean()),oracle_crossfit_success_upper=float(upper.mean()),gain=float(dg.mean()),gain_ci95=ci(dg),gain_by_split=gain.mean(1).tolist(),censor_gain_bounds=[float(worst.mean()),float((upper-baseline).mean())],gain_worst_unknown_ci95=ci(worst),unknown=int(un[c].sum()),hindsight_finite_seed_oracle=float(lo[c].mean(-1).max(1).mean()),all_eta_fail_fraction=float((lo[c].sum((1,2))==0).mean()))
   # Q8 coverage is exploratory: 7/8 proxy is NOT B15. Also expected-success≥.75 support.
   good=lo[c].mean(-1)>=.875;covered=good.any(-1);cover=None
   for k in range(1,17):
    if any(np.all(good[:,list(ix)].any(1)[covered]) for ix in itertools.combinations(range(16),k)):cover=k;break
   scores[tie,name].update(coverage_7of8_union=float(covered.mean()),coverage_7of8_intersection=float(good.all(-1).mean()),minimum_eta_cover_of_coverable_states=cover,coverage_not_B15=True)
  cross=[]
  for target in range(5):
   for source in range(5):
    if target==source:continue
    vals=[];worst=[];matched=[]
    for h in per:
     v,u=h['v'][target],h['u'][target];f=h['fixed'];idx=np.arange(32);d=v[:,f[target]]-v[:,f[source]];vals.append(d);worst.append(d-u[:,f[source]]);matched.append(v[idx,h['state'][target]]-v[idx,h['state'][source]])
    delta=np.mean(vals,axis=0);lower=np.mean(worst,axis=0);mm=np.mean(matched,axis=0)
    cross.append(dict(target=names[target],source=names[source],fixed_regret=float(delta.mean()),fixed_regret_ci95=ci(delta),fixed_regret_ci97_5=ci(delta,.025),fixed_regret_simultaneous20_ci=ci(delta,.05/20),worst_unknown_regret=float(lower.mean()),worst_unknown_ci97_5=ci(lower,.025),worst_unknown_simultaneous20_ci=ci(lower,.05/20),by_split=np.mean(vals,axis=1).tolist(),matched_minus_mismatched_state_oracle=float(mm.mean()),matched_minus_mismatched_ci95=ci(mm)))
  scores[tie,'cross']=cross
 primary=[scores['discovery_global_prior',name] for name in names];cross=scores['discovery_global_prior','cross'];extreme=[r for r in cross if {r['target'],r['source']}=={'rotate_negative','rotate_positive'}]
 gatea=any(r['fixed_regret']>=.05 and r['fixed_regret_ci97_5'][0]>0 and r['worst_unknown_ci97_5'][0]>0 for r in extreme)
 family_a=any(r['fixed_regret']>=.05 and r['fixed_regret_simultaneous20_ci'][0]>0 and r['worst_unknown_simultaneous20_ci'][0]>0 for r in cross)
 passing=[r['controller'] for r in primary[:5] if r['gain']>=.05 and r['gain_ci95'][0]>0 and r['gain_worst_unknown_ci95'][0]>0 and min(r['gain_by_split'])>0];gateb=len(passing)>=3
 duration=[]
 for f in (HERE/'raw').glob('*.json'):duration.append(read(f)['elapsed_seconds'])
 out=dict(status='COMPLETED',config=dict(states=32,eta=16,seeds=8,controller_names=names),primary=primary,cross_controller=cross,sensitivity_bank_order=[scores['bank_order',n] for n in names],choices=choice_records,gate=dict(A_extreme_fixed_regret_pass=gatea,A_family_simultaneous_fixed_regret_pass=family_a,B_headroom_at_least3_pass=gateb,headroom_controllers=passing,selection_transfer_qualified=(gatea or family_a) and gateb,interpretation='Failure is insufficient benchmark qualification; it is not proof of only-level controller dependence.'),unknown_total=int(un.sum()),new_rollouts=len(duration),existing_reused=len(cases)-len(duration),cpu_episode_hours=sum(duration)/3600,mean_episode_seconds=float(np.mean(duration)),labels_sha256=sha(HERE/'labels.npz'),controller_scope='Exact same states/geometry/nativeFlow/checkpoint/dynamics/safety/evaluator/eta/seed; only FF vector-field execution response changed. Native FF is anchor, not kappa=0.',continuous_basin_claim=False)
 write(HERE/'RESULTS.json',out);print(json.dumps(dict(gate=out['gate'],primary=primary,cpu_hours=out['cpu_episode_hours']),indent=2))
if __name__=='__main__':main()
