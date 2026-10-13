"""Real checkpoint evidence inventory and matched committed-action sensitivity, NOT C3 free-t0 confirmation."""
import sqlite3,json,itertools
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from study import OUT,read,write
from controller_audit import tensor_metrics

def main():
 cat=read(OUT/'C0_CONTROLLER_CATALOG.json');native='11ea8860e05ea19aef5c10ec12041c592d69460260e28a4f726af4533d19bcda';db=sqlite3.connect('file:'+str(OUT.parents[1]/'shared_rollout_db/rollout.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
 groups=defaultdict(list);assets=[]
 for r in cat:
  if r.get('rows',0) and r['flow_checkpoint_sha256']!=native and not str(r['flow_checkpoint_sha256']).startswith('NONE'):
   paths=[]
   for row in db.execute('SELECT original_source_file FROM rollout WHERE controller_uid=? LIMIT 1',(r['controller_uid'],)):
    paths.append(row[0])
   assets.append(dict(controller=r['controller_uid'],checkpoint_sha256=r['flow_checkpoint_sha256'],rows=r['rows'],quality=r['compatibility_quality'],conditioning=r['conditioning_version'],source_examples=paths))
  if r.get('rows',0) and r['compatibility_quality']=='EXACT_PROFILE' and r['conditioning_version']=='training_dev_parent_initial_state_episode_eta_v1+committed_base_t0_then_alt_future_v1':
   sig=tuple(r[k] for k in ['scenario_uid','horizon','dt','safety_config_hash','success_semantics_version','rng_semantics_version']);groups[sig].append(r)
 panels=[];used=set();folder=OUT/'legacy_flow_panels';folder.mkdir(exist_ok=True)
 for gi,(sig,controls) in enumerate(groups.items()):
  if len(controls)<2:continue
  cells=defaultdict(lambda:defaultdict(dict));uids={}
  for c in controls:
   for rr in db.execute('select rollout_uid,state_uid,eta_uid,seed_key,success,numerical_failure,conflict_quarantined from rollout where controller_uid=?',(c['controller_uid'],)):
    ru,s,e,k,suc,nu,qu=rr
    if qu:continue
    cells[c['controller_uid']][s,e][k]=-1 if nu else suc;uids[c['controller_uid'],s,e,k]=ru
  # All checkpoint pairs with ≥4 states, ≥4eta,≥16seeds, no outcome filtering.
  for a,b in itertools.combinations(controls,2):
   if a['flow_checkpoint_sha256']==b['flow_checkpoint_sha256']:continue
   ca,cb=a['controller_uid'],b['controller_uid'];common=set(cells[ca])&set(cells[cb]);sk={se:set(cells[ca][se])&set(cells[cb][se]) for se in common};by=defaultdict(set)
   for (s,e),k in sk.items():
    if len(k)>=16:by[s].add(e)
   if len(by)<4:continue
   patterns={tuple(sorted(es)) for es in by.values() if len(es)>=2};best=None
   for ee0 in patterns:
    ss0=[s for s in sorted(by) if set(ee0)<=by[s] and len(set.intersection(*(sk[s,e] for e in ee0)))>=16]
    if len(ss0)>=4 and (best is None or len(ss0)*len(ee0)>best[0]):best=(len(ss0)*len(ee0),ss0,list(ee0))
   if best is None:continue
   _,ss,ee=best
   y=np.zeros((2,len(ss),len(ee),16),np.int8);ru=[]
   for i,s in enumerate(ss):
    kk=sorted(set.intersection(*(sk[s,e] for e in ee)))[:16]
    for h,c in enumerate([ca,cb]):
     for j,e in enumerate(ee):
      for l,k in enumerate(kk):y[h,i,j,l]=cells[c][s,e][k];ru.append(uids[c,s,e,k])
   fn=folder/f'panel{len(panels)}.npz';np.savez_compressed(fn,Y=y,state_uids=ss,eta_uids=ee,rollout_uids=ru);used.update(ru)
   panels.append(dict(controller_uids=[ca,cb],checkpoint_sha256=[a['flow_checkpoint_sha256'],b['flow_checkpoint_sha256']],file=str(fn),**tensor_metrics(y)))
 write(OUT/'C3_ASSET_AUDIT.json',dict(status='ASSET AUDIT COMPLETED; C3 TRANSFER CONDITIONAL ON C2',alternative_profiles=assets,compatible_free_t0_FF_alternative_profiles=0,legacy_committed_action_panels=panels,legacy_unique_rollouts=len(used),conditioning_boundary='These labels are Q conditional on a committed native-v10 first action/h0 and legacy projection/eta execution. They cannot be merged with free-t0 FF labels or used as C3 confirmation.',minimum_future_design='After C2 qualifies: keep fixed geometry and FF execution; ≥3 train Flow checkpoints +1 held-out; independent heldout64states K16 Q16=16384 heldout labels alone. Source per-controller state count from N1, labels Q4. Query descriptors without outcomes; no source checkpoint assigned to a new identity without hash.',new_rollouts=0));write(OUT/'C3_LEGACY_ROLLOUT_INDEX.json',sorted(used));print('legacy',len(assets),len(panels),len(used),flush=True)
if __name__=='__main__':main()
