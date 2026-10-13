import sqlite3
import numpy as np
from study import OUT,read,write
from analyze_learning import ci
r=max(read(OUT/'C3_ASSET_AUDIT.json')['legacy_committed_action_panels'],key=lambda r:r['states']*r['etas']);a=np.load(r['file']);y=a['Y'];lo=(y==1).astype(float);u=(y<0).astype(float);rows=[]
for target in [0,1]:
 d=[];low=[];hi=[];chosen=[]
 for flip in [0,1]:
  aa=slice(0,8) if not flip else slice(8,16);bb=slice(8,16) if not flip else slice(0,8);q=lo[:,:,:,aa].mean(-1);v=lo[target,:,:,bb].mean(-1);uu=u[target,:,:,bb].mean(-1);j=q.mean(1).argmax(-1);diff=v[:,j[target]]-v[:,j[1-target]];d.append(diff);low.append(diff-uu[:,j[1-target]]);hi.append(diff+uu[:,j[target]]);chosen.append(j.tolist())
 v=np.mean(d,0);vl=np.mean(low,0);vh=np.mean(hi,0);rows.append(dict(target=target,crossfit_fixed_regret=float(v.mean()),ci95=ci(v),simultaneous136_ci=ci(v,.05/136),censor_bounds=[float(vl.mean()),float(vh.mean())],worst_unknown_ci95=ci(vl),discovery_best_indices=chosen))
db=sqlite3.connect('file:'+str(OUT.parents[1]/'shared_rollout_db/rollout.sqlite')+'?mode=ro',uri=True);eta=[dict(uid=str(e),physical=db.execute('select eta1,eta2,eta3 from eta where eta_uid=?',(str(e),)).fetchone()) for e in a['eta_uids']];checkpoints=read(OUT/'FLOW_CHECKPOINT_INDEX.json');cp=[next((c['path'] for c in checkpoints if c['sha256']==h),h) for h in r['checkpoint_sha256']]
write(OUT/'LEGACY_FLOW_FOCUS.json',dict(states=r['states'],etas=eta,seeds=16,checkpoints=cp,controller_uids=r['controller_uids'],unknown=r['unknown'],rows=rows,scope='Support-largest legacy panel selected by coverage; exploratory cross-controller result conditional on committed first action and legacy projection, not current nativeFF free-t0 C3. Two candidate pool differs from current K16, no current eta13-optimality claim.'));print(cp,eta,rows)
