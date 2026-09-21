"""Post-hoc waiting diagnostics; never used for candidate selection."""
import json
import numpy as np
from scipy.special import expit
from audit_c1_progress_conditioned_wait import OUT,BASE,SEED,DT,KAPPA,save

rows=json.loads((OUT/'rows.json').read_text())
groups={}
for label in ['success','deadlock','timeout']:
    rr=[r for r in rows if ('success' if r['outcome']['success'] else 'deadlock' if r['outcome']['deadlock'] else 'timeout')==label]
    ds=[np.load(OUT/'scores'/f"{r['rid']:04d}_{r['action']}.npz") for r in rr]
    cat=lambda key:np.concatenate([d[key] for d in ds])
    S=cat('S');Sn=cat('Snew');rate=cat('future_rate');low=cat('low_applied').astype(bool)
    groups[label]={}
    for name,mask in [('low',low),('low_no_progress',low&(rate<=0)),('low_progress_above_existing_threshold',low&(rate>.005))]:
        groups[label][name]=dict(frames=int(mask.sum()),retained_penalty_fraction=float(Sn[mask].sum()/S[mask].sum()) if mask.any() else None)

# A later, equally long window checks already established stagnation. It does not
# enter the t=5 decision, costs, or ranking. Successful episodes are not extended.
later=[]
for r in rows:
    if not r['outcome']['deadlock']:continue
    z=np.load(BASE/'traces'/f"{r['rid']:04d}_{SEED}_{r['action']}.npz")
    lo=500;end=700
    if len(z['success'])<end:continue
    initial=np.linalg.norm(z['positions_before'][0]-z['goals'],axis=1).sum()+1e-5
    Dend=np.linalg.norm(z['positions_after'][end-1]-z['goals'],axis=1).sum()/initial
    dist=np.linalg.norm(z['positions_before'][lo:end]-z['goals'],axis=2)
    U=1-np.prod(1-expit((dist-.08)/.02),axis=1)
    rate=(dist.sum(1)/initial-Dend)/((end-np.arange(lo,end))*DT)
    S=KAPPA**2/(KAPPA**2+np.sum((z['candidate'][lo:end]/.5)**2,axis=1))
    new=U*S*expit((.005-rate)/.00125)
    later.append(dict(rid=r['rid'],action=r['action'],S=float(S.mean()),Snew=float(new.mean()),
        actual_stagnation_fraction=float(z['candidate_deadlock'][lo:end].mean())))
save(OUT/'waiting_diagnostics.json',dict(scoring_window_groups=groups,
    late_window='25..35 seconds, post-hoc only, same10s length; no influence on selection',late_deadlock_branches=later,
    late_summary={k:float(np.mean([r[k] for r in later])) for k in ['S','Snew','actual_stagnation_fraction']},
    mathematical_zero_progress_retention=float(expit(4))))
print(json.dumps({'groups':groups,'late_summary':{k:float(np.mean([r[k] for r in later])) for k in ['S','Snew','actual_stagnation_fraction']}},indent=2))
