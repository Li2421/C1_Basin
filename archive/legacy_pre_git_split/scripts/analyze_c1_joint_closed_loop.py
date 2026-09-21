"""Paired analysis and trajectory aggregation falsification; no parameter fitting."""
import json
import numpy as np
from scipy.special import expit
from audit_c1_joint_witness_risk import ROOT, save, clean, aggregate, trajectory_values, PROTOCOL

OUT=ROOT/'results/c1_joint_closed_loop_audit'

def temporal(z,end):
    dt=float(z['dt']);dist=np.linalg.norm(z['positions_before'][:end]-z['goals'],axis=-1)
    V=dist.sum(1)/(dist[0].sum()+1e-5);W=round(4/dt)
    U=1-np.prod(1-expit((dist-.08)/.02),axis=1)
    p=np.full(end,np.nan)
    p[W:]=U[W:]*expit((.02-(V[:-W]-V[W:]))/.005)
    maxrisk=1+PROTOCOL['eta']*PROTOCOL['tau']*np.logaddexp(0.,(PROTOCOL['delta']+1)/PROTOCOL['tau'])
    g=z['risk'][:end]/maxrisk;valid=np.isfinite(g)
    lo=np.where(valid,g,0);hi=np.where(valid,g,1);weights=np.full(end,dt)
    P=aggregate(p,weights)
    # Bounds, not a definition of q at zero. Normalized risk is bounded by [0,1].
    return dict(exposure_lower=.5*aggregate(lo,weights)+.5*P,
        exposure_upper=.5*aggregate(hi,weights)+.5*P,
        missing_seconds=float((~valid).sum()*dt),
        geometry_integral_lower=float(lo.sum()*dt),geometry_integral_upper=float(hi.sum()*dt),
        stall_integral=float(np.nansum(p)*dt),
        # Raw, unnormalized integral prototype: no dilution by adding low-risk time.
        integral_lower=float(.5*(lo.sum()+np.nansum(p))*dt),
        integral_upper=float(.5*(hi.sum()+np.nansum(p))*dt),
        unfinished_seconds=float(U.sum()*dt))

def main():
    rows=json.loads((OUT/'episodes.json').read_text());pairs=[];delays=[];descent=[]
    for r in rows:
        rid=r['rid'];arm=r['arm'];path=OUT/'traces'/f'{rid:04d}_{arm}.npz'
        with np.load(path) as z:
            end=len(z['risk']);common0=temporal(z,end)
            if arm.startswith('descent'):
                changed=z['changed'].astype(bool)
                descent.append(dict(rid=rid,arm=arm,n_changed=int(changed.sum()),
                    n_non_decreasing=int(np.sum((z['risk'][changed]-z['risk_before'][changed])>=0))))
            if arm=='baseline':continue
            baseline=next(b for b in rows if b['rid']==rid and b['arm']=='baseline')
            with np.load(OUT/'traces'/f'{rid:04d}_baseline.npz') as b:
                common=min(end,len(b['risk']));baseT=temporal(b,len(b['risk']))
                pr=dict(rid=rid,arm=arm,before=baseline['outcome'],after=r['outcome'],
                    common_seconds=common*float(z['dt']),
                    common_exposure_change=trajectory_values(z,common)['exposure']-trajectory_values(b,common)['exposure'],
                    common_remaining_change=trajectory_values(z,common)['remaining_distance']-trajectory_values(b,common)['remaining_distance'],
                    full_before=baseT,full_after=common0)
                pairs.append(pr)
                if arm in ['wait_2s','loop_2s']:
                    n=min(len(b['risk']),end-40)
                    delays.append(dict(**pr,
                        delayed_path_max_error=float(np.max(np.abs(z['positions_after'][40:40+n]-b['positions_after'][:n]))),
                        extra_seconds=r['seconds']-baseline['seconds'],
                        naive_exposure_change=r['exposure']-baseline['exposure']))
    stats={}
    for arm in ['descent_002','descent_010','random_010']:
        pp=[p for p in pairs if p['arm']==arm]
        stats[arm]=dict(n=len(pp),
            success_lost=sum(p['before']=='success' and p['after']!='success' for p in pp),
            success_gained=sum(p['before']!='success' and p['after']=='success' for p in pp),
            common_exposure_decreased=sum(p['common_exposure_change']<0 for p in pp),
            common_remaining_improved=sum(p['common_remaining_change']<0 for p in pp),
            common_exposure_down_but_remaining_worse=sum(p['common_exposure_change']<0 and p['common_remaining_change']>0 for p in pp))
    save(OUT/'paired_analysis.json',dict(stats=stats,pairs=pairs,delayed_controls=delays,monotonic_descent_checks=descent,
        integral_note='Integral is an exploratory duration-aware cost, not a calibrated deadlock probability or validated training replacement. Undefined q is bounded, never imputed as actual zero risk.'))
    print(json.dumps(clean(dict(stats=stats,delays=delays)),indent=2))

if __name__=='__main__':main()
