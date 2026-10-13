"""Preregistered Q16/new-state paired analysis; no candidate/state selection."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/name).read_text())
def write(name,obj):(HERE/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def bounds(y,u,axis=-1):return y.mean(axis), (y+u).mean(axis)
def diff_interval(a,b):return [float(a[0]-b[1]),float(a[1]-b[0])]
def wilson(success,n,z=1.959963984540054):
    if n==0:return [0.,1.]
    p=success/n;den=1+z*z/n;cen=(p+z*z/(2*n))/den
    r=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(max(0,cen-r)),float(min(1,cen+r))]
def boot_ci(v,rng,reps=10000):
    v=np.asarray(v);idx=rng.integers(0,len(v),(reps,len(v)))
    return np.quantile(v[idx].mean(1),[.025,.975]).tolist()

def crossfit(y,u,eta_indices=(1,2)):
    """2 controllers x 2 eta x 16 seeds; fixed train-lower choice rule."""
    terms=[]
    for train,test in ((slice(0,8),slice(8,16)),(slice(8,16),slice(0,8))):
        qt=y[:,:,train].mean(-1)
        own=np.argmax(qt,axis=1)
        blind=int(np.argmax(qt.mean(0)))
        l,h=bounds(y[:,:,test],u[:,:,test])
        point=[];low=[];high=[]
        for c in range(2):
            if own[c]==blind:point.append(0.);low.append(0.);high.append(0.)
            else:
                point.append(float(l[c,own[c]]-l[c,blind]))
                low.append(float(l[c,own[c]]-h[c,blind]))
                high.append(float(h[c,own[c]]-l[c,blind]))
        terms.append(dict(train='0-7' if train.start==0 else '8-15',
                          own_eta_indices=[eta_indices[int(j)] for j in own],
                          blind_eta_index=eta_indices[blind],point=float(np.mean(point)),
                          lower=float(np.mean(low)),upper=float(np.mean(high))))
    return dict(folds=terms,point=float(np.mean([t['point'] for t in terms])),
                lower=float(np.mean([t['lower'] for t in terms])),upper=float(np.mean([t['upper'] for t in terms])))

def main():
    p=read('protocol.json');cases=read('cases.json');post=read('postflight.json')
    assert post['total']==320 and post['missing']==post['quarantined']==0
    assert len(cases)==320 and p['eta_indices']==[1,2]
    names=p['controller_order'];states=p['states'];stateids=[s['state_uid'] for s in states]
    y=np.zeros((2,5,2,16));u=np.zeros_like(y);outcomes={};raw_records={}
    for c in cases:
        r=read('raw/'+c['rollout_uid']+'.json')
        raw_records[c['rollout_uid']]=r
        assert r['case']==c and r['state_uid']==c['state_uid']
        ci=names.index(c['controller_name']);si=stateids.index(c['state_uid']);ei=p['eta_indices'].index(c['eta_index']);seed=c['seed']
        y[ci,si,ei,seed]=r['success'];u[ci,si,ei,seed]=r['numerical_failure']
        outcomes[r['outcome']]=outcomes.get(r['outcome'],0)+1
    assert np.all(y+u<=1)
    rng=np.random.default_rng(2026101217)
    rows=[]
    for si,state in enumerate(states):
        q=[]
        for ci,name in enumerate(names):
            vals=[]
            for ei,eindex in enumerate(p['eta_indices']):
                s=int(y[ci,si,ei].sum());unknown=int(u[ci,si,ei].sum());valid=16-unknown
                vals.append(dict(eta_index=eindex,success=s,unknown=unknown,valid=valid,
                    Q16_interval=[s/16,(s+unknown)/16],observed_valid_wilson95=wilson(s,valid),
                    outcome_counts={term:sum(raw_records[c['rollout_uid']]['outcome']==term
                        for c in cases if c['controller_name']==name and c['state_uid']==state['state_uid'] and c['eta_index']==eindex)
                        for term in ('success','timeout','collision','deadlock','numerical_failure')}))
            q.append(vals)
        # eta2 - eta1 for each controller.
        d0=diff_interval(q[0][1]['Q16_interval'],q[0][0]['Q16_interval'])
        d1=diff_interval(q[1][1]['Q16_interval'],q[1][0]['Q16_interval'])
        interaction=[d0[0]-d1[1],d0[1]-d1[0]]
        qlo,qhi=bounds(y[:,si],u[:,si])
        v=.5*(qlo[0].max()+qlo[1].max())-.5*(qlo[0]+qlo[1]).max()
        vlo=max(0.,.5*(qlo[0].max()+qlo[1].max())-.5*(qhi[0]+qhi[1]).max())
        vhi=min(.5,.5*(qhi[0].max()+qhi[1].max())-.5*(qlo[0]+qlo[1]).max())
        paired=(y[0,si,1]-y[0,si,0])-(y[1,si,1]-y[1,si,0])
        draws=rng.integers(0,16,(10000,16));paired_ci=np.quantile(paired[draws].mean(1),[.025,.975]).tolist()
        cross=crossfit(y[:,si],u[:,si])
        rows.append(dict(state_index=si,state_uid=state['state_uid'],draw_seed=state['metadata']['draw_seed'],
            cells={names[0]:q[0],names[1]:q[1]},
            preference_eta2_minus_eta1={names[0]:d0,names[1]:d1},
            strict_opposite_preferences=bool(d0[0]>0 and d1[1]<0),
            interaction_interval=interaction,interaction_paired_seed_bootstrap95=paired_ci,
            V_Q16_plugin=float(v),V_unknown_safe_lower=float(vlo),V_unknown_safe_upper=float(vhi),
            crossfit=cross))
    # Fixed 5-state bootstrap; unknown-safe lower as preregistered gate.
    interactions=np.array([r['interaction_interval'] for r in rows]);cross=np.array([[r['crossfit'][k] for k in ('point','lower','upper')] for r in rows])
    ix=rng.integers(0,5,(10000,5))
    int_lower_ci=np.quantile(interactions[ix,0].mean(1),[.025,.975]).tolist()
    cross_point_ci=np.quantile(cross[ix,0].mean(1),[.025,.975]).tolist()
    cross_lower_ci=np.quantile(cross[ix,1].mean(1),[.025,.975]).tolist()
    # Additional paired-seed sensitivity, not a change to preregistered gates.
    hierarchical=[]
    for _ in range(10000):
        selected_states=rng.integers(0,5,5)
        vals=[]
        for si in selected_states:
            seeds=rng.integers(0,16,16)
            z=y[:,si][:,:,seeds].mean(-1)
            vals.append(float((z[0,1]-z[0,0])-(z[1,1]-z[1,0])))
        hierarchical.append(np.mean(vals))
    hierarchical_ci=np.quantile(hierarchical,[.025,.975]).tolist()
    positive=int(np.sum(cross[:,0]>0));strict=int(sum(r['strict_opposite_preferences'] for r in rows))
    critic_confirmed=bool(interactions[:,0].mean()>0 and int_lower_ci[0]>0)
    selector_confirmed=bool(cross[:,1].mean()>0 and cross_lower_ci[0]>0 and positive>=3)
    result=dict(protocol_sha256=__import__('hashlib').sha256((HERE/'protocol.json').read_bytes()).hexdigest(),
        state_count=5,controllers=names,eta_indices=p['eta_indices'],seeds=16,
        new_continuations=sum(w['new'] for w in post['workers']),postflight={k:v for k,v in post.items() if k!='workers'},
        outcomes=outcomes,states=rows,
        mean_interaction_unknown_safe_interval=interactions.mean(0).tolist(),
        mean_interaction_state_bootstrap_lower95=int_lower_ci,
        mean_interaction_hierarchical_paired_seed_bootstrap95=hierarchical_ci,
        mean_V_Q16_plugin=float(np.mean([r['V_Q16_plugin'] for r in rows])),
        mean_V_unknown_safe_interval=[float(np.mean([r['V_unknown_safe_lower'] for r in rows])),
                                     float(np.mean([r['V_unknown_safe_upper'] for r in rows]))],
        mean_crossfit_point=float(cross[:,0].mean()),mean_crossfit_unknown_safe_interval=[float(cross[:,1].mean()),float(cross[:,2].mean())],
        crossfit_state_bootstrap_point95=cross_point_ci,crossfit_state_bootstrap_unknown_lower95=cross_lower_ci,
        positive_crossfit_states=positive,strict_opposite_preference_states=strict,
        exact_one_sided_sign_p_for_positive_crossfit=float(binomtest(positive,5,.5,alternative='greater').pvalue),
        critic_empirical_confirmation=critic_confirmed,selector_empirical_confirmation=selector_confirmed,
        note='Five new states; state bootstrap and sign count do not certify population necessity. Unknown-safe bounds retained.')
    write('necessity_results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='states'},indent=2))

if __name__=='__main__':main()
