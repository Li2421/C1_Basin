"""Why better controller-conditional mean Q need not improve statewise B15.

At a fully resolved state, B15 probability is monotonically increasing in p.
After averaging heterogeneous states, E[g(p)] != g(E[p]); eta order can change.
This diagnoses residual coarsening, not an intrinsic defect of Bernoulli NLL.
Only old source TRAIN labels define constant preferences. Opened confirmation
is used descriptively, never to select a checkpoint or alter the candidate set.
"""
import json,math
import numpy as np
from scipy.special import gammaln,logsumexp
from .state_support import OUT,read,write
from .support_confirmation import OUT as TEST
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import csvwrite

def choose_log(n,k):
    return -np.inf if k<0 or k>n else float(gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1))

def probability_from_observed_counts(s,f):
    n=s+f
    if n<16:return np.nan
    return float(np.exp(logsumexp([choose_log(s,16),math.log(f)+choose_log(s,15) if f else -np.inf])-choose_log(n,16)))

def main():
    d=np.load(OUT/'dataset.npz');test=np.load(TEST/'dataset.npz');rows=[];decisions=[];summary=[];uncertainty=[]
    train=(d['split']=='train')&(d['source']=='old46');states=sorted(set(d['state_index'][train]))
    for c,controller in enumerate(('alt','second')):
        stat=[]
        for e in (10,15):
            ii=np.flatnonzero(train&(d['eta_index']==e));s=d['success'][c,ii].astype(int);f=d['failure'][c,ii].astype(int)
            assert len(ii)==46
            q=s/(s+f);g=q**15*(16-15*q)
            u=np.array([probability_from_observed_counts(int(a),int(b)) for a,b in zip(s,f)])
            standard_s=d['standard_success'][c,ii];standard_f=d['standard_failure'][c,ii]
            st=dict(controller=controller,eta_index=e,source_states=46,source_Q_mean=float(q.mean()),
                source_trial_weighted_Q=float(s.sum()/(s+f).sum()),
                B15_of_source_mean_Q=float(q.mean()**15*(16-15*q.mean())),
                mean_of_statewise_B15_plugin=float(g.mean()),
                source_B15_Ustat_mean=float(np.nanmean(u)),Ustat_unavailable_states=int(np.isnan(u).sum()),
                source_standard_B15=int((standard_s>=15).sum()),source_standard_nonB15=int((standard_f>=2).sum()),
                source_standard_unknown=int(((standard_s<15)&(standard_f<2)).sum()))
            jj=np.flatnonzero(test['eta_index']==e);ts=test['success'][c,jj];tf=test['failure'][c,jj]
            st.update(confirmation_Q_mean=float((ts/(ts+tf)).mean()),confirmation_B15=int((ts>=15).sum()),confirmation_nonB15=int((tf>=2).sum()))
            rows.append(st);stat.append(st)
        by_mean=int(np.argmax([r['source_trial_weighted_Q'] for r in stat]));by_robust=int(np.argmax([r['source_B15_Ustat_mean'] for r in stat]))
        summary.append(dict(controller=controller,source_mean_Q_preferred_eta=(10,15)[by_mean],
            source_B15_probability_preferred_eta=(10,15)[by_robust],preference_reversal=by_mean!=by_robust))
        for rule,choice in (('source_controller_mean_Q',by_mean),('source_controller_B15_prior',by_robust),('fixed_eta10',0)):
            idx=np.flatnonzero(test['eta_index']==(10,15)[choice]);s=test['success'][c,idx];f=test['failure'][c,idx]
            decisions.append(dict(controller=controller,rule=rule,eta_index=(10,15)[choice],B15=int((s>=15).sum()),
                unknown=int(((s<15)&(f<2)).sum()),selected_Q=float((s/(s+f)).mean()),cases=64))
        ss=d['success'][c,train].reshape(46,2).astype(int);ff=d['failure'][c,train].reshape(46,2).astype(int)
        uu=np.array([[probability_from_observed_counts(int(a),int(b)) for a,b in zip(s,f)] for s,f in zip(ss,ff)])
        valid=np.isfinite(uu).all(1);qq=ss/(ss+ff)
        ts=test['success'][c].reshape(64,2);tf=test['failure'][c].reshape(64,2);known=((ts>=15)|(tf>=2)).all(1)
        samples=[('source_B15_Ustat',uu[valid,0]-uu[valid,1]),('source_Q',(qq[:,0]-qq[:,1])[valid]),
            ('confirmation_B15',(ts[known,0]>=15).astype(float)-(ts[known,1]>=15)),
            ('confirmation_Q',(ts/(ts+tf))[:,0]-(ts/(ts+tf))[:,1])]
        rng=np.random.default_rng(20261004101+c)
        for label,values in samples:
            boots=rng.integers(len(values),size=(10000,len(values)));ci=np.quantile(values[boots].mean(1),[.025,.975])
            uncertainty.append(dict(controller=controller,contrast='eta10_minus_eta15',estimand=label,
                families=len(values),mean=float(values.mean()),CI_low=float(ci[0]),CI_high=float(ci[1])))
    csvwrite(OUT/'coarsened_Q_vs_B15.csv',rows);csvwrite(OUT/'coarsened_decision_confirmation.csv',decisions)
    csvwrite(OUT/'coarsened_preference_uncertainty.csv',uncertainty)
    doc=dict(classification='CONTROLLER_MEAN_Q_AND_STATEWISE_ROBUSTNESS_PREFERENCES_CAN_DIFFER',source_preferences=summary,
        mathematical_identity='g(p)=p^15*(16-15p) is increasing, but E_h[g(p_h)] != g(E_h[p_h]); averaging states before the nonlinear robust criterion can reverse eta order.',
        source_Ustat='[choose(s,16)+f*choose(s,15)]/choose(s+f,16), complete16 or stronger observed valid continuation evidence; no synthetic unobserved seeds.',
        caveats=['Descriptive posthoc adjudication on an already opened independent panel, not a newly optimized model.',
            'U-statistic assumes exchangeable Bernoulli continuations and excludes numerical attempts; unavailable n<16 is explicitly counted.',
            'This does not refute pointwise probability learning: perfectly resolved Q(h,eta,C) ranks B15 probability consistently with Q.',
            'Known-controller empirical prior is privileged; it is not an unseen-controller or cross-scene solution.',
            'Observed preference reversals are not automatically significant population reversals: source and confirmation family-bootstrap intervals are reported separately. Do not turn a directional mechanism into a proven sole cause.',
            'It explains the direction of coarse prediction/selection, not every individual neural error.'],
        inference='When state residuals remain poorly resolved, useful controller information can improve average Q and NLL while moving selection away from robust-tail eta. This is not sufficient evidence that ranking loss is needed.')
    write(OUT/'coarsened_decision_adjudication.json',doc);print(json.dumps(doc,indent=2));print(decisions)

if __name__=='__main__':main()
