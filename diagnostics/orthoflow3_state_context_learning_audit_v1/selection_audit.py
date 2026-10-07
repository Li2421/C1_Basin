"""Certified B15 bounds and strong prior controls from source FIT only."""
import numpy as np
from .experiment import OUT,KINDS,SEEDS,read,write,csvwrite,metrics

def bounds(z,d,ix):
    cases=[]
    for c in range(2):
      for state in np.unique(d['state_index'][ix]):
        pos=np.flatnonzero(d['state_index'][ix]==state);i=int(pos[np.argmax(z[c,pos])])
        s=d['success'][c,ix[pos]];f=d['failure'][c,ix[pos]]
        good=s>=15;bad=f>=2;unknown=~good&~bad
        j=int(np.where(pos==i)[0][0])
        cases.append(dict(controller=c,state=int(state),selected_eta=int(d['eta_index'][ix[i]]),
            selected_certified_B15=bool(good[j]),selected_certified_nonB15=bool(bad[j]),selected_unresolved=bool(unknown[j]),
            oracle_certified=bool(good.any()),oracle_unresolved=bool(not good.any() and unknown.any())))
    return cases

def main():
    d=dict(np.load(OUT/'source_data_db.npz'));protocol=read(OUT/'protocol.json');rows=[];detail=[];priors=[]
    for kind in KINDS:
      for seed in SEEDS:
        cases=[]
        for fold in range(3):
            p=OUT/'models'/kind/f'fold{fold}_seed{seed}';a=np.load(p/'predictions.npz')
            cases.extend(bounds(a['best_correct'],d,a['indices']))
        rows.append(dict(kind=kind,seed=seed,cases=len(cases),
            B15_confirmed=sum(v['selected_certified_B15'] for v in cases),
            nonB15_confirmed=sum(v['selected_certified_nonB15'] for v in cases),
            unresolved=sum(v['selected_unresolved'] for v in cases),
            oracle_confirmed=sum(v['oracle_certified'] for v in cases),oracle_unresolved=sum(v['oracle_unresolved'] for v in cases)))
        detail.extend(dict(kind=kind,seed=seed,**v) for v in cases)
    for fold in range(3):
        sp=protocol['folds'][fold];fit=np.flatnonzero(np.isin(d['state_index'],sp['fit'])&np.isin(d['eta_index'],[10,15]))
        te=np.flatnonzero(np.isin(d['state_index'],sp['test'])&np.isin(d['eta_index'],[10,15]))
        for kind in ('source_eta_count_prior','known_controller_eta_count_prior'):
            z=np.zeros((2,len(te)))
            for c in range(2):
              for e in (10,15):
                ii=fit[d['eta_index'][fit]==e];controllers=[c] if kind.startswith('known_') else [0,1]
                s=d['success'][controllers][:,ii].sum();f=d['failure'][controllers][:,ii].sum()
                q=(s+1)/(s+f+2);z[c,d['eta_index'][te]==e]=np.log(q/(1-q))
            m,_=metrics(z,d,te);priors.append(dict(kind=kind,fold=fold,**m))
    csvwrite(OUT/'certified_selection_metrics.csv',rows);csvwrite(OUT/'certified_selection_states.csv',detail)
    csvwrite(OUT/'non_neural_prior_metrics.csv',priors)
    for kind in ('source_eta_count_prior','known_controller_eta_count_prior'):
        r=[v for v in priors if v['kind']==kind]
        print(kind,'NLL',np.average([v['NLL'] for v in r],weights=[v['n_pairs'] for v in r]),'B15',sum(v['B15'] for v in r),'strong',sum(v['strong_correct'] for v in r))
    print([r for r in rows if r['kind'] in ('eta_only','full_raw','wide_db_full_raw')])

if __name__=='__main__':main()
