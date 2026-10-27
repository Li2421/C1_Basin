"""Counts and controller-level safeguards on the same paired evaluation set."""
from experiment import *
import numpy as np
import collections

def metrics(rows,qualification,scene,methods):
    eligible={q['controller'] for q in qualification if q['qualified'] and q['scene']==scene};lookup={};types={}
    for r in rows:
        c=r['case'];s=c['state'];ctl=c['controller_payload']['nominal'];name=ctl['controller']
        if name not in eligible or c['method'] not in methods:continue
        lookup[(name,s['N'],s['index'],c['seed'],c['method'])]=r;types[name]=ctl['type_code']
    states=sorted({(k[1],k[2]) for k in lookup});controllers=[]
    for name in sorted(types):
        ss=[s for s in states if all((name,*s,seed,m) in lookup for seed in (0,1) for m in methods)]
        if not ss:continue
        rates={};counts={};harms={};state_success={}
        for m in methods:
            rr=[lookup[(name,*s,seed,m)] for s in ss for seed in (0,1)]
            counts[m]=dict(episodes=len(rr),**{field:int(sum(r[field] for r in rr)) for field in ('success','collision','timeout','numerical_failure')})
            rates[m]={field:counts[m][field]/len(rr) for field in ('success','collision','timeout','numerical_failure')}
            rates[m]['success_upper']=min(1.,rates[m]['success']+rates[m]['numerical_failure'])
            state_success[m]=np.asarray([[lookup[(name,*s,seed,m)]['success'] for seed in (0,1)] for s in ss]).mean(-1)
            if m!='safety':
                base=[lookup[(name,*s,seed,'safety')] for s in ss for seed in (0,1)]
                harms[m]=dict(known=sum(bool(a['success'] and not b['success'] and not b['numerical_failure']) for a,b in zip(base,rr)),upper_including_unknown=sum(bool(a['success'] and not b['success']) for a,b in zip(base,rr)))
        # Same-state two-seed clusters; deterministic parity split is declared
        # here before computing oracle values. This is a descriptive reference.
        menu=['safety','fixed_eta10','fixed_eta13','human_progress'];oracle=None
        if all(m in methods for m in menu) and len(ss)>1:
            Y=np.stack([state_success[m] for m in menu],-1);halves=np.arange(len(ss))%2;total=0.;choices=[]
            for h in (0,1):
                fit=halves==h;test=~fit;choice=int(Y[fit].mean(0).argmax());total+=Y[test,choice].sum();choices.append(menu[choice])
            oracle=dict(menu=menu,crossfit=total/len(ss),in_sample_optimistic=float(Y.mean(0).max()),selected_by_training_half=choices,state_split='sorted (N,index) alternating halves; seeds never split',ci95=None,scope='descriptive baseline-plus-three-L1 menu; not the complete twelve-action oracle')
        controllers.append(dict(controller=name,type=types[name],states=len(ss),counts=counts,rates=rates,episode_harms=harms,L1_menu_reference=oracle))
    out=[]
    for label in [*sorted({c['type'] for c in controllers}),'type_equal']:
        cc=[c for c in controllers if label=='type_equal' or c['type']==label]
        if not cc:continue
        types_here=sorted({c['type'] for c in cc});typegroups=[[c for c in cc if c['type']==t] for t in types_here]
        for m in methods:
            equal={f:float(np.mean([np.mean([c['rates'][m][f] for c in group]) for group in typegroups])) for f in ('success','success_upper','collision','timeout','numerical_failure')}
            count={f:sum(c['counts'][m][f] for c in cc) for f in ('episodes','success','collision','timeout','numerical_failure')}
            row=dict(scene=scene,type=label,method=m,controller_count=len(cc),type_equal_rates=equal,pooled_counts=count)
            if m!='safety':
                row['episode_harms']={f:sum(c['episode_harms'][m][f] for c in cc) for f in ('known','upper_including_unknown')}
                row['controllers_at_least_10pp_below_safety']=sum(c['rates'][m]['success']<=c['rates']['safety']['success']-.1+1e-12 for c in cc)
                row['controllers_at_least_10pp_below_eta10']=sum(c['rates'][m]['success']<=c['rates']['fixed_eta10']['success']-.1+1e-12 for c in cc)
            out.append(row)
    return dict(endpoints=out,controllers=controllers,complete_menu_regret=dict(average=None,maximum=None,reason='Only chosen actions are evaluated on the independent evaluation states; the complete twelve-action matrix is unobserved. No extra rollout generated for an oracle.'))
