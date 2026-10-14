"""Budget minima, crossed-bootstrap decision variability and paired contrasts."""
from core import *
from evaluate import descriptors

def e3_bootstrap(bi):
    gate();lab=np.load(R4/'LABELS.npz')['outcomes'][:,128:,:,4:8];Y=(lab==1).mean(-1);U=(lab==-1).mean(-1);boot=np.load(HERE/'bootstrap_main.npz');sw=boot['state_weights'];rw=boot['repeat_weights'];cols,nm=descriptors(METHODS);possible=np.zeros((5000,nm));certain=possible.copy();anytarget=possible.copy()
    for target in range(7):
        sim=np.load(HERE/'simulations'/f'{target}.npz');vals=[];upp=[]
        for c in cols[:nm]:
            a=sim['choices'][STRATEGIES.index(c['strategy']),bi,METHODS.index(c['method'])];b=np.full_like(a,10);lo,hi=contrast(Y[target][None,None],U[target][None,None],a,b);vals.append(lo.mean(1));upp.append(hi.mean(1))
        vl=np.array(vals);vh=np.array(upp)
        for start in range(0,5000,250):
            stop=min(start+250,5000);low=(sw[start:stop]@vl.reshape(nm*200,64).T).reshape(stop-start,nm,200);high=(sw[start:stop]@vh.reshape(nm*200,64).T).reshape(stop-start,nm,200);pl=low<=-.10+1e-12;pc=high<=-.10+1e-12
            weights=rw[target,start:stop,None,:];possible[start:stop]+=np.sum(pl*weights,2)/7;certain[start:stop]+=np.sum(pc*weights,2)/7;anytarget[start:stop]+=np.any(pl&(weights>0),2)/7
    np.savez_compressed(HERE/'endpoints'/f'main_B{int(BUDGETS[bi])}_E3_bootstrap.npz',possible=possible,certain=certain,any_sample_target_possible=anytarget)
    print('E3 bootstrap complete',int(BUDGETS[bi]),flush=True)

def minimum(flags):
    idx=np.flatnonzero(flags);return int(BUDGETS[idx[0]]) if len(idx) else '>128'
def minima_boot(flags):
    result=np.full(flags.shape[1],129,int)
    for bi in range(6,-1,-1):result[flags[bi]]=int(BUDGETS[bi])
    return result
def dist(values):
    counts={str(int(b)):int((values==b).sum()) for b in [0,4,8,16,32,64,128,129]};q=np.quantile(values,[.025,.5,.975],method='nearest')
    return dict(counts=counts,right_censored_rate=float((values==129).mean()),quantiles=[int(x) if x<129 else '>128' for x in q],coding='129 represents >128 right censoring, not observed budget129')

def main():
    gate();res=[read(HERE/'endpoints'/f'main_B{B}.json') for B in BUDGETS]
    packs=[];cat=[]
    for B in BUDGETS:
        with np.load(HERE/'endpoints'/f'main_B{B}_bootstrap.npz') as z:packs.append({k:z[k] for k in z.files})
        with np.load(HERE/'endpoints'/f'main_B{B}_E3_bootstrap.npz') as z:cat.append({k:z[k] for k in z.files})
    columns=json.loads(str(packs[0]['columns_json']));nm=int(packs[0]['main_method_columns']);keys=[c['key'] for c in columns[:nm]];specs=json.loads(str(packs[0]['specs_json']));budgets=[];bytarget=[];boot_decisions={};budget_draws={}
    for mi,key in enumerate(keys):
        flags_spt=[];flags_ct=[];point_spt=[];point_ct=[]
        for bi,r in enumerate(res):
            s=next(s for s in r['summaries'] if s['key']==key);point_spt.append(s['SPT']);point_ct.append(s['CT']);i1=next(i for i,x in enumerate(specs) if x['key']==key and x['metric']=='E1');i2=next(i for i,x in enumerate(specs) if x['key']==key and x['metric']=='E2_combined');samples=packs[bi]['spec_bootstrap'];obs=packs[bi]['spec_point'];se=samples.std(0,ddof=1);critical=r['maxT_critical']
            # Fixed-width max-T decision bootstrap: outer resampling shifts effect
            # centers, retains the registered CI width; does not retune a policy.
            safe=(samples[:,i1,0]-critical*se[i1,0]>-.02)&(samples[:,i2,0]-critical*se[i2,0]>0)&(cat[bi]['possible'][:,mi]==0)
            base=next(i for i,x in enumerate(specs) if x['key']==key and x['metric']=='E2_base_u');v10=next(i for i,x in enumerate(specs) if x['key']==key and x['metric']=='E2_v10_local');flags_spt.append(safe);flags_ct.append(safe&(samples[:,base,0]>0)&(samples[:,v10,0]>0))
            s['E3']['ci95_bootstrap_possible']=np.quantile(cat[bi]['possible'][:,mi],[.025,.975]).tolist();s['E3']['ci95_bootstrap_any_sample']=np.quantile(cat[bi]['any_sample_target_possible'][:,mi],[.025,.975]).tolist()
        ds=minima_boot(np.array(flags_spt));dc=minima_boot(np.array(flags_ct));boot_decisions[key]=(ds,dc);budget_draws[key+'__SPT']=ds;budget_draws[key+'__CT']=dc
        budgets.append(dict(key=key,method=columns[mi]['method'],strategy=columns[mi]['strategy'],B_SPT=minimum(point_spt),B_CT=minimum(point_ct),B_SPT_bootstrap=dist(ds),B_CT_bootstrap=dist(dc)))
        for target in range(7):
            point=[];cert=[];bs=[];ci=next(i for i,c in enumerate(columns) if c['kind']=='oracle' and c['key']==key+'|vs_oracle')
            for bi,r in enumerate(res):
                row=next(x for x in r['rows'] if x['key']==key and x['target']==target);point.append(row['G']['within_2pp_point']);cert.append(row['G']['within_2pp_certified']);bs.append(packs[bi]['target_bootstrap'][target,:,ci,0]>=-.02)
            values=minima_boot(np.array(bs));bytarget.append(dict(key=key,target=target,controller=read(HERE/'PROTOCOL.json')['controller_order'][target],necessary=target in (1,4),B_2pp=minimum(point),B_2pp_certified=minimum(cert),B_2pp_bootstrap=dist(values)));budget_draws[key+'__target'+str(target)+'__2pp']=values
    # Keep original endpoint records intact; enriched E3 intervals are separate here.
    for bi,r in enumerate(res):write(HERE/'endpoints'/f'main_B{int(BUDGETS[bi])}_with_E3_CI.json',r)
    contrasts=[]
    for strategy in STRATEGIES:
        for safe in (False,True):
            suf='-safe' if safe else '';ref=strategy+'|E'+suf
            for method in ['H'+suf,'C-R0'+suf,'C-F2'+suf,'C-ctl-R0'+suf,'C-ctl-F2'+suf]:
                key=strategy+'|'+method;a=next(x for x in budgets if x['key']==key);b=next(x for x in budgets if x['key']==ref);effects=[]
                for r in res:
                    effects += [dict(budget=r['budget'],**x) for x in r['inferences'] if x['key']==strategy+'|'+method+'-'+'E'+suf and x['metric'] in ('paired_all','paired_necessary')]
                savings={}
                for metric in ('SPT','CT'):
                    aa=a['B_'+metric];bb=b['B_'+metric]
                    if isinstance(aa,int) and isinstance(bb,int):point=bb-aa
                    elif isinstance(aa,int):point='>'+str(128-aa)
                    elif isinstance(bb,int):point='<'+str(bb-128)
                    else:point='INCONCLUSIVE: both >128'
                    ea,eb=boot_decisions[key][0 if metric=='SPT' else 1],boot_decisions[ref][0 if metric=='SPT' else 1];joint=(ea<129)&(eb<129);diff=eb[joint]-ea[joint];at_budget=next((x for x in effects if x['metric']=='paired_all' and x['budget']==aa),None);supported=isinstance(aa,int) and (not isinstance(bb,int) or aa<bb) and at_budget is not None and at_budget['ci95_simultaneous'][0]>0
                    savings[metric]=dict(point_budget_saved=point,both_uncensored_bootstrap_rate=float(joint.mean()),ci95_budget_saved_conditional_both_uncensored=np.quantile(diff,[.025,.975]).tolist() if len(diff) else None,source_right_censored_rate=float((ea==129).mean()),E_right_censored_rate=float((eb==129).mean()),supported_sample_efficiency=bool(supported),criterion='strictly earlier registered SPT/CT plus paired all-target simultaneous CI lower>0 at that source method budget')
                contrasts.append(dict(strategy=strategy,source_method=method,target_only_reference='E'+suf,budget_savings=savings,paired_success_curves=effects))
    state_value=[]
    for r in res:
        for x in r['inferences']:
            if x['metric']=='paired_all' and any(x['key']==st+'|'+a+'-'+b for st in STRATEGIES for a,b in [('C-R0','C-ctl-R0'),('C-R0-safe','C-ctl-R0-safe'),('C-F2','C-ctl-F2'),('C-F2-safe','C-ctl-F2-safe')]):
                state_value.append(dict(budget=r['budget'],**x,verdict='SUPPORTED positive statewise value' if x['ci95_simultaneous'][0]>0 else 'SUPPORTED nonpositive statewise value' if x['ci95_simultaneous'][1]<=0 else 'INCONCLUSIVE'))
    np.savez_compressed(HERE/'BUDGET_BOOTSTRAP_DRAWS.npz',**budget_draws)
    write(HERE/'SAMPLE_EFFICIENCY.json',dict(utc=utc(),status='COMPLETE',budgets=budgets,target_B2pp=bytarget,source_knowledge_contrasts=contrasts,statewise_vs_controller=state_value,bootstrap_minima='5000 exact crossed outer draws; budget decision CIs use fixed original max-T widths, not nested bootstrap; right censoring retained',B2pp_minima='paired conservative point threshold; certified versions require budget-family simultaneous CI lower>=-2pp',right_censor_code=129))
    print(json.dumps(dict(SPT=[x for x in budgets if isinstance(x['B_SPT'],int)],source_efficiency_supported=[x['strategy']+'|'+x['source_method'] for x in contrasts if x['budget_savings']['SPT']['supported_sample_efficiency']]),ensure_ascii=False))
if __name__=='__main__':main() if sys.argv[1]=='finalize' else e3_bootstrap(int(sys.argv[1]))
