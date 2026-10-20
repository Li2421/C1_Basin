"""Only three genuinely new controller types constitute confirmation."""
from common import *
from statistics_pipeline import evaluate_group,equivalence
from families import family,UNSEEN
import csv

def load(prefix):
    with np.load(HERE/f'{prefix}_EVAL_BUNDLE.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(HERE/f'{prefix}_EVAL_ANALYSIS_CHOICES.npz',allow_pickle=False) as z:methods={k:z[k] for k in z.files if k not in ('names','types')}
    data['uid_class']=equivalence(data['uids']);return data,methods

def run_group(data,methods,mask,fixed):return evaluate_group({k:v[mask] for k,v in data.items()},{k:v[mask] for k,v in methods.items()},True,5000,fixed)

def endpoints(report):
    pairs=report['paired_comparisons'];p=report['methods']['pipeline']
    def comparison(key,threshold=0):
        r=pairs.get(key)
        return dict(**r,passed=r['CI95_pp'][0]>threshold) if r is not None else dict(available=False)
    human=comparison('pipeline__minus__best_human_schedule',-1)
    if 'CI95_pp' in human:human['noninferior']=human.pop('passed');human['superior']=human['CI95_pp'][0]>0
    return dict(endpoint1=dict(**comparison('pipeline__minus__all_safety_baseline'),harm_possible=p['harm_possible'],harm_certain=p['harm_certain']),endpoint2=human,endpoint3=dict(**comparison('pipeline__minus__no_schedule'),selected_non_diagonal_fraction=p['selected_non_diagonal_fraction']),endpoint4=comparison('pipeline__minus__legacy_pipeline'),endpoint5=comparison('self_schedule__minus__fixed500'),endpoint6=comparison('pipeline__minus__no_controller'))

def descriptors():
    base=read(HERE/'BASE_PROTOCOL.json');eta13=np.asarray(base['eta'][13]);eta10=np.asarray(base['eta'][10]);direction=eta10-eta13;rows=[];switches={}
    records=read(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json')['states']
    for kind in sorted({r['type'] for r in records}):
        switches[kind]={}
        for method in ('self_schedule','fixed500','pipeline'):
            aa=[r['methods'][method] for r in records if r['type']==kind];counts={str(t):sum(not np.array_equal(a['eta_a'],a['eta_b']) and a['switch_step']==t for a in aa) for t in (200,300,400,500,600)}
            switches[kind][method]=dict(diagonal_no_effective_switch=sum(np.array_equal(a['eta_a'],a['eta_b']) for a in aa),non_diagonal_counts=counts,total=len(aa))
    for r in records:
        a=r['methods']['pipeline'];early=np.asarray(a['eta_a']);late=np.asarray(a['eta_b']);delta=late-early;norm=float(np.linalg.norm(delta));cos=float(np.clip(np.dot(delta,direction)/(norm*np.linalg.norm(direction)),-1,1)) if norm else None
        rows.append(dict(controller=r['controller'],type=r['type'],cohort=r['cohort'],state_index=r['state_index'],eta_early=early,eta_late=late,switch_step=a['switch_step'] if norm else None,diagonal=not norm,distance_early_eta13=float(np.linalg.norm(early-eta13)),distance_early_eta10=float(np.linalg.norm(early-eta10)),distance_late_eta13=float(np.linalg.norm(late-eta13)),distance_late_eta10=float(np.linalg.norm(late-eta10)),direction_cosine=cos,primary_NoC_same_action=r['methods']['pipeline']['candidate_index']==r['methods']['no_controller']['candidate_index']))
    write(HERE/'SELECTED_ETA_DESCRIPTIONS.json',dict(utc=utc(),rows=rows,switch_distributions=switches,human_reference_only=dict(eta13=eta13,eta10=eta10)))
    with np.load(HERE/'TRAINING_CONTROLLER_FEATURES.npz',allow_pickle=False) as z:source=z['features']
    with np.load(HERE/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:names=z['names'];types=z['types'];features=z['features']
    low=np.nanmin(source,0);high=np.nanmax(source,0);oob=[]
    for n,t,v in zip(names,types,features):
        if t not in UNSEEN:continue
        valid=np.isfinite(v);outside=((v<low)|(v>high))&valid
        oob.append(dict(controller=str(n),type=str(t),observed_dimensions=int(valid.sum()),out_of_range_dimensions=int(outside.sum()),out_of_range_fraction=float(outside.sum()/valid.sum()) if valid.any() else None,any_dimension_outside=bool(outside.any()),missing_dimensions=int((~valid).sum())))
    write(HERE/'FEATURE_OUT_OF_RANGE.json',dict(utc=utc(),rows=oob,training_minimum=low,training_maximum=high,reference='All104 source training controllers, first24 features; NaN not outside'))
    return rows,switches,oob

def transitions():
    out=[]
    for prefix,cohort in [('UNSEEN','unseen'),('KNOWN','known')]:
        if not (HERE/f'{prefix}_EVAL_BUNDLE.npz').exists():continue
        data,methods=load(prefix)
        if 'nominal_only' not in methods:continue
        for i,n in enumerate(data['names']):
            for si in range(data['labels'].shape[1]):
                a=methods['nominal_only'][i,si];b=methods['all_safety_baseline'][i,si]
                for q in (0,1):out.append(dict(cohort=cohort,controller=str(n),type=str(data['types'][i]),state_index=si,seed=q,nominal_label=int(data['labels'][i,si,a,q]),nominal_collision=int(data['collision'][i,si,a,q]),safety_label=int(data['labels'][i,si,b,q]),safety_collision=int(data['collision'][i,si,b,q]),safety_timeout=int(data['timeout'][i,si,b,q]),safety_unknown=int(data['unknown'][i,si,b,q])))
    if out:
        with (HERE/'NOMINAL_TO_SAFETY_TRANSITIONS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    return dict(rows=len(out),nominal_collision_cases=sum(r['nominal_collision'] for r in out),after_safety={k:sum(r['nominal_collision'] and r[k] for r in out) for k in ('safety_collision','safety_timeout','safety_unknown')},after_safety_success=sum(r['nominal_collision'] and r['safety_label']==1 for r in out),known_cohort='N/A: revised prompt omits nominal-only evaluation')

def main():
    assert (HERE/'NEW_EVAL_BUNDLE.receipt.json').exists();gate=read(HERE/'UNSEEN_CAPABILITY.json')['rows'];status={r['controller']:r for r in gate};reports={};tests={};gaps=[]
    if (HERE/'UNSEEN_EVAL_BUNDLE.npz').exists():
        data,methods=load('UNSEEN');types=data['types'];nc=len(types)
        for group in [*sorted(set(types.tolist())),'combined']:
            mask=np.ones(nc,bool) if group=='combined' else types==group;report=run_group(data,methods,mask,True);reports[group]=dict(type_equal=report);tests[group]=endpoints(report)
        no_repair=np.asarray([status[str(n)]['no_repair_needed'] for n in data['names']]);all_no_types={str(t) for t in set(types) if np.all(no_repair[types==t])};needs_type=~np.isin(types,list(all_no_types))
        if needs_type.any():reports['needs_repair_types']=dict(type_equal=run_group(data,methods,needs_type,True));tests['endpoint1_needs_repair_combined']=endpoints(reports['needs_repair_types']['type_equal'])['endpoint1']
        if no_repair.any():reports['no_repair_instances']=dict(type_equal=run_group(data,methods,no_repair,True))
        for t in UNSEEN:
            count=int((types==t).sum())
            if count<4:gaps.append(dict(type=t,eligible_instances=count,reason='Fewer than4 gate-eligible instances; no replacements'))
            if t in all_no_types:tests[t]['endpoint1'].update(excluded_from_repair_effectiveness=True,reason='All instances already pass safety >=98% at capability gate; retained for nonharm')
        reports['combined']['controller_equal_descriptive']=evaluate_group(data,methods,False,5000,False)
    else:gaps.append(dict(reason='No unseen controller passed capability; no replacements'))
    descriptions,switches,oob=descriptors();transition=transitions()
    result=dict(utc=utc(),evidence='CONFIRMATORY_TRULY_UNSEEN_CONTROLLER_TYPES',reports=reports,endpoints=tests,primary_method=read(HERE/'FROZEN_PIPELINE.json')['primary_method'],endpoint7=dict(selected_eta_file='SELECTED_ETA_DESCRIPTIONS.json',feature_range_file='FEATURE_OUT_OF_RANGE.json',nominal_to_safety=transition,oracle_scope='Actually evaluated non-nominal actions only; opposite-seed crossfit and in-sample both reported'),gaps=gaps,bootstrap_replicates=5000,bootstrap_seed=202610193201,unseen_combined='Fixed type strata, equal type weight; paired controller/state resampling, two seeds clustered',diagnostic_labels_read=False,posthoc_may_not_change_these_results=True)
    write(HERE/'CONFIRMATORY_RESULTS.json',result)
    seal('CONFIRMATORY_RESULTS.receipt.json',[HERE/n for n in ('CONFIRMATORY_RESULTS.json','SELECTED_ETA_DESCRIPTIONS.json','FEATURE_OUT_OF_RANGE.json','NOMINAL_TO_SAFETY_TRANSITIONS.csv','confirmatory_statistics.py','statistics_pipeline.py','NEW_EVAL_BUNDLE.receipt.json','NEW_EVAL_CHOICES.receipt.json') if (HERE/n).exists()],endpoints1_to7_sealed_before_posthoc_label_submission=True,confirmation_not_used_for_retuning=True)
    if (HERE/'KNOWN_EVAL_BUNDLE.npz').exists():
        data,methods=load('KNOWN');types=data['types'];known={}
        for group in [*sorted(set(types.tolist())),'combined','learned','optimization','search']:
            mask=np.asarray([group=='combined' or t==group or family(t)==group for t in types])
            if not mask.any():known[group]=dict(available=False,reason='Original nine learning types excluded by revised prompt');continue
            known[group]=dict(type_equal=run_group(data,methods,mask,False))
            if group=='combined':known[group]['controller_equal']=evaluate_group(data,methods,False,5000)
        write(HERE/'KNOWN_NEW_STATE_RESULTS.json',dict(utc=utc(),evidence='DEVELOPMENT_KNOWN_TYPES_NEW_STATES',reports=known,scope='42 instances:5 optimization types, prioritized search and conservative reinforcement learning; original9 learning types not evaluated',unavailable='No-schedule, top2–4 and nominal-only deliberately omitted by revised prompt'))
        seal('KNOWN_NEW_STATE_RESULTS.receipt.json',[HERE/'KNOWN_NEW_STATE_RESULTS.json',HERE/'confirmatory_statistics.py',HERE/'statistics_pipeline.py',HERE/'NEW_EVAL_BUNDLE.receipt.json'])
    print('Endpoints1–7 sealed; posthoc diagnostics may now begin',flush=True)

if __name__=='__main__':main()
