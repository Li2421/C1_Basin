"""Read saved evidence only: eta distributions, switches, harms and transitions."""
from common import *
import csv
def save_csv(name,rows):
    with (HERE/name).open('w') as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def main():
    assert (HERE/'CONFIRMATORY_RESULTS.receipt.json').exists()
    with np.load(HERE/'NEW_EVAL_BUNDLE.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(HERE/'NEW_EVAL_ANALYSIS_CHOICES.npz',allow_pickle=False) as z:choices={k:z[k] for k in z.files if k not in ('names','types')}
    records=read(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json')['states'];types=data['types'];names=data['names'];transitions=[];switches=[];harms=[]
    def status(i,s,a,q):
        if data['labels'][i,s,a,q]<0:return 'unknown'
        if data['labels'][i,s,a,q]==1:return 'success'
        if data['collision'][i,s,a,q]:return 'collision'
        if data['timeout'][i,s,a,q]:return 'timeout'
        return 'other_known_non_success'
    for i,name in enumerate(names):
        pl=[];pu=[];bl=[];bu=[]
        for s in range(24):
            p=int(choices['pipeline'][i,s]);b=int(choices['all_safety_baseline'][i,s]);n=int(choices['nominal_only'][i,s]);record=next(r for r in records if r['controller']==name and r['state_index']==s);a=record['methods']['pipeline'];non_diag=not np.array_equal(a['eta_a'],a['eta_b'])
            for q in (0,1):
                transitions.append(dict(controller=str(name),type=str(types[i]),state=s,seed=q,nominal_status=status(i,s,n,q),safety_status=status(i,s,b,q),nominal_uid=str(data['uids'][i,s,n,q]),safety_uid=str(data['uids'][i,s,b,q])))
                rid=str(data['uids'][i,s,p,q]);path=HERE/'new_evaluation/raw'/f'{rid}.json';raw=read(path) if path.exists() else None;physical_steps=raw['physical_steps'] if raw else None
                switches.append(dict(controller=str(name),type=str(types[i]),state=s,seed=q,non_diagonal=non_diag,chosen_switch_step=a['switch_step'] if non_diag else None,physical_steps=physical_steps,late_eta_physically_executed=bool(physical_steps>a['switch_step']) if non_diag and physical_steps is not None else False if not non_diag else None,status=status(i,s,p,q),rollout_uid=rid,trace_evidence_available=raw is not None))
                pl.append(data['labels'][i,s,p,q]==1);pu.append(data['labels'][i,s,p,q]!=0);bl.append(data['labels'][i,s,b,q]==1);bu.append(data['labels'][i,s,b,q]!=0)
        pl=np.asarray(pl,float);pu=np.asarray(pu,float);bl=np.asarray(bl,float);bu=np.asarray(bu,float);same=(data['uids'][i,np.arange(24),choices['pipeline'][i]]==data['uids'][i,np.arange(24),choices['all_safety_baseline'][i]]).reshape(-1);lower=np.where(same,0,pl-bu).mean()*100;upper=np.where(same,0,pu-bl).mean()*100
        harms.append(dict(controller=str(name),type=str(types[i]),pipeline_success_lower_pct=float(pl.mean()*100),safety_success_lower_pct=float(bl.mean()*100),gain_pp=float((pl-bl).mean()*100),gain_unknown_lower_pp=float(lower),gain_unknown_upper_pp=float(upper),harm_possible=lower<=-10+1e-12,harm_certain=upper<=-10+1e-12))
    descriptions=read(HERE/'SELECTED_ETA_DESCRIPTIONS.json')['rows'];eta_summary={}
    for kind in sorted(set(types)):
        rr=[r for r in descriptions if r['type']==kind];early=np.asarray([r['eta_early'] for r in rr]);late=np.asarray([r['eta_late'] for r in rr]);cos=[r['direction_cosine'] for r in rr if r['direction_cosine'] is not None]
        eta_summary[str(kind)]=dict(controller_states=len(rr),eta_early_quantiles=np.quantile(early,[.05,.25,.5,.75,.95],axis=0),eta_late_quantiles=np.quantile(late,[.05,.25,.5,.75,.95],axis=0),direction_cosine_quantiles=np.quantile(cos,[.05,.25,.5,.75,.95]) if cos else None,undefined_diagonal_directions=sum(r['diagonal'] for r in rr),distances={k:np.quantile([r[k] for r in rr],[.05,.25,.5,.75,.95]) for k in ('distance_early_eta13','distance_early_eta10','distance_late_eta13','distance_late_eta10')})
    transition_summary={}
    for kind in sorted(set(types)):
        rr=[r for r in transitions if r['type']==kind];matrix={};collision=[r for r in rr if r['nominal_status']=='collision']
        for r in rr:
            key=r['nominal_status']+'→'+r['safety_status'];matrix[key]=matrix.get(key,0)+1
        transition_summary[str(kind)]=dict(all_transition_counts=matrix,total=len(rr),known_nominal_collision_count=len(collision),after_nominal_collision_fractions={k:sum(r['safety_status']==k for r in collision)/len(collision) if collision else None for k in ('success','timeout','unknown','collision')})
    switch_summary={}
    for kind in sorted(set(types)):
        rr=[r for r in switches if r['type']==kind];supported=[r for r in rr if r['trace_evidence_available']];non_diag=[r for r in supported if r['non_diagonal']]
        switch_summary[str(kind)]=dict(total_seed_cases=len(rr),trace_supported=len(supported),non_diagonal_seed_cases=len(non_diag),late_eta_executed=sum(r['late_eta_physically_executed'] for r in non_diag),late_eta_executed_fraction_among_non_diagonal=float(np.mean([r['late_eta_physically_executed'] for r in non_diag])) if non_diag else None,limitation='Chosen switch time is distinct from a late eta actually executed before termination')
    save_csv('NOMINAL_TO_SAFETY_TRANSITIONS.csv',transitions);save_csv('PIPELINE_SWITCH_EXECUTION.csv',switches);save_csv('CONTROLLER_HARM_DETAILS.csv',harms);write(HERE/'NOMINAL_TO_SAFETY_TRANSITIONS.json',dict(utc=utc(),types=transition_summary));write(HERE/'SELECTED_ETA_SUMMARY.json',dict(utc=utc(),quantile_levels=[.05,.25,.5,.75,.95],types=eta_summary));write(HERE/'PIPELINE_SWITCH_EXECUTION.json',dict(utc=utc(),types=switch_summary));write(HERE/'CONTROLLER_HARM_DETAILS.json',dict(utc=utc(),controllers=harms))
    seal('DESCRIPTIVE_ANALYSIS.receipt.json',[HERE/n for n in ('NOMINAL_TO_SAFETY_TRANSITIONS.csv','NOMINAL_TO_SAFETY_TRANSITIONS.json','PIPELINE_SWITCH_EXECUTION.csv','PIPELINE_SWITCH_EXECUTION.json','CONTROLLER_HARM_DETAILS.csv','CONTROLLER_HARM_DETAILS.json','SELECTED_ETA_SUMMARY.json','descriptive_analysis.py')],saved_evidence_read_only=True)
    print('Descriptive eta/switch/harm/transition evidence complete',flush=True)
if __name__=='__main__':main()
