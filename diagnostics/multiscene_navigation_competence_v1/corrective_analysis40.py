"""Paired controller/state cluster inference after immutable choice receipts."""
from common import *
from design_generated import ARMS
from selector_data40 import MENU
import csv

def choices_for(family,names):
    methods={arm:np.eye(9,dtype=np.float32)[[ARMS.index(arm)]*len(names)] for arm in ARMS}
    for tag,label in [('leave_one_family','zero_shot_selector'),('final_locked','zero_shot_selector')]:
        p=HERE/'selector_choices40'/f'{family}_{tag}.json'
        if not p.exists() or read(p).get('status')=='GAP':continue
        receipt=p.with_suffix('.receipt.json');assert receipt.exists() and all(sha(path)==h for path,h in read(receipt)['files'].items())
        value=read(p);assert value['controller_names']==names and not value['target_label_file_opened']
        methods[label]=np.eye(9,dtype=np.float32)[[ARMS.index(a) for a in value['chosen_arms']]]
        methods['source_average_best']=np.eye(9,dtype=np.float32)[[ARMS.index(value['source_mean']['chosen_arm'])]*len(names)]
    p=HERE/'selector_choices40'/f'{family}_fewshot.json'
    if p.exists():
        receipt=p.with_suffix('.receipt.json');assert all(sha(path)==h for path,h in read(receipt)['files'].items());value=read(p)
        methods['four_pilot_schedule_right_hand']=np.eye(9,dtype=np.float32)[[ARMS.index(value['choices'][n]['simplified_four_pilot_choice']) for n in names]]
        for k in [1,2,4,8,16]:
            entries=[value['choices'][n]['curves'][str(k)] for n in names]
            if any(e['status']=='GAP' for e in entries):continue
            for label,key in [('empirical','empirical_choice_counts'),('bayesian','bayesian_choice_counts')]:
                w=np.zeros((len(names),9),np.float32);w[:,2:]=[np.asarray(e[key])/e['repeats'] for e in entries];methods[f'fewshot_{label}_{k}']=w
    return methods

def cluster_bootstrap(Y,methods,seed,repeats=5000):
    # Seeds stay paired within their original state; controller first, then state clusters.
    y=Y.mean(-1);C,S,A=y.shape;rng=np.random.default_rng(seed);ci=rng.integers(C,size=(repeats,C));si=rng.integers(S,size=(repeats,C,S));b=y[ci[:,:,None],si];out={}
    for name,w in methods.items():out[name]=(b*w[ci][:,:,None,:]).sum(-1).mean((1,2))
    out['complete_menu_oracle_in_sample']=b.mean(2)[:,:,2:].max(-1).mean(1)
    out['best_fixed_oracle_in_sample']=b.mean(2)[:,:,2:5].max(-1).mean(1)
    out['best_schedule_oracle_in_sample']=b.mean(2)[:,:,5:8].max(-1).mean(1)
    return out

def crossfit_oracle(Y,arm_indices):
    y=Y.mean(-1);rate=np.zeros((len(y),y.shape[1]),float)
    for parity in [0,1]:
        source=np.arange(y.shape[1])%2==parity;target=~source;choice=np.asarray(arm_indices)[y[:,source][:,:,arm_indices].mean(1).argmax(-1)]
        for c,a in enumerate(choice):rate[c,target]=y[c,target,a]
    return rate

def analyze_family(family):
    cfg=read(HERE/'SELECTOR40_CONFIG_FROZEN.json');p=HERE/'selector_labels40'/f'{family}.npz';z=np.load(p);Y=z['success'];U=z['unknown'];C=z['collision'];T=z['timeout'];names=z['controller_names'].tolist();methods=choices_for(family,names);seed=cfg['bootstrap_seed']+'ABCDEFG'.index(family)
    boot=cluster_bootstrap(Y,methods,seed,cfg['bootstrap']);oracle=Y[:,:,2:].mean((1,3)).max(-1);fixed=Y[:,:,2:5].mean((1,3)).max(-1);oracle_hi=(Y+U)[:,:,2:].mean((1,3)).max(-1);results={}
    for name,w in methods.items():
        selected=(Y*w[:,None,:,None]).sum(2);selected_u=(U*w[:,None,:,None]).sum(2);rates=selected.mean((1,2));upper=(selected+selected_u).mean((1,2));regret=oracle-rates
        results[name]=dict(success_pct=float(rates.mean()*100),success95_pct=np.percentile(boot[name],[2.5,97.5])*100,success_identified_pct=[float(rates.mean()*100),float(upper.mean()*100)],collision_pct=float((C*w[:,None,:,None]).sum(2).mean()*100),timeout_pct=float((T*w[:,None,:,None]).sum(2).mean()*100),unknown_pct=float(selected_u.mean()*100),mean_regret_pp=float(regret.mean()*100),max_regret_pp=float(regret.max()*100),regret_identified_pp=[float((oracle-upper).mean()*100),float((oracle_hi-rates).mean()*100)],regret95_pp=np.percentile((boot['complete_menu_oracle_in_sample']-boot[name])*100,[2.5,97.5]),harm_controllers=int(np.sum(rates<=Y[:,:,1].mean((1,2))-.1+1e-12)),catastrophe_controllers=int(np.sum(rates<=Y[:,:,2].mean((1,2))-.1+1e-12)),controller_success_pct=rates*100,evidence='CONFIRMATORY conditional on immutable source-only choices and disclosed coverage')
    for name,idx in [('complete_menu_oracle',list(range(2,9))),('best_fixed_oracle',list(range(2,5)))]:
        cf=crossfit_oracle(Y,idx);insamp=oracle if name=='complete_menu_oracle' else fixed
        results[name]=dict(success_pct=float(insamp.mean()*100),success95_pct=np.percentile(boot[name+'_in_sample'],[2.5,97.5])*100,crossfit_success_pct=float(cf.mean()*100),oracle_estimation='Both in-sample upper and state-parity crossfit lower; neither used for prospective choices',evidence='DEVELOPMENT oracle')
    pairs={}
    for name,left,right in [('selector_vs_safety','zero_shot_selector','safety_baseline'),('selector_regret_minus_source_average','source_average_best','zero_shot_selector'),('schedule_vs_fixed_ten','S13','F10'),('right_hand_vs_schedule','right_hand','S13')]:
        if left in boot and right in boot:pairs[name]=dict(difference_pp=results[left]['success_pct']-results[right]['success_pct'],paired95_pp=np.percentile((boot[left]-boot[right])*100,[2.5,97.5]))
    nominal_coll=C[:,:,0]>0;transfers={k:int(np.sum(nominal_coll*value[:,:,1])) for k,value in [('success',Y),('timeout',T),('unknown',U),('collision',C)]};transfers['nominal_collision_trials']=int(nominal_coll.sum());transfers['rates_given_nominal_collision']={k:v/transfers['nominal_collision_trials'] if transfers['nominal_collision_trials'] else None for k,v in transfers.items() if k!='nominal_collision_trials'}
    schedule_ix=[5,6,7];fixed_ix=[2,3,4];rates=Y.mean((1,3));schedule=rates[:,schedule_ix].max(-1);best_fixed=rates[:,fixed_ix].max(-1);right=rates[:,8];safety=rates[:,1];best=rates[:,2:].max(-1)
    winning=[ARMS[2+int(i)] for i in rates[:,2:].argmax(-1)];preference='持续右手规则' if right.mean()>schedule.mean() else '时间表' if schedule.mean()>right.mean() else '时间表与持续右手规则持平'
    descriptive=dict(best_correction_minus_safety_pp=float((best-safety).mean()*100),best_correction_minus_safety95_pp=np.percentile((boot['complete_menu_oracle_in_sample']-boot['safety_baseline'])*100,[2.5,97.5]),winning_arm_by_controller=dict(zip(names,winning)),preferred_arm_family=preference,schedule_minus_best_fixed_pp=float((schedule-best_fixed).mean()*100),schedule_minus_best_fixed95_pp=np.percentile((boot['best_schedule_oracle_in_sample']-boot['best_fixed_oracle_in_sample'])*100,[2.5,97.5]),right_hand_minus_best_schedule_pp=float((right-schedule).mean()*100),right_hand_minus_best_schedule95_pp=np.percentile((boot['right_hand']-boot['best_schedule_oracle_in_sample'])*100,[2.5,97.5]),best_arm_inference='Posthoc descriptive oracle; selection bias bracketed by separate crossfit oracle',nominal_to_safety=transfers,timeout_not_deadlock=True,success_uplift_defines_deadlock_repair=True)
    fewshot={};fp=HERE/'selector_choices40'/f'{family}_fewshot.json'
    if fp.exists():
        frozen=read(fp);armrates=Y[:,:,2:].mean((1,3));armupper=(Y+U)[:,:,2:].mean((1,3))
        for k in cfg['ks']:
            for label,key in [('empirical','empirical_choices'),('bayesian','bayesian_choices')]:
                values=[]
                for c,name in enumerate(names):
                    entry=frozen['choices'][name]['curves'][str(k)]
                    if entry['status']=='GAP':continue
                    selected=np.asarray(entry[key],int);regret=oracle[c]-armrates[c,selected]
                    values.append(dict(controller=name,mean_regret_pp=float(regret.mean()*100),p90_regret_pp=float(np.quantile(regret,.9)*100),probability_regret_at_most5pp=float(np.mean(regret<=.05+1e-10)),identified_regret_pp=[float((oracle[c]-armupper[c,selected]).mean()*100),float((oracle_hi[c]-armrates[c,selected]).mean()*100)]))
                fewshot[f'{label}_{k}']=dict(known_labels_per_arm=k,total_known_label_budget=7*k,controllers=values,all_controllers_available=len(values)==len(names),mean_regret_pp=float(np.mean([v['mean_regret_pp'] for v in values])) if values else None,mean_p90_regret_pp=float(np.mean([v['p90_regret_pp'] for v in values])) if values else None)
    out=dict(utc=utc(),family=family,controllers=names,methods=results,paired_comparisons=pairs,descriptive=descriptive,fewshot_curves=fewshot,unknown_trials=int(U.sum()),bootstrap=dict(repeats=cfg['bootstrap'],seed=seed,first_layer='controller',second_layer='state with both seeds paired'),target_labels_opened_only_after_choice_receipts=True)
    write(HERE/f'CORRECTIVE_ANALYSIS40_{family}.json',out);return out

def analyze_scaling(families):
    plan=read(HERE/'SELECTOR40_MODEL_PLAN_FROZEN.json');rows=[]
    for entry in plan['predictions']:
        if not entry['tag'].startswith('m'):continue
        f=entry['target'];z=np.load(HERE/'selector_labels40'/f'{f}.npz');Y=z['success'];oracle=Y[:,:,2:].mean((1,3)).max(-1);p=read(HERE/'selector_choices40'/f"{f}_{entry['tag']}.json");choice=[ARMS.index(a) for a in p['chosen_arms']];rate=np.asarray([Y[c,:,a,:].mean() for c,a in enumerate(choice)]);m=int(entry['tag'].split('_')[0][1:]);rows.append(dict(family=f,m=m,combination=entry['tag'],sources=entry['sources'],regret_pp=float((oracle-rate).mean()*100),controller_regret_pp=(oracle-rate)*100))
    cfg=read(HERE/'SELECTOR40_CONFIG_FROZEN.json');rng=np.random.default_rng(cfg['bootstrap_seed']+100);available=sorted({r['family'] for r in rows});slopes=[]
    for _ in range(cfg['bootstrap']):
        x=[];y=[]
        for f in rng.choice(available,len(available),replace=True) if available else []:
            for m in [0,1,3,5]:
                group=[r for r in rows if r['family']==f and r['m']==m]
                if group:
                    for i in rng.integers(len(group),size=len(group)):x.append(m);y.append(group[int(i)]['regret_pp'])
        if len(set(x))>=2:slopes.append(float(np.polyfit(x,y,1)[0]))
    expected={(f,m,i) for f in families for m in [0,1,3,5] for i in range(3 if m in [1,3] else 1)};observed={(r['family'],r['m'],int(r['combination'].rsplit('combination',1)[1])) for r in rows};missing=sorted(expected-observed)
    x=[r['m'] for r in rows];y=[r['regret_pp'] for r in rows];point=float(np.polyfit(x,y,1)[0]) if len(set(x))>=2 else None;ci=np.percentile(slopes,[2.5,97.5]).tolist() if slopes else None
    result=dict(rows=rows,mean_regret_by_training_family_count={str(m):float(np.mean([r['regret_pp'] for r in rows if r['m']==m])) if any(r['m']==m for r in rows) else None for m in [0,1,3,5]},slope_pp_per_family=point,slope95=ci,missing_fixed_combinations=missing,complete_preregistered_curve=not missing,endpoint3_passed=bool(not missing and ci is not None and ci[1]<0),partial_curve_only=bool(missing),bootstrap='target-family clusters then fixed-combination clusters,5000 paired repetitions',blocked_source_combinations_not_replaced=True)
    write(HERE/'SCALE_CURVE40_RESULTS.json',result)
    p=HERE/'SCALE_CURVE40.csv'
    with p.open('w') as file:
        w=csv.DictWriter(file,fieldnames=['family','m','combination','sources','regret_pp']);w.writeheader();w.writerows([{k:r[k] for k in w.fieldnames} for r in rows])
    return result

def analyze_AF():
    receipt=read(HERE/'ALL_AF_CHOICES40_FROZEN.receipt.json');assert all(sha(p)==h for p,h in receipt['files'].items())
    families=read(HERE/'AF_MULTIAGENT_CAPABILITY_GATES.json')['eligible_families'];results={f:analyze_family(f) for f in families};cfg=read(HERE/'SELECTOR40_CONFIG_FROZEN.json');Y=[];weights={name:[] for name in ['zero_shot_selector','source_average_best']}
    for f in families:
        z=np.load(HERE/'selector_labels40'/f'{f}.npz');methods=choices_for(f,z['controller_names'].tolist())
        if not all(name in methods for name in weights):continue
        Y.append(z['success'])
        for name in weights:weights[name].append(methods[name])
    if Y:
        boots=cluster_bootstrap(np.concatenate(Y),{k:np.concatenate(v) for k,v in weights.items()},cfg['bootstrap_seed']+200,cfg['bootstrap']);ci=np.percentile((boots['source_average_best']-boots['zero_shot_selector'])*100,[2.5,97.5]);delta=float(np.mean([r['methods']['zero_shot_selector']['mean_regret_pp']-r['methods']['source_average_best']['mean_regret_pp'] for r in results.values() if 'zero_shot_selector' in r['methods']]))
        ypool=np.concatenate(Y);wpool={k:np.concatenate(v) for k,v in weights.items()};delta=float(((ypool*wpool['source_average_best'][:,None,:,None]).sum(2).mean()-(ypool*wpool['zero_shot_selector'][:,None,:,None]).sum(2).mean())*100)
        endpoint1=dict(passed=bool(ci[1]<0 and len(Y)==6),available_family_paired_difference_pp=delta,paired95_pp=ci,complete_six_family_coverage=len(Y)==6,pooling='Eligible controllers equally weighted; state/seed means paired')
    else:endpoint1=dict(passed=False,status='GAP',reason='No target had both a frozen source-only selector and source average comparator')
    practical=[f for f,r in results.items() if r['methods'].get('zero_shot_selector',{}).get('mean_regret_pp',float('inf'))<=5+1e-10]
    scale=analyze_scaling(families);out=dict(utc=utc(),families=results,endpoint1=endpoint1,endpoint2=dict(passed=len(practical)>=4,qualifying_families=practical,count=len(practical)),endpoint3=dict(passed=scale['endpoint3_passed'],slope=scale['slope_pp_per_family'],slope95=scale['slope95'],complete_curve=scale['complete_preregistered_curve']),endpoint4=dict(status='LOCKED_NOT_OPENED'),G_labels_read=0)
    p=HERE/'AF_ANALYSIS40_RESULTS.json';write(p,out)
    seal('AF_ANALYSIS_FROZEN.receipt.json',[p,HERE/'SCALE_CURVE40_RESULTS.json',HERE/'SCALE_CURVE40.csv',HERE/'ALL_AF_CHOICES40_FROZEN.receipt.json',HERE/'corrective_analysis40.py']+[HERE/f'CORRECTIVE_ANALYSIS40_{f}.json' for f in families],before_G_corrective_labels=True)
    return out

def analyze_G():
    assert (HERE/'AF_ANALYSIS_FROZEN.receipt.json').exists() and (HERE/'FINAL_MODEL_FROZEN.receipt.json').exists()
    receipt=read(HERE/'selector_choices40/G_final_locked.receipt.json');assert all(sha(p)==h for p,h in receipt['files'].items())
    result=analyze_family('G');selector=result['methods']['zero_shot_selector'];uplift=result['paired_comparisons']['selector_vs_safety'];endpoint=dict(passed=selector['mean_regret_pp']<=5+1e-10 and uplift['paired95_pp'][0]>0,regret_pp=selector['mean_regret_pp'],regret_identified_pp=selector['regret_identified_pp'],selected_minus_safety_pp=uplift['difference_pp'],selected_minus_safety95_pp=uplift['paired95_pp'],evaluation_once=True,G_source_rows=0,AF_and_finalmodel_frozen_before_G_corrective_labels=True)
    p=HERE/'LOCKED_G40_RESULTS.json';write(p,dict(utc=utc(),endpoint4=endpoint,analysis=result))
    seal('LOCKED_G40_COMPLETE.receipt.json',[p,HERE/'CORRECTIVE_ANALYSIS40_G.json',HERE/'evaluation40_G_POSTFLIGHT.json',HERE/'FINAL_MODEL_FROZEN.receipt.json',HERE/'AF_ANALYSIS_FROZEN.receipt.json',HERE/'selector_choices40/G_final_locked.receipt.json'],evaluated_once=True,no_G_based_adjustment=True)
    return endpoint

if __name__=='__main__':analyze_AF() if sys.argv[1]=='AF' else analyze_G() if sys.argv[1]=='G' else analyze_family(sys.argv[1])
