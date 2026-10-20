"""Fixed-architecture source-diversity curves with exact-source fit caching."""
from common import *
from train_critic import load_data,key_for,folder_for,fit
from single_source_selection import controller_key
from frozen_networks import stable_int
import itertools,fcntl
import jax


def plan():
    for path,digest in read(HERE/'ARCHITECTURE_FIXED.receipt.json')['files'].items():assert sha(path)==digest
    configuration=read(HERE/'CRITIC_TRAINING_PLAN.json');data=load_data(configuration['bundle_path']);types=sorted(set(data['types']));draws=[];sources=set()
    for held in types:
        available=[t for t in types if t!=held]
        for count in (1,2,4,7,10,13):
            if count>len(available):continue
            options=list(itertools.combinations(available,count));rng=np.random.default_rng(stable_int('crosstype2-scaling',202610182101,held,count))
            selected=rng.choice(len(options),size=min(10,len(options)),replace=False) if count<len(available) else np.asarray([0])
            for draw,index in enumerate(selected):
                source=options[int(index)];draws.append(dict(held_type=str(held),source_types=list(source),m=count,draw=draw))
                sources.add(source)
                if count>=2:sources.update(tuple(t for t in source if t!=inner) for inner in source)
    tasks=[]
    for source in sorted(sources,key=lambda s:(len(s),s)):
        for mode in ('full','no_controller'):
            for init in ('scratch','warm'):
                task=dict(source_types=list(source),source_key=key_for(source),mode=mode,initialization=init,task_uid=hashlib.sha256(json.dumps([source,mode,init]).encode()).hexdigest())
                if not (folder_for(task)/'COMPLETE.receipt.json').exists():tasks.append(task)
    # m=1 checkpoint/init/lambda: source-controller inner fits, separately keyed.
    for kind in types:
        source=[str(kind)]
        for name in data['names'][data['types']==kind]:
            for mode in ('full','no_controller'):
                for init in ('scratch','warm'):
                    task=dict(source_types=source,held_controller=str(name),source_key=controller_key(source,str(name)),mode=mode,initialization=init,task_uid=hashlib.sha256(json.dumps([source,mode,init,'controller',str(name)]).encode()).hexdigest())
                    if not (folder_for(task)/'COMPLETE.receipt.json').exists():tasks.append(task)
    write(HERE/'SCALING_DRAWS.json',dict(utc=utc(),root_seed=202610182101,draws=draws,distinct_source_sets=len(sources),selection_outcome_blind=True,m1_inner='Source controller leave-one; explicitly authorized by user'))
    record={**configuration,'tasks':tasks,'utc':utc(),'plan_receipt':'SCALING_TRAINING_PLAN.receipt.json','fixed_architecture_receipt_sha256':sha(HERE/'ARCHITECTURE_FIXED.receipt.json'),'scaling_only':True}
    write(HERE/'SCALING_TRAINING_PLAN.json',record)
    seal('SCALING_TRAINING_PLAN.receipt.json',[HERE/'SCALING_TRAINING_PLAN.json',HERE/'SCALING_DRAWS.json',HERE/'scaling.py',HERE/'single_source_selection.py',HERE/'select_mixed.py',HERE/'USER_DESIGN_DECISIONS.receipt.json',HERE/'ARCHITECTURE_FIXED.receipt.json'],before_scaling_fitting=True)
    print('Scaling:',len(draws),'held-type source subsets;',len(tasks),'additional cached three-seed branch fits')


def run(worker):
    assert jax.default_backend()=='gpu';configuration=read(HERE/'SCALING_TRAINING_PLAN.json');data=load_data(configuration['bundle_path']);claims=HERE/'scaling_claims';claims.mkdir(exist_ok=True)
    for p,digest in read(HERE/'SCALING_TRAINING_PLAN.receipt.json')['files'].items():assert sha(p)==digest
    for task in configuration['tasks']:
        if (folder_for(task)/'COMPLETE.receipt.json').exists():continue
        claim=claims/f"{task['task_uid']}.json"
        try:fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        except FileExistsError:continue
        with os.fdopen(fd,'w') as f:f.write(json.dumps(dict(utc=utc(),worker=worker,task=task)))
        fit(data,task,configuration)
    write(HERE/f'SCALING_WORKER_{worker}_COMPLETE.json',dict(utc=utc(),worker=worker))


def evaluate():
    from select_mixed import select
    from critic_model import choices
    from baselines import repair_class,source_fixed
    from statistics_pipeline import best
    configuration=read(HERE/'CRITIC_TRAINING_PLAN.json');data=load_data(configuration['bundle_path']);lower=(data['labels'][...,0]==1).astype(float);rows=[]
    type_classes={str(t):int(repair_class(best(lower[data['types']==t].mean((0,1))))) for t in sorted(set(data['types']))}
    cache={}
    for row in read(HERE/'SCALING_DRAWS.json')['draws']:
        source=row['source_types'];key=key_for(source)
        if key not in cache:
            p,model,selection=select(source,configuration,data);cache[key]=(choices(p)['mean'],model,selection)
        chosen,model,selection=cache[key];mask=data['types']==row['held_type'];selected=np.take_along_axis(lower[mask],chosen[mask,...,None],axis=-1)[...,0]
        oracle=lower[mask].max(-1);fixed,_,_=source_fixed(data,source)
        rows.append(dict(**row,source_best_repair_classes=sorted(set(type_classes[t] for t in source)),repair_class_count=len(set(type_classes[t] for t in source)),regret_pp=float((oracle-selected).mean()*100),success_pct=float(selected.mean()*100),gain_vs_progress_pp=float((selected-lower[mask,:,12]).mean()*100),gain_vs_source_fixed_pp=float((selected-lower[mask,:,fixed]).mean()*100),source_key=key,controller_weight=float(model.get('full_weight',1.))))
    write(HERE/'SCALING_RESULTS.json',dict(utc=utc(),evidence='DEVELOPMENT',rows=rows,source_type_best_repair_classes=type_classes,single_source_user_authorized_controller_inner=True,training_oracle='Common seed0 in-sample regret',selection_outer_labels_used=False))
    import csv
    with (HERE/'SCALING_RESULTS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    seal('SCALING_RESULTS.receipt.json',[HERE/'SCALING_RESULTS.json',HERE/'SCALING_RESULTS.csv',HERE/'SCALING_TRAINING_PLAN.receipt.json',HERE/'scaling.py'])
    print('Both scaling-axis data tables complete:',len(rows),'outer subsets')


if __name__=='__main__':
    with jax.default_matmul_precision('highest'):
        if sys.argv[1]=='run':run(int(sys.argv[2]))
        else:globals()[sys.argv[1]]()
