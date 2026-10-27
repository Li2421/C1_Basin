"""5000 paired hierarchical bootstrap draws; frozen models, reselected local controls."""
from common import *
import numpy as np

B=5000
METHODS=('ridge','knn','global_fixed','eta13','safety','crossfit','calibration','in_sample')


def interval(values):
    a=np.asarray(values,float)
    return [float(v)for v in np.percentile(a,[2.5,97.5])]


def c_decision(estimate,ci,rule='mean_difference_le_0.05'):
    assert rule in('mean_difference_le_0.05','upper95_difference_le_0.05')
    mean_passed=estimate<=.05;upper_passed=ci[1]<=.05
    passed=mean_passed if rule=='mean_difference_le_0.05'else upper_passed
    return dict(verdict='PASSED'if passed else 'NOT_PASSED',ci=ci,estimate=float(estimate),
        decision_rule=rule,mean_difference_within5pp=bool(mean_passed),
        stricter_upper95_within5pp=bool(upper_passed),
        interpretation='Regret(ridge)-regret(calibration)=calibration-ridge; original mean predicate and95%CI-bound predicate reported separately.')


def sampled_rates(table,choice,rng,replicates=B):
    y=np.asarray(table['y'],float);u=np.asarray(table['unknown'],float);C,S,A=y.shape
    out={k:np.empty(replicates)for k in METHODS};upper={k:np.empty(replicates)for k in METHODS}
    draws=choice['methods']['calibration']['draws'];menus=np.asarray([d['actions']for d in draws]);cal=np.asarray([d['states']for d in draws])
    halves=[np.arange(S)[np.arange(S)%2==i]for i in (0,1)]
    for begin in range(0,replicates,20):
        n=min(20,replicates-begin);ct=rng.integers(C,size=(n,C));st=np.empty((n,C,S),int)
        for half in halves:st[:,:,half]=rng.choice(half,size=(n,C,len(half)),replace=True)
        sample=y[ct[:,:,None],st];up=np.minimum(1.,sample+u[ct[:,:,None],st]);q=sample.mean(1);qu=up.mean(1)
        for key in METHODS[:5]:
            a=choice[key];out[key][begin:begin+n]=q[:,:,a].mean(1);upper[key][begin:begin+n]=qu[:,:,a].mean(1)
        q0=q[:,halves[0],:].mean(1);q1=q[:,halves[1],:].mean(1)
        a0=np.argmax(q0,axis=1);a1=np.argmax(q1,axis=1);rr=np.arange(n)
        out['crossfit'][begin:begin+n]=(q1[rr,a0]+q0[rr,a1])/2
        upper['crossfit'][begin:begin+n]=(qu[:,halves[1],:].mean(1)[rr,a0]+qu[:,halves[0],:].mean(1)[rr,a1])/2
        # Same prespecified 200 independent calibration trials, selecting on4 states,
        # scoring on their complement in every bootstrap draw.
        train=q[:,cal,:].mean(2)
        scores=np.take_along_axis(train,menus[None],axis=2)
        chosen=menus[np.arange(len(menus))[None],np.argmax(scores,axis=2)]
        overall=np.take_along_axis(q.sum(1),chosen,axis=1)
        selected_cal=np.take_along_axis(train,chosen[:,:,None],axis=2)[:,:,0]*4
        out['calibration'][begin:begin+n]=((overall-selected_cal)/(S-4)).mean(1)
        trainu=qu[:,cal,:].mean(2);overallu=np.take_along_axis(qu.sum(1),chosen,axis=1)
        upper['calibration'][begin:begin+n]=((overallu-np.take_along_axis(trainu,chosen[:,:,None],axis=2)[:,:,0]*4)/(S-4)).mean(1)
        best=np.argmax(q.mean(1),axis=1)
        out['in_sample'][begin:begin+n]=q.mean(1)[rr,best];upper['in_sample'][begin:begin+n]=qu.mean(1)[rr,best]
    return out,upper


def aggregate_boot(scene_samples,names,rng):
    sample=rng.integers(len(names),size=(B,len(names)))
    inner=rng.integers(B,size=(B,len(names)))
    return {key:np.stack([scene_samples[n][key]for n in names],axis=1)[inner,sample].mean(1)for key in METHODS}


def main():
    freeze=read(HERE/'MODEL_FROZEN.receipt.json')
    for p,h in freeze['files'].items():assert sha(p)==h,p
    tables=read(HERE/'LABEL_TABLES.json');choices=read(HERE/'LOSO_CHOICES.json');lofo=read(HERE/'LOFO_CHOICES.json')
    rng=np.random.default_rng(2026102413);boot={};upper={};per_scene={}
    for name,choice in choices.items():
        a,b=sampled_rates(tables[name],choice,rng);boot[name]=a;upper[name]=b
        per_scene[name]={key:dict(rate=choice['methods'][key]['rate'],upper_rate=choice['methods'][key]['upper'],
            ci=interval(a[key]),unknown_upper_ci=interval(b[key]))for key in METHODS}
        print('Bootstrap',name,flush=True)
    names=list(choices);macro=aggregate_boot(boot,names,rng)
    A=macro['ridge']-macro['global_fixed']
    C=macro['calibration']-macro['ridge']
    family_results={}
    for family,rows in lofo.items():
        family_samples={}
        for name,row in rows.items():
            # Reuse the same paired controller/state draw generator with fixed
            # LOFO prospective choices; local controls are irrelevant to test乙.
            effective=dict(choices[name]);effective.update({k:row[k]for k in METHODS[:5]})
            a,_=sampled_rates(tables[name],effective,rng);family_samples[name]=a
        sample=aggregate_boot(family_samples,list(rows),rng)
        family_results[family]=dict(scene_names=list(rows),ridge_minus_global_ci=interval(sample['ridge']-sample['global_fixed']),
            ridge_minus_global=float(np.mean([r['methods']['ridge']['rate']-r['methods']['global_fixed']['rate']for r in rows.values()])))
    aci=interval(A);bci=family_results['bottleneck']['ridge_minus_global_ci'];cci=interval(C)
    cfile=HERE/'C_CRITERION_CLARIFICATION.json'
    crule=read(cfile)['decision_rule']if cfile.exists()else 'mean_difference_le_0.05'
    cestimate=float(np.mean([c['methods']['calibration']['rate']-c['methods']['ridge']['rate']for c in choices.values()]))
    decisions=dict(A=dict(verdict='PASSED'if aci[0]>0 else 'NOT_PASSED',ci=aci,
        estimate=float(np.mean([c['methods']['ridge']['rate']-c['methods']['global_fixed']['rate']for c in choices.values()]))),
        B=dict(verdict='PASSED'if bci[0]>0 else 'NOT_PASSED',ci=bci,estimate=family_results['bottleneck']['ridge_minus_global']),
        C=c_decision(cestimate,cci,crule),
        D=dict(verdict='DESCRIPTIVE',source='EXTRAPOLATION_RESULTS.json'))
    write(HERE/'STATISTICS.json',dict(decisions=decisions,per_scene=per_scene,lofo=family_results,
        macro={key:dict(rate=float(np.mean([c['methods'][key]['rate']for c in choices.values()])),ci=interval(macro[key]))for key in METHODS},
        replicates=B,method='Paired scene -> controller type -> initial state; fixed half membership within state bootstrap; frozen prospective source fits; local controls reselected',
        limitation='Intervals condition on frozen source-fit decisions; do not represent new independent training-scene samples/refitting the complete nested learner. Unknown lower bound, fixed-policy upper intervals separately.'))
    with (HERE/'METHOD_COMPARISON.md').open('w')as f:
        f.write('| 留出场景 | 主岭回归 | 次kNN | 全局固定 | η13 | 安全 | 低成本校准 | 交叉拟合最优 | 样本内最优 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        order=('ridge','knn','global_fixed','eta13','safety','calibration','crossfit','in_sample')
        for name,row in per_scene.items():f.write('| '+name+' | '+' | '.join(f"{100*row[k]['rate']:.1f}%"for k in order)+' |\n')


if __name__=='__main__':main()
