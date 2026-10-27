"""Completed-data summaries; no failed/unknown rollout is imputed successful."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent

def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def loadrows(name):return list(map(json.loads,(OUT/name).open()))if(OUT/name).exists()else[]
def avg(x):return float(np.mean(x))if len(x)else None
def key(r):return (r.get('scene',r.get('scenario')),r.get('state_index',r.get('legacy_state_uid')),r.get('controller','flow'),r['seed'])
def block(r):return key(r)[:2]

def ci(values):
    v=np.asarray(values,float)
    if len(v)<2:return None
    rng=np.random.default_rng(20261027);samples=v[rng.integers(0,len(v),(10000,len(v)))].mean(axis=1)
    return np.quantile(samples,[.025,.975]).tolist()

def local_escape(r):
    if not r.get('trace'):return dict(stall_onset=None,escape_delay=None)
    a=np.load(r['trace']);p=a['positions'];u=a['actions'];g=a['goals'];d=np.linalg.norm(g[None]-p,axis=-1)
    speed=np.linalg.norm(u,axis=-1);window=40;start=None;group=None
    for t in range(window,len(p)):
        mask=(d[t-window]-d[t]<.05)&(speed[t-window:t].mean(axis=0)<.025)&(d[t]>.08)
        if mask.any():start=t;group=np.flatnonzero(mask);break
    if start is None:return dict(stall_onset=None,escape_delay=None)
    for t in range(start+1,len(p)):
        if np.all((d[start,group]-d[t,group]>.05)|(d[t,group]<=.08)):
            return dict(stall_onset=start*.05,escape_delay=(t-start)*.05,group=group.tolist())
    return dict(stall_onset=start*.05,escape_delay=None,group=group.tolist())

def metric(rows,baseline=None):
    n=len(rows);groups=defaultdict(list)
    for r in rows:groups[block(r)].append(r['success'])
    m=dict(n=n,success=sum(r['success']for r in rows),rate=avg([r['success']for r in rows]),
        cluster_bootstrap_ci=ci([np.mean(v)for v in groups.values()]),physical_state_blocks=len(groups),
        outcomes=dict(Counter(r['outcome']for r in rows)),conditional_completion_time=avg([r['completion_time']for r in rows if r['success']]),
        truncated_time=avg([r['truncated_time']for r in rows]),mean_wall_seconds=avg([r['elapsed_seconds']for r in rows]),
        mean_ms_per_step=avg([1000*r['elapsed_seconds']/r['episode_length']for r in rows if r['episode_length']]),
        restoration_steps=sum(r.get('restoration_steps',0)for r in rows),numerical_failures=sum(r.get('numerical_failure',0)for r in rows),
        original_constraint_violation_steps=sum(r.get('constraint_violation_steps',0)for r in rows))
    escapes=[r.get('local_escape',{})for r in rows];stalled=[e for e in escapes if e.get('stall_onset')is not None]
    m.update(diagnostic_stall_episodes=len(stalled),diagnostic_escapes=sum(e.get('escape_delay')is not None for e in stalled),
        conditional_local_escape_time=avg([e['escape_delay']for e in stalled if e.get('escape_delay')is not None]))
    if baseline:
        paired=[(r,baseline[key(r)])for r in rows if key(r)in baseline];ss=[(r,b)for r,b in paired if r['success']and b['success']]
        diffs=defaultdict(list)
        for r,b in paired:diffs[block(r)].append(r['success']-b['success'])
        m.update(paired_success_n=len(ss),paired_success_method_time=avg([r['completion_time']for r,b in ss]),
            paired_success_baseline_time=avg([b['completion_time']for r,b in ss]),
            paired_success_time_delta=avg([r['completion_time']-b['completion_time']for r,b in ss]),
            paired_rescues=sum(r['success']and not b['success']for r,b in paired),
            paired_breaks=sum(b['success']and not r['success']for r,b in paired),
            paired_rate_delta_ci=ci([np.mean(v)for v in diffs.values()]),
            paired_truncated_delta=avg([r['truncated_time']-b['truncated_time']for r,b in paired]))
    return m

def main():
    rows=loadrows('candidates.jsonl');assert len({r['rollout_uid']for r in rows})==len(rows)
    ind=[r for r in rows if r['spec'].get('test_label')];legacy=loadrows('legacy_candidates.jsonl')+loadrows('legacy_prior_candidates.jsonl')
    for r in ind+legacy:r['local_escape']=local_escape(r)
    dump(OUT/'local_escape_metrics.json',[dict(rollout_uid=r['rollout_uid'],**r['local_escape'])for r in ind+legacy])
    labels=list(dict.fromkeys(r['spec']['test_label']for r in ind));baseline={key(r):r for r in ind if r['spec']['test_label']=='B3'}
    independent={name:metric([r for r in ind if r['spec']['test_label']==name],baseline)for name in labels}
    ind_scene={s:{name:metric([r for r in ind if r['scene']==s and r['spec']['test_label']==name])for name in labels}for s in sorted({r['scene']for r in ind})}
    ind_ctl={c:{name:metric([r for r in ind if r['controller']==c and r['spec']['test_label']==name])for name in labels}for c in sorted({r['controller']for r in ind})}
    methods=list(dict.fromkeys(r['method']for r in legacy));lb={key(r):r for r in legacy if r['method']=='B3_archived'}
    legacy_metrics={m:metric([r for r in legacy if r['method']==m],lb)for m in methods}
    legacy_scene={s:{m:metric([r for r in legacy if r['scenario']==s and r['method']==m])for m in methods}for s in sorted({r['scenario']for r in legacy})}
    f40=list(map(tuple,json.load(open(OUT/'expanded_A40_failures.json'))));hard=list(map(tuple,json.load(open(OUT/'expanded_A168_failures.json'))))
    def dk(r):return(r['scene'],0 if r['controller']=='centralized_sampling'else 1,r['state_index'])
    dev=[r for r in rows if not r['spec'].get('test_label')and r['seed']==0]
    b3names=['eta'+str(i)for i in range(16)]+['b3_'+str(i)for i in range(24)]
    base={}
    for r in dev:
        if r['spec']['name']in b3names:base.setdefault(dk(r),{})[r['spec']['name']]=r
    curve=[]
    for K in (8,16,24,32,40):
        success=sum(any(r['success']for name,r in rr.items()if name in b3names[:K])for rr in base.values())
        curve.append(dict(method='B3',budget=K,solved=success,cohort=len(base)))
    for extra in (16,32,64,128):
        solved=len(base)-len(f40)
        for k in f40:
            solved+=any(r['success']for r in dev if dk(r)==k and r['spec']['name']in ['b3_deep_'+str(i)for i in range(extra)])
        curve.append(dict(method='B3',budget=40+extra,solved=int(solved),cohort=len(base)))
    fixed={};attribution={}
    for mode in ('add','blend'):
        for recipe in ('goal','cluster','wall'):
            pool=[r for r in dev if dk(r)in hard and r['spec']['mode']==mode and r['spec'].get('recipe')==recipe and not r['spec'].get('keep')and not r['spec'].get('schedule')and not r['spec']['name'].endswith('_g0')]
            solved={dk(r)for r in pool if r['success']}
            fixed[mode+'_'+recipe]=dict(candidates=len(pool),candidate_success=sum(r['success']for r in pool),rescued=len(solved),rescued_cases=sorted(solved),remaining=sorted(set(hard)-solved),
                collision=sum(r.get('collision',0)for r in pool),unknown=sum(r.get('numerical_failure',0)for r in pool),seconds=sum(r['elapsed_seconds']for r in pool))
            attribution[mode+'_'+recipe]={label:sorted({dk(r)for r in pool if r['success']and cond(r['spec'])})for label,cond in [
                ('T_only',lambda s:s['eta'][4]==0),('Y_only',lambda s:s['eta'][3]==0 and s['eta'][4]!=0),('T_and_Y',lambda s:s['eta'][3]!=0 and s['eta'][4]!=0)]}
            # b5_pool order is explicit; do not use completion-order JSONL for budgets.
            ordered=[]
            for sign in (-.75,.75):ordered.append(f'{mode}_{recipe}_T{sign}')
            ordered += [f'{mode}_{recipe}_r{ri}_t{sign}'for ri in range(26)for sign in(0.,.75,-.75)]
            ordered += [f'{mode}_{recipe}_sob{i}'for i in range(16)]
            for K in (2,8,16,32,64,80,96):
                win={dk(r)for r in pool if r['success']and r['spec']['name']in ordered[:K]}
                curve.append(dict(method=mode+'_'+recipe,budget=168+K,solved=64-len(hard)+len(win),cohort=64,
                    hard_queue_rescued=len(win),hard_queue_n=len(hard)))
    union={mode:sorted(set().union(*(set(map(tuple,fixed[mode+'_'+t]['rescued_cases']))for t in ('goal','cluster','wall'))))for mode in ('add','blend')}
    ablations={}
    for label in ('GRTY','GPTY','GTR','GTY','noT','noY'):
        rr=[r for r in dev if r['spec'].get('ablation')==label and dk(r)in hard and r['spec']['mode']=='add'
            and not r['spec'].get('schedule')and not r['spec']['name'].endswith('_g0')]
        ablations[label]=dict(candidates=len(rr),success=sum(r['success']for r in rr),cases_rescued=sorted({dk(r)for r in rr if r['success']}),
                             reused=sum(r['source']=='exact_zero_coefficient_reuse'for r in rr))
    schedules={}
    for mode in ('b3','add','blend'):
        rr=[r for r in dev if dk(r)in hard and r['spec']['mode']==mode and(r['spec'].get('theta')is not None or r['spec'].get('schedule'))and not r['spec'].get('keep')]
        schedules[mode]=dict(candidates=len(rr),success=sum(r['success']for r in rr),cases_rescued=sorted({dk(r)for r in rr if r['success']}))
    ranges={}
    for mode in('b3','add'):
        rr=[r for r in dev if r['spec']['mode']==mode and(r['spec']['name'].startswith('b3g0')or r['spec']['name'].endswith('_g0'))and not r['spec'].get('keep')]
        ranges[mode]=dict(candidates=len(rr),success=sum(r['success']for r in rr),cases_rescued=sorted({dk(r)for r in rr if r['success']}))
    equivalent_seeds={}
    for name in labels:
        rr=[r for r in ind if r['spec']['test_label']==name];pairs=defaultdict(list)
        for r in rr:pairs[key(r)[:3]].append(r)
        equal=0
        for pair in pairs.values():
            if len(pair)==2:
                a,b=[np.load(r['trace'])['positions']for r in pair];equal+=np.array_equal(a,b)
        equivalent_seeds[name]=dict(pairs=len(pairs),bitwise_identical_trajectories=int(equal))
    summary=dict(independent=independent,independent_by_scene=ind_scene,independent_by_controller=ind_ctl,
        legacy=legacy_metrics,legacy_by_scene=legacy_scene,oracle=dict(cohorts=len(base),A40_failed=len(f40),A168_failed=len(hard),fixed=fixed,recipe_union=union,
        attribution=attribution,ablations=ablations,schedules=schedules,goal_zero_pilot=ranges),budget_curve=curve,
        future_seed_identity=equivalent_seeds,total_rows=len(rows),main_sources=dict(Counter(r['source']for r in rows)),
        legacy_rows=len(legacy),numerical_failures=sum(r.get('numerical_failure',0)for r in rows+legacy),
        main_actual_simulator_seconds=sum(r['elapsed_seconds']for r in rows if r['source']=='new_rollout'),
        legacy_simulator_seconds=sum(r['elapsed_seconds']for r in legacy),
        legacy_zero_extension_checks=sum(r['bitwise_zero_extension_checks']for r in legacy))
    dump(OUT/'summary.json',summary)
    with (OUT/'budget_curves.csv').open('w')as f:
        w=csv.DictWriter(f,lineterminator='\n',fieldnames=['method','budget','solved','cohort','hard_queue_rescued','hard_queue_n']);w.writeheader();w.writerows(curve)
    with (OUT/'independent_metrics.csv').open('w')as f:
        fields=['method','n','success','rate','conditional_completion_time','truncated_time','mean_wall_seconds','mean_ms_per_step','paired_success_n','paired_success_method_time','paired_success_baseline_time','paired_truncated_delta','diagnostic_stall_episodes','diagnostic_escapes','conditional_local_escape_time']
        w=csv.DictWriter(f,lineterminator='\n',fieldnames=fields);w.writeheader()
        for name,m in independent.items():w.writerow(dict(method=name,**{k:m[k]for k in fields[1:]}))
    try:
        import matplotlib
    except ImportError:
        print('Summary complete; use plot.py with .venv-c1-plots for figures.');return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for name in ['B3']+list(fixed):
        rr=[r for r in curve if r['method']==name];ax[0].plot([r['budget']for r in rr],[r['solved']/64 for r in rr],marker='.',label=name)
    ax[0].set(xlabel='Candidate budget (staged search)',ylabel='Solved development cases / 64',ylim=(.4,1.02));ax[0].legend(fontsize=7,ncol=2);ax[0].grid(alpha=.2)
    names=['safety','B3','T_only','add_goal','blend_goal','add_cluster','blend_cluster','RHR_A','RHR_B']
    ax[1].barh(names,[independent[n]['rate']for n in names]);ax[1].set(xlabel='Frozen independent test success / 32',xlim=(0,1));ax[1].grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'results.png',dpi=180);fig.savefig(OUT/'results.pdf');plt.close(fig)
    print(json.dumps({k:summary[k]for k in('total_rows','main_sources','legacy_rows','numerical_failures')},indent=2))

if __name__=='__main__':main()
