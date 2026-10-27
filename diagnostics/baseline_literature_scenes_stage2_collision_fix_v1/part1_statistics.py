"""Saved evidence only, identical v2 bootstrap and final instance taxonomy."""
from experiment import *
from audit import V2
from stats_kernel import bootstrap,contrast
from analysis_kernels_resume import classify,weighted_rates,controller_drops,LABELS
from results_fix import row as compact_row
from collections import Counter,defaultdict
import numpy as np

METHODS=('safety','fixed_eta10','fixed_eta13','human_progress')
GRID=[(W,tau) for W in (2.5,5.,10.) for tau in (.02,.05,.1)]

def main(scene):
    selection=read(HERE/'PART1_STAGE_SELECTIONS.json')[scene];stage=selection['stage']
    if not stage:return
    cases=read(HERE/f'{stage}_CASES.json');current=[];old=[];groups={};sources=[]
    for c in cases:
        p=HERE/stage/'raw'/f"{c['rollout_uid']}.json";raw=read(p);dp=HERE/'DIAGNOSTICS'/p.name
        assert dp.exists(),('diagnostic_missing',str(dp));diag=read(dp)
        r=compact_row(raw,diag);r.update(layer='paper_dynamics',scene=scene,deadlock_agents=diag.get('local',[]).count('robot_filter_equilibrium_deadlock'),wall_agents=diag.get('local',[]).count('wall_filter_equilibrium'))
        distances=np.asarray(raw['final_goal_distances']);missing=distances>c['scenario_payload']['config']['goal_tolerance']
        r['near_goal_wandering']=int(bool(raw['timeout']) and bool(missing.any()) and np.all(distances[missing]<=3*c['scenario_payload']['config']['goal_tolerance']))
        current.append(r)
        source=V2/f'optional_dynamics_{scene}'/'raw'/f"{c['paired_v2_rollout_uid']}.json";old.append(compact_row(read(source)))
        sources.append(dict(raw=str(p),sha256=sha(p),diagnostic=str(dp),diagnostic_sha256=sha(dp),v2_raw=str(source),v2_sha256=sha(source)))
        if c['method']!='safety':continue
        ctl=r['controller']
        if ctl not in groups:groups[ctl]=dict(controller=ctl,type=r['type'],episodes=0,successes=0,unknowns=0,counts={g:Counter() for g in GRID},with_deadlock=False)
        g=groups[ctl];g['episodes']+=1;g['successes']+=r['success'];g['unknowns']+=r['unknown'];g['with_deadlock']|=bool(r['deadlock_episode'])
        if not r['success']:
            grids={(x['window_seconds'],x['cone_tolerance']):x['local'] for x in diag['grid']}
            if not grids and r['unknown']:grids={k:['unknown']*r['N'] for k in GRID}
            assert set(grids)==set(GRID)
            for point,cats in grids.items():g['counts'][point].update(cat for cat,failed in zip(cats,missing) if failed)
    instances=[]
    for ctl,g in groups.items():
        upper=(g['successes']+g['unknowns'])/g['episodes'];counts=g.pop('counts');cat=classify(True,upper,counts[(5.,.05)])
        sensitivity=[dict(window_seconds=W,cone_tolerance=tau,classification=classify(True,upper,counts[(W,tau)]),unarrived_agent_episode_categories=dict(counts[(W,tau)])) for W,tau in GRID]
        instances.append(dict(**g,success_lower=g['successes']/g['episodes'],success_upper=upper,classification=cat,
            primary_unarrived_agent_episode_categories=dict(counts[(5.,.05)]),sensitivity=sensitivity,sensitivity_changes=any(x['classification']!=cat for x in sensitivity)))
    first_path=V2/'COMPACT_REVISED_MAIN_RESULTS_20261021.json'
    first=[r for r in read(first_path) if r['scene']==scene and r['layer']=='distributed' and r['method'] in METHODS]
    records={'fixed':current,'v2':old,'first':first};types={r['controller']:r['type'] for r in current};ctls=sorted(types)
    states=sorted({(r['N'],r['state']) for r in current});arms=[(stage,m) for stage in records for m in METHODS]
    lookup={(s,r['controller'],r['N'],r['state'],r['seed'],r['method']):r for s,rr in records.items() for r in rr}
    M=np.empty((len(ctls),len(states),2*len(arms)))
    for i,ctl in enumerate(ctls):
        for j,(n,index) in enumerate(states):
            for k,(s,m) in enumerate(arms):
                rr=[lookup[s,ctl,n,index,seed,m] for seed in (0,1)]
                M[i,j,2*k]=np.mean([r['success'] for r in rr]);M[i,j,2*k+1]=np.mean([r['upper'] for r in rr])
    failed={r['controller'] for r in instances if r['classification']==LABELS[2]};global_fail={r['controller'] for r in instances if r['classification']==LABELS[0]}
    scopes={'all_qualified':ctls,'controller_failure':[c for c in ctls if c in failed],'remaining_instances':[c for c in ctls if c not in failed],'safety_layer_global_failure':[c for c in ctls if c in global_fail]}
    cache={};inferences={}
    for scope,names in scopes.items():
        key=tuple(names)
        if not names:inferences[scope]=dict(status='N/A',reason='No instances in this class');continue
        if key in cache:inferences[scope]=cache[key];continue
        ix=[ctls.index(c) for c in names];point,draws=bootstrap(M[ix],np.asarray([types[c] for c in names]));methods={}
        for m in METHODS:
            k=2*arms.index(('fixed',m));r=dict(success=float(point[k]),success_ci95=np.quantile(draws[:,k],[.025,.975]).tolist(),success_upper=float(point[k+1]),
                versus_v2=contrast(point,draws,arms,{('fixed',m):1,('v2',m):-1}),versus_first=contrast(point,draws,arms,{('fixed',m):1,('first',m):-1}))
            if m!='safety':
                r['correction_gain']=contrast(point,draws,arms,{('fixed',m):1,('fixed','safety'):-1})
                r['gain_vs_first_difference']=contrast(point,draws,arms,{('fixed',m):1,('fixed','safety'):-1,('first',m):-1,('first','safety'):1})
            methods[m]=r
        cache[key]=dict(status='complete',controllers=names,methods=methods,bootstrap_iterations=5000)
        inferences[scope]=cache[key]
        print(scene,scope,'5000 bootstrap complete',flush=True)
    write(HERE/f'STATISTICS_{scene}.json',dict(scene=scene,stage=stage,planned=len(cases),completed=len(current),counts={k:sum(r[k] for r in current) for k in ('success','collision','timeout','unknown','constraint_violation_steps','sampled_speed_violation_steps','acceleration_bound_violation_steps')},
          inferences=inferences,weighted_rates=weighted_rates(current,('near_goal_wandering','direct_stationary_episode','direct_stationary_agent_fraction')),controller_drops=controller_drops(current),instances=instances,
          discrimination=dict(qualified=len(instances),controller_failures=len(failed),safety_global_failures=len(global_fail),state_related=sum(r['success_lower']>.05 and r['success_upper']<.95 for r in instances),with_4a_prime=sum(r['with_deadlock'] for r in instances)),
          first_stage_source_sha256=sha(first_path),new_rollouts=0))
    write(HERE/f'COMPACT_FIXED_{scene}.json',current)
    write(HERE/f'STATISTICS_SOURCES_{scene}.json',sources)
if __name__=='__main__':main(sys.argv[1])
