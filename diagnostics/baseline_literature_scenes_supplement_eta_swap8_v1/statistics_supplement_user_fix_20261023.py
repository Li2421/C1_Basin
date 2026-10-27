"""Evidence-only paired v2 kernels; fixed selections never reselected in bootstrap."""
import bootstrap
from experiment import *
from evidence_store import load,locate
from stats_kernel import bootstrap as clustered_bootstrap,contrast
from analysis_kernels_resume import classify,LABELS,weighted_rates
from collections import defaultdict,Counter
import numpy as np

METHODS=('safety','eta10','eta13','eta13_to_10')
NEW=('lit_swap8_free','lit_swap8_obstacles','lit_swap8_moving')
GRID=[(W,tau) for W in (2.5,5.,10.) for tau in (.02,.05,.1)]

def infer(rows,arms,seeds):
    if not rows:return dict(status='N/A',reason='No qualifying available controller cohort')
    types={r['controller']:r['type'] for r in rows};ctls=sorted(types);states=sorted({(r['N'],r['state']) for r in rows});lookup={(r['controller'],r['N'],r['state'],r['seed'],r['action']):r for r in rows}
    M=np.full((len(ctls),len(states),2*len(arms)),np.nan)
    for i,ctl in enumerate(ctls):
        for j,(n,index) in enumerate(states):
            for k,a in enumerate(arms):
                rr=[lookup[(ctl,n,index,seed,a)] for seed in seeds if (ctl,n,index,seed,a) in lookup]
                assert len(rr)==len(seeds),('Incomplete paired cohort',ctl,n,index,a,len(rr))
                M[i,j,2*k]=np.mean([r['success'] for r in rr]);M[i,j,2*k+1]=np.mean([r['upper'] for r in rr])
    point,draws=clustered_bootstrap(M,np.array([types[c] for c in ctls]));result={}
    def diff(a,b):
        if a==b:return dict(delta=0.,ci95=[0.,0.],identified_bounds=[0.,0.],unknown_sensitive_ci95=[0.,0.])
        return contrast(point,draws,arms,{a:1,b:-1})
    for a in arms:
        k=2*arms.index(a);result[a]=dict(success_lower=float(point[k]),success_upper=float(point[k+1]),ci95=np.quantile(draws[:,k],[.025,.975]).tolist(),versus={b:diff(a,b) for b in arms})
    return dict(status='complete',methods=result,controllers=ctls,types=sorted(set(types.values())),states=states,seeds=list(seeds),bootstrap_iterations=5000,bootstrap_seed=2026102003,weighting='Equal type -> instance -> state; seeds remain paired',fixed_training_selection=True)

def compact(stage,c,diagnostics=False):
    r=load(stage,c);s=c['state'];ctl=c['controller_payload']['nominal'];row=dict(rollout_uid=c['rollout_uid'],controller=ctl['controller'],type=ctl['type_code'],scene=s['scene'],N=s['N'],state=s['index'],seed=c['seed'],action=c['requested_action'],success=r['success'],upper=min(1,r['success']+r['numerical_failure']),unknown=r['numerical_failure'],collision=r['collision'],timeout=r['timeout'],constraint_violation_steps=r.get('constraint_violation_steps',0),sampled_speed_violation_steps=r.get('sampled_speed_violation_steps',0),restoration_steps=r.get('restoration_steps',0),collision_types=r.get('collision_types',{}),raw_path=str(locate(stage,c)))
    if diagnostics:
        p=HERE/'DIAGNOSTICS'/f"{c['rollout_uid']}.json";assert p.exists(),str(p);d=read(p);row['diagnostic_path']=str(p);row['categories']=d['local'];row['grid']=d['grid'];row.update(deadlock_episode=int('robot_filter_equilibrium_deadlock' in d['local']),wall_episode=int('wall_filter_equilibrium' in d['local']),static_wall_agents=d.get('static_wall_agents',0),moving_wall_agents=d.get('moving_wall_agents',0),canonical_joint_episode=int('filter_equilibrium_deadlock' in d.get('canonical_joint',[])),actual_joint_episode=int('filter_equilibrium_deadlock' in d.get('actual_product_joint',[])))
        row.update(layer='distributed',method=row['action'],deadlock_agents=d['local'].count('robot_filter_equilibrium_deadlock'),wall_agents=d['local'].count('wall_filter_equilibrium'),static_wall_agent_fraction=d.get('static_wall_agents',0)/s['N'],moving_wall_agent_fraction=d.get('moving_wall_agents',0)/s['N'])
        distances=np.asarray(r['final_goal_distances']);row['unarrived']=(distances>c['scenario_payload']['config']['goal_tolerance']).tolist();missing=np.array(row['unarrived']);row['near_goal_wandering']=int(bool(r['timeout']) and bool(missing.any()) and np.all(distances[missing]<=3*c['scenario_payload']['config']['goal_tolerance']))
    return row

def eta():
    chosen={}
    for which in ('old','new'):chosen.update(read(HERE/f'{which.upper()}_FROZEN_CHOICES.json')['scenes'])
    chosen['lit_swap8_moving']=dict(read(HERE/'MOVING_FROZEN_CHOICES.json')['choice'],transferred_from='lit_swap8_free',has_own_training_ranking=False)
    results={}
    for stage in ('old_eval','new_eval','moving_eval'):
        rows=[compact(stage,c) for c in read(HERE/f'{stage}_CASES.json')]
        for scene in sorted({r['scene'] for r in rows}):
            rr=[r for r in rows if r['scene']==scene];arms=list(dict.fromkeys(r['action'] for r in rr));results[scene]=dict(selection=chosen[scene],evaluation=infer(rr,arms,(0,)),counts={a:{k:sum(r[k] for r in rr if r['action']==a) for k in ('success','unknown','collision','timeout')} for a in arms})
    for scene,c in chosen.items():
        if scene not in results:results[scene]=dict(selection=c,evaluation=dict(status='N/A',reason='No qualifying controller/available training cohort'))
    distinct={s:(r['selection'].get('best_fixed'),r['selection'].get('best_pair')) for s,r in results.items() if 'best_fixed' in r['selection']}
    write(HERE/'ETA_SCREEN.json',dict(utc=utc(),scenes=results,winners=distinct,best_fixed_differs=len({x[0] for x in distinct.values()})>1,best_pair_differs=len({x[1] for x in distinct.values()})>1,moving_excluded_from_screen=True,new_rollouts=0))
    print('Eta statistics complete',flush=True)

def swap():
    cases=read(HERE/'swap_main_CASES.json');qual={s:{r['controller'] for r in read(HERE/'QUALIFICATION_PRIMARY_20261023.json') if r['scene']==s and r['qualified']} for s in NEW};rows=[compact('swap_main',c,True) for c in cases];results={}
    for scene in NEW:
        rr=[r for r in rows if r['scene']==scene];instances=[]
        for ctl in sorted({r['controller'] for r in rr}):
            base=[r for r in rr if r['controller']==ctl and r['action']=='safety'];assert len(base)==64;success=sum(r['success'] for r in base);unknown=sum(r['unknown'] for r in base);upper=(success+unknown)/64;counts={p:Counter() for p in GRID}
            for r in base:
                if r['success']:continue
                grids={(d['window_seconds'],d['cone_tolerance']):d['local'] for d in r['grid']}
                if r['unknown']:grids={p:['unknown']*8 for p in GRID}
                assert set(grids)==set(GRID)
                for p,categories in grids.items():counts[p].update(cat for cat,missing in zip(categories,r['unarrived']) if missing)
            eligible=ctl in qual[scene];primary=classify(eligible,upper,counts[(5.,.05)]);sensitivity=[dict(window_seconds=W,cone_tolerance=t,classification=classify(eligible,upper,counts[(W,t)]),unarrived_agent_categories=dict(counts[(W,t)])) for W,t in GRID]
            instances.append(dict(controller=ctl,type=base[0]['type'],qualified=eligible,success_lower=success/64,success_upper=upper,classification=primary,primary_unarrived_agent_categories=dict(counts[(5.,.05)]),sensitivity=sensitivity,sensitivity_changes=any(s['classification']!=primary for s in sensitivity),has_4a_prime=any(r['deadlock_episode'] for r in base)))
        failed={r['controller'] for r in instances if r['classification']==LABELS[2]};globalfail={r['controller'] for r in instances if r['classification']==LABELS[0]};names=qual[scene]
        scopes={'all_qualified':names,'controller_failure':failed,'remaining_instances':names-failed,'safety_layer_global_failure':globalfail};cache={};inferences={}
        for name,scope in scopes.items():
            key=tuple(sorted(scope))
            if key not in cache:cache[key]=infer([r for r in rr if r['controller'] in scope],list(METHODS),(0,1))
            inferences[name]=cache[key]
        counts={a:{k:sum(r[k] for r in rr if r['action']==a) for k in ('success','unknown','collision','timeout','constraint_violation_steps','sampled_speed_violation_steps','restoration_steps','deadlock_episode','wall_episode','static_wall_agents','moving_wall_agents','canonical_joint_episode','actual_joint_episode','near_goal_wandering')} for a in METHODS}
        collisions={a:{kind:sum(r['collision_types'].get(kind,0) for r in rr if r['action']==a) for kind in ('agent','static','moving')} for a in METHODS}
        results[scene]=dict(qualified=len(names),main_requests=len(rr),all_48_baseline=True,instances=instances,counts=counts,collision_types=collisions,inferences=inferences,weighted_rates_all_instances=weighted_rates(rr,('static_wall_agent_fraction','moving_wall_agent_fraction','near_goal_wandering')),weighted_rates_qualified=weighted_rates([r for r in rr if r['controller'] in names],('static_wall_agent_fraction','moving_wall_agent_fraction','near_goal_wandering')),discrimination=dict(qualified=len(names),state_related=sum(r['qualified'] and r['success_lower']>.05 and r['success_upper']<.95 for r in instances),with_4a_prime=sum(r['qualified'] and r['has_4a_prime'] for r in instances),controller_failure=len(failed),safety_layer_global_failure=len(globalfail)))
        print('Swap statistics complete',scene,flush=True)
    write(HERE/'SWAP8_COMPACT.json',rows)
    # Matched robot states and RNG; physical moving obstacle is the treatment.
    pairrows=[];lookup={(r['scene'],r['controller'],r['state'],r['seed'],r['action']):r for r in rows};effects={}
    for action in METHODS:
        eligible={r['controller'] for r in rows if r['scene']=='lit_swap8_free' and r['action']==action}&{r['controller'] for r in rows if r['scene']=='lit_swap8_moving' and r['action']==action};paired=[];created=broken=0
        for ctl in sorted(eligible):
            for index in range(32):
                for seed in (0,1):
                    a=lookup['lit_swap8_free',ctl,index,seed,action];b=lookup['lit_swap8_moving',ctl,index,seed,action];created+=not a['deadlock_episode'] and bool(b['deadlock_episode']);broken+=bool(a['deadlock_episode']) and not b['deadlock_episode']
                    for scene,r in [('free',a),('moving',b)]:paired.append(dict(r,action=scene))
        effects[action]=dict(inference=infer(paired,['free','moving'],(0,1)),deadlock_created=int(created),deadlock_removed_candidate=int(broken),matched_episodes=len(paired)//2,interpretation='Raw paired candidates; capability-qualified evidence and collision/timeouts must be distinguished. Original invalidated moving/layout data excluded')
    write(HERE/'SWAP8_RESULTS.json',dict(utc=utc(),scenes=results,moving_vs_free=effects,new_rollouts=0,user_fix='20261023',original_constraint_layout_excluded=True,timeout_is_not_deadlock=True,unqualified_baseline_retained=True,deadlock_evidence_requires_capability_gate=True))
if __name__=='__main__':eta() if sys.argv[1]=='eta' else swap()
