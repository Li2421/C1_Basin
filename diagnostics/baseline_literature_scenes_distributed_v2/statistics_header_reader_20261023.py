"""Paired type/controller/state inference, with exact unknown bounds."""
from experiment import *
from result_inventory import find_raw,qualification
from reclassify_instances_20261021 import small_diagnosis
import numpy as np
import collections
import csv

FIXED=('safety','fixed_eta10','fixed_eta13','human_progress')
LAYERS=('central','distributed')

def bootstrap(M,types,iterations=5000):
    groups=[np.flatnonzero(types==t) for t in sorted(set(types))]
    point=np.mean([np.mean([np.nanmean(M[i],axis=0) for i in group],axis=0) for group in groups],axis=0)
    rng=np.random.default_rng(2026102003);out=[]
    available=[np.flatnonzero(np.isfinite(v).all(-1)) for v in M]
    for _ in range(iterations):
        ts=[]
        for ti in rng.integers(len(groups),size=len(groups)):
            group=groups[ti];cs=[]
            for ci in rng.choice(group,len(group),replace=True):
                ix=available[ci];cs.append(M[ci,rng.choice(ix,len(ix),replace=True)].mean(0))
            ts.append(np.mean(cs,axis=0))
        out.append(np.mean(ts,axis=0))
    return point,np.asarray(out)

def contrast(point,draws,arms,coefficients):
    lower=[];upper=[];ordinary=[]
    for arm,value in coefficients.items():
        j=2*arms.index(arm)
        ordinary.append((j,value))
        lower.append((j if value>0 else j+1,value))
        upper.append((j+1 if value>0 else j,value))
    calc=lambda vector,pairs:sum(vector[...,j]*value for j,value in pairs)
    return dict(delta=float(calc(point,ordinary)),ci95=np.quantile(calc(draws,ordinary),[.025,.975]).tolist(),
        identified_bounds=[float(calc(point,lower)),float(calc(point,upper))],
        unknown_sensitive_ci95=[float(np.quantile(calc(draws,lower),.025)),float(np.quantile(calc(draws,upper),.975))])

def collect():
    records=[];sources=[]
    stages=[f'{layer}_{kind}_{scene}' for layer in LAYERS for kind in ('baseline','fixed') for scene in NAMES]
    stages += ['learning_eval_'+s for s in NAMES if (HERE/f'learning_eval_{s}_CASES.json').exists()]
    stages += ['central_learning_eval_'+s for s in ('lit_narrow_passage','lit_doorway')]
    for stage in stages:
        for c in read(HERE/f'{stage}_CASES.json'):
            path=find_raw(c['rollout_uid'],stage,c['state']['scene']);raw=read(path)
            diag_path=HERE/'DIAGNOSTICS'/f"{c['rollout_uid']}.json";diag=small_diagnosis(diag_path.read_bytes())
            ctl=c['controller_payload']['nominal'];s=c['state'];cats=diag['local']
            records.append(dict(layer=c.get('safety_layer','central'),scene=s['scene'],N=s['N'],state=s['index'],controller=ctl['controller'],type=ctl['type_code'],method=c['method'],seed=c['seed'],
                success=raw['success'],upper=min(1,raw['success']+raw['numerical_failure']),collision=raw['collision'],timeout=raw['timeout'],unknown=raw['numerical_failure'],
                deadlock_episode=int('robot_filter_equilibrium_deadlock' in cats),wall_episode=int('wall_filter_equilibrium' in cats),
                deadlock_agents=cats.count('robot_filter_equilibrium_deadlock'),wall_agents=cats.count('wall_filter_equilibrium'),categories=cats,
                canonical_joint_episode=int('filter_equilibrium_deadlock' in diag.get('canonical_joint',[])),actual_joint_episode=int('filter_equilibrium_deadlock' in diag.get('actual_product_joint',[])),rollout_uid=c['rollout_uid']))
            sources.append(dict(rollout_uid=c['rollout_uid'],logical_stage=stage,raw=str(path),raw_sha256=sha(path),diagnostic=str(diag_path),diagnostic_sha256=sha(diag_path)))
    return records,sources

def controller_census(records):
    groups=collections.defaultdict(list);out=[]
    for r in records:
        if r['method']=='safety':groups[(r['scene'],r['layer'],r['controller'])].append(r)
    q={s:{r['controller'] for r in qualification(s) if r['qualified']} for s in NAMES}
    for (scene,layer,ctl),rr in sorted(groups.items()):
        lo=np.mean([r['success'] for r in rr]);hi=np.mean([r['upper'] for r in rr])
        out.append(dict(scene=scene,layer=layer,controller=ctl,type=rr[0]['type'],qualified=ctl in q[scene],episodes=len(rr),success_lower=float(lo),success_upper=float(hi),
            controller_failure_definite=bool(hi<=.05),controller_failure_possible=bool(lo<=.05<hi),
            state_related_failure_definite=bool(lo>.05 and hi<.95),state_related_failure_possible=bool(hi>.05 and lo<.95 and not (lo>.05 and hi<.95)),
            at_least_one_4a_prime=any(r['deadlock_episode'] for r in rr),deadlock_evidence_eligible=bool(ctl in q[scene] and hi>.05)))
    return out

def summarize(records,arms,scope,remove_failures=False,census=()):
    q={s:{r['controller'] for r in qualification(s) if r['qualified']} for s in NAMES}
    failures={(r['scene'],r['layer'],r['controller']) for r in census if r['controller_failure_definite']}
    lookup={};typ={}
    for r in records:
        if r['scene'] not in scope or r['controller'] not in q[r['scene']] or (r['layer'],r['method']) not in arms:continue
        if remove_failures and (r['scene'],r['layer'],r['controller']) in failures:continue
        typ[r['controller']]=r['type'];lookup[(r['controller'],r['scene'],r['N'],r['state'],r['seed'],(r['layer'],r['method']))]=r
    ctls=sorted(typ);states=sorted({(key[1],key[2],key[3]) for key in lookup})
    M=np.full((len(ctls),len(states),2*len(arms)),np.nan);metrics={};counts={str(a):collections.Counter() for a in arms};harm={}
    for ci,ctl in enumerate(ctls):
        for si,state in enumerate(states):
            rows=[[lookup.get((ctl,*state,seed,arm)) for seed in (0,1)] for arm in arms]
            if any(r is None for rr in rows for r in rr):continue
            for ai,(arm,rr) in enumerate(zip(arms,rows)):
                M[ci,si,2*ai]=np.mean([r['success'] for r in rr]);M[ci,si,2*ai+1]=np.mean([r['upper'] for r in rr])
                for r in rr:
                    counts[str(arm)].update({k:r[k] for k in ('success','collision','timeout','unknown','deadlock_episode','wall_episode','deadlock_agents','wall_agents','canonical_joint_episode','actual_joint_episode')})
                    counts[str(arm)]['episodes']+=1;counts[str(arm)]['agents']+=r['N']
            for ai,arm in enumerate(arms):
                if arm[1]=='safety' or (arm[0],'safety') not in arms:continue
                base=rows[arms.index((arm[0],'safety'))]
                h=harm.setdefault(str(arm),dict(known=0,upper_including_unknown=0))
                h['known']+=sum(a['success'] and not b['success'] and not b['unknown'] for a,b in zip(base,rows[ai]))
                h['upper_including_unknown']+=sum(a['success'] and not b['success'] for a,b in zip(base,rows[ai]))
    supported=np.isfinite(M).all(-1).any(-1);M=M[supported];ctls=[c for c,ok in zip(ctls,supported) if ok]
    if not ctls:return dict(status='N/A',reason='No complete qualified paired state clusters')
    types=np.asarray([typ[c] for c in ctls]);tables=[];comparisons=[]
    for label in [*sorted(set(types)),'type_equal']:
        mask=np.ones(len(types),bool) if label=='type_equal' else types==label
        point,draws=bootstrap(M[mask],types[mask])
        for ai,arm in enumerate(arms):
            row=dict(type=label,layer=arm[0],method=arm[1],success_rate=float(point[2*ai]),ci95=np.quantile(draws[:,2*ai],[.025,.975]).tolist(),success_upper=float(point[2*ai+1]),upper_ci95=np.quantile(draws[:,2*ai+1],[.025,.975]).tolist(),controllers=int(mask.sum()))
            base=(arm[0],'safety')
            if arm!=base and base in arms:row['delta_vs_safety']=contrast(point,draws,arms,{arm:1,base:-1})
            if arm[1]=='learned_scene':
                fixes=[(arm[0],m) for m in FIXED[1:] if (arm[0],m) in arms]
                row['delta_vs_fixed']={a[1]:contrast(point,draws,arms,{arm:1,a:-1}) for a in fixes}
                if fixes:
                    ix=[2*arms.index(a) for a in fixes];i=2*ai
                    delta=draws[:,i]-draws[:,ix].max(1)
                    row['delta_vs_best_aggregate_fixed']=dict(delta=float(point[i]-point[ix].max()),ci95=np.quantile(delta,[.025,.975]).tolist(),bootstrap_reselects_best_fixed=True,
                        identified_bounds=[float(point[i]-point[np.asarray(ix)+1].max()),float(point[i+1]-point[ix].max())])
            tables.append(row)
        for m in sorted({a[1] for a in arms}):
            C=('central',m);D=('distributed',m)
            if C in arms and D in arms:
                item=dict(type=label,method=m,distributed_minus_central=contrast(point,draws,arms,{D:1,C:-1}))
                if m!='safety' and ('central','safety') in arms and ('distributed','safety') in arms:
                    item['correction_gain_difference']=contrast(point,draws,arms,{D:1,('distributed','safety'):-1,C:-1,('central','safety'):1})
                comparisons.append(item)
    drops={}
    for arm in arms:
        base=(arm[0],'safety')
        if arm==base or base not in arms:continue
        i=2*arms.index(arm);j=2*arms.index(base)
        lows=np.nanmean(M[...,i],1);ups=np.nanmean(M[...,i+1],1);blo=np.nanmean(M[...,j],1);bup=np.nanmean(M[...,j+1],1)
        drops[str(arm)]=dict(definite_10pp_drop=[c for c,l,u,b in zip(ctls,lows,ups,blo) if u-b<=-.1+1e-12],possible_10pp_drop=[c for c,l,b in zip(ctls,lows,bup) if l-b<=-.1+1e-12])
    return dict(status='complete',statistics=tables,layer_comparisons=comparisons,pooled_counts={k:dict(v) for k,v in counts.items()},harms_vs_safety=harm,controller_10pp_drops=drops,
        controllers=ctls,state_clusters=states,removed_definite_controller_failures=remove_failures,scope=scope,bootstrap_iterations=5000,weighting='Equal type, equal controller within type, equal available (scene,N,index) state within controller; seeds and all arms paired')

def crossfit_oracle(records,layer,scene):
    rr=[r for r in records if r['layer']==layer and r['scene']==scene and r['method'] in FIXED[1:]]
    groups=collections.defaultdict(list)
    for r in rr:groups[r['controller']].append(r)
    choices=[];virtual=[]
    for ctl,rows in groups.items():
        states=sorted({(r['N'],r['state']) for r in rows});fold={s:i%2 for i,s in enumerate(states)}
        for held in (0,1):
            training=[r for r in rows if fold[(r['N'],r['state'])]!=held]
            if not training:continue
            rates={m:np.mean([r['success'] for r in training if r['method']==m]) for m in FIXED[1:]}
            best=max(FIXED[1:],key=lambda m:rates[m]);choices.append(dict(controller=ctl,held_out_state_fold=held,method=best,training_rates=rates))
            virtual += [dict(r,method='fixed_oracle_crossfit') for r in rows if fold[(r['N'],r['state'])]==held and r['method']==best]
    result=summarize([*records,*virtual],[(layer,m) for m in (*FIXED,'fixed_oracle_crossfit')],[scene])
    return dict(descriptive=True,state_folds='Sorted (N,index) parity, both seeds held together; select other fold only; frozen fixed-menu tie order',choices=choices,result=result)

def main():
    prereg();assert not (HERE/'MIGRATION_NUMBERS.receipt.json').exists()
    records,sources=collect();census=controller_census(records);results={}
    arms=[(layer,m) for layer in LAYERS for m in FIXED]
    for scene in NAMES:
        fixed=summarize(records,arms,[scene],census=census)
        removed=summarize(records,arms,[scene],True,census)
        entry=dict(fixed_full=fixed,excluding_controller_failures=removed,learned_evaluation={})
        keys={(r['N'],r['state']) for r in records if r['scene']==scene and r['layer']=='distributed' and r['method']=='learned_scene'}
        subset=[r for r in records if r['scene']==scene and (r['N'],r['state']) in keys]
        if keys:
            learned_arms=[('distributed',m) for m in (*FIXED,'learned_scene')]
            if scene in ('lit_narrow_passage','lit_doorway'):learned_arms += [('central',m) for m in (*FIXED,'learned_scene')]
            entry['learned_evaluation']=summarize(subset,learned_arms,[scene],census=census)
        else:entry['learned_evaluation']=dict(status='N/A',reason=read(HERE/f'LEARNING_GAP_{scene}.json'))
        entry['crossfit_fixed_oracle']={layer:crossfit_oracle(subset or [r for r in records if r['scene']==scene],layer,scene) for layer in LAYERS}
        results[scene]=entry
    results['four_scene_type_equal']=summarize(records,arms,list(NAMES),census=census)
    results['four_scene_excluding_controller_failures']=summarize(records,arms,list(NAMES),True,census)
    learned_scenes=[s for s in NAMES if any(r['scene']==s and r['layer']=='distributed' and r['method']=='learned_scene' for r in records)]
    if learned_scenes:
        keys={(r['scene'],r['N'],r['state']) for r in records if r['layer']=='distributed' and r['method']=='learned_scene'}
        subset=[r for r in records if (r['scene'],r['N'],r['state']) in keys]
        label='four_scene_distributed_learned_evaluation' if len(learned_scenes)==4 else 'supported_scene_distributed_learned_evaluation'
        results[label]=summarize(subset,[('distributed',m) for m in (*FIXED,'learned_scene')],learned_scenes,census=census)
    write(HERE/'MIGRATION_RESULTS.json',dict(utc=utc(),evidence='DEVELOPMENT',results=results))
    write(HERE/'CONTROLLER_LEVEL_FAILURES.json',census);write(HERE/'MIGRATION_RESULT_SOURCES.json',sources)
    write(HERE/'COMPACT_MAIN_RESULTS.json',records)
    fields=('scene','layer','controller','type','N','state','seed','success','collision','timeout','unknown','deadlock_episode','wall_episode','deadlock_agents','wall_agents','rollout_uid')
    with (HERE/'SAFETY_BASELINE.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in records if r['method']=='safety')
    rows=[]
    for scene,entry in results.items():
        table=entry.get('fixed_full',entry)
        rows += [dict(scene=scene,**r) for r in table.get('statistics',[])]
        if 'learned_evaluation' in entry:rows += [dict(scene=scene+'_learned_evaluation',**r) for r in entry['learned_evaluation'].get('statistics',[]) if r['method']=='learned_scene']
    fields=('scene','type','layer','method','success_rate','ci95','success_upper','upper_ci95','controllers')
    with (HERE/'MIGRATION_RESULTS.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows({k:canonical(r[k]) if isinstance(r[k],(list,dict)) else r[k] for k in fields} for r in rows)
    seal('MIGRATION_NUMBERS.receipt.json',[HERE/n for n in ('MIGRATION_RESULTS.json','MIGRATION_RESULTS.csv','CONTROLLER_LEVEL_FAILURES.json','MIGRATION_RESULT_SOURCES.json','COMPACT_MAIN_RESULTS.json','SAFETY_BASELINE.csv','statistics.py')],bootstrap_iterations=5000,before_optional_dynamics_and_charts=True)

if __name__=='__main__':main()
