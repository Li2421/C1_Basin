"""Seal optional numerical evidence separately from immutable main numbers."""
from experiment import *
from result_inventory import find_raw
from statistics import summarize,bootstrap,contrast,FIXED
import numpy as np
import collections

def collect(stage):
    cases=read(HERE/f'{stage}_CASES.json');rows=[];sources=[]
    for c in cases:
        try:path=find_raw(c['rollout_uid'],stage,c['state']['scene'])
        except FileNotFoundError:continue
        raw=read(path);s=c['state'];ctl=c['controller_payload']['nominal'];dp=HERE/'DIAGNOSTICS'/f"{c['rollout_uid']}.json"
        diag=read(dp) if dp.exists() else {};cats=diag.get('local',[])
        rows.append(dict(layer='paper_dynamics' if c['execution_engine']=='paper_dynamics' else 'distributed',scene=s['scene'],N=s['N'],state=s['index'],controller=ctl['controller'],type=ctl['type_code'],method=c['method'],seed=c['seed'],
            success=raw['success'],upper=min(1,raw['success']+raw['numerical_failure']),unknown=raw['numerical_failure'],collision=raw['collision'],timeout=raw['timeout'],
            deadlock_episode=int('robot_filter_equilibrium_deadlock' in cats),wall_episode=int('wall_filter_equilibrium' in cats),deadlock_agents=cats.count('robot_filter_equilibrium_deadlock'),wall_agents=cats.count('wall_filter_equilibrium'),
            canonical_joint_episode=int('filter_equilibrium_deadlock' in diag.get('canonical_joint',[])),actual_joint_episode=int('filter_equilibrium_deadlock' in diag.get('actual_product_joint',[])),
            diagnostics_available=dp.exists(),direct_stationary_episode=any(diag.get('direct_stationary',[])),direct_stationary_agent_fraction=sum(diag.get('direct_stationary',[]))/s['N'],constraint_violation_steps=raw.get('constraint_violation_steps',0),sampled_speed_violation_steps=raw.get('sampled_speed_violation_steps',0),acceleration_bound_violation_steps=raw.get('acceleration_bound_violation_steps',0),rollout_uid=c['rollout_uid']))
        sources.append(dict(rollout_uid=c['rollout_uid'],raw=str(path),raw_sha256=sha(path),diagnostic=str(dp) if dp.exists() else None,diagnostic_sha256=sha(dp) if dp.exists() else None))
    return rows,sources,len(cases)

def paired_stage_comparison(first,second,scene,first_name,second_name,methods):
    from study_cases import qualified
    eligible=qualified(scene);lookup={};types={}
    for name,records in ((first_name,first),(second_name,second)):
        for r in records:
            if r['scene']!=scene or r['controller'] not in eligible or r['method'] not in methods:continue
            types[r['controller']]=r['type'];lookup[(r['controller'],r['N'],r['state'],r['seed'],(name,r['method']))]=r
    ctls=sorted(types);states=sorted({(k[1],k[2]) for k in lookup});arms=[(name,m) for name in (first_name,second_name) for m in methods]
    M=np.full((len(ctls),len(states),2*len(arms)),np.nan)
    for ci,ctl in enumerate(ctls):
        for si,(n,index) in enumerate(states):
            rr=[[lookup.get((ctl,n,index,seed,a)) for seed in (0,1)] for a in arms]
            if any(r is None for row in rr for r in row):continue
            for ai,row in enumerate(rr):M[ci,si,2*ai]=np.mean([r['success'] for r in row]);M[ci,si,2*ai+1]=np.mean([r['upper'] for r in row])
    keep=np.isfinite(M).all(-1).any(-1);M=M[keep];ctls=[c for c,ok in zip(ctls,keep) if ok]
    if not ctls:return dict(status='N/A',reason='No complete qualified pairs')
    point,draws=bootstrap(M,np.asarray([types[c] for c in ctls]));out=[]
    for m in methods:
        a=(first_name,m);b=(second_name,m)
        row=dict(method=m,second_minus_first=contrast(point,draws,arms,{b:1,a:-1}))
        if m!='safety':row['correction_gain_difference']=contrast(point,draws,arms,{b:1,(second_name,'safety'):-1,a:-1,(first_name,'safety'):1})
        out.append(row)
    return dict(status='complete',first=first_name,second=second_name,comparisons=out,controllers=ctls,states=states,bootstrap_iterations=5000)

def main():
    first=read(HERE/'COMPACT_REVISED_MAIN_RESULTS_20261021.json');result={};sources=[];all_dynamics=[]
    stages=sorted(p.name.removesuffix('_CASES.json') for p in HERE.glob('optional_*_CASES.json') if not p.name.endswith('_MISSING_CASES.json'))
    for stage in stages:
        rows,ss,planned=collect(stage);sources+=ss
        counts={key:sum(r[key] for r in rows) for key in ('success','unknown','collision','timeout','constraint_violation_steps','sampled_speed_violation_steps','acceleration_bound_violation_steps','direct_stationary_episode')}
        counts.update(planned=planned,completed=len(rows),diagnosis_unavailable=sum(not r['diagnostics_available'] for r in rows))
        entry=dict(counts=counts,status='complete' if len(rows)==planned else 'N/A',reason=None if len(rows)==planned else 'Optional gap; unwritten outcomes are absent, never unknown')
        entry['diagnosis_status']='complete' if rows and counts['diagnosis_unavailable']==0 else 'N/A: missing saved-trace diagnoses; absent diagnoses are not negative findings'
        if rows and len(rows)==planned:
            scene=rows[0]['scene']
            if stage.startswith('optional_dynamics_'):
                all_dynamics+=rows;entry['statistics']=summarize(rows,[('paper_dynamics',m) for m in FIXED],[scene])
                from derived_metrics import weighted_rates
                if not counts['diagnosis_unavailable']:entry['weighted_rates']=weighted_rates(rows,('direct_stationary_episode','direct_stationary_agent_fraction'))
                entry['joint_diagnostic_definition']='Actual product normal cone and a coupled reference in acceleration/front-velocity coordinates; see DYNAMICS_DIAGNOSTICS.md. This is not reuse of the single-integrator velocity cone.'
                entry['versus_first_stage']=paired_stage_comparison([r for r in first if r['layer']=='distributed'],rows,scene,'single_integrator','paper_dynamics',FIXED)
            elif stage.startswith('optional_full_circle_'):
                entry['statistics']=summarize(rows,[('distributed',m) for m in ('safety','fixed_eta10')],[scene])
                entry['versus_reciprocal']=paired_stage_comparison([r for r in first if r['layer']=='distributed'],rows,scene,'reciprocal','full_responsibility',('safety','fixed_eta10'))
            elif stage.startswith('optional_local_progress_'):
                virtual=[dict(r,method='human_local_progress') for r in rows]
                entry['statistics']=summarize([*first,*virtual],[('distributed',m) for m in (*FIXED,'human_local_progress')],[scene])
                baseline=[r for r in first if r['layer']=='distributed' and r['scene']==scene and r['method']=='safety']
                entry['versus_broadcast']=paired_stage_comparison([r for r in first if r['layer']=='distributed'],[*baseline,*rows],scene,'broadcast_progress','individual_progress',('safety','human_progress'))
            else:
                entry['background_only']=True
                from study_cases import qualified
                q=qualified(scene);baseline={(r['layer'],r['controller'],r['N'],r['state'],r['seed']):r for r in first if r['method']=='safety' and r['scene']==scene}
                transitions={}
                for layer in ('central','distributed'):
                    for scope in ('all48','qualified'):
                        counter=collections.Counter()
                        for r in rows:
                            if scope=='qualified' and r['controller'] not in q:continue
                            if not r['collision']:continue
                            other=baseline[(layer,r['controller'],r['N'],r['state'],r['seed'])]
                            outcome='unknown' if other['unknown'] else 'collision' if other['collision'] else 'success' if other['success'] else 'timeout'
                            counter['nominal_collision_episodes']+=1;counter[outcome]+=1
                        denominator=counter['nominal_collision_episodes']
                        transitions[layer+'_'+scope]=dict(counts=dict(counter),fractions={k:counter[k]/denominator if denominator else None for k in ('success','timeout','unknown','collision')})
                entry['nominal_collision_to_safety_transitions']=transitions
            if counts['diagnosis_unavailable'] and 'statistics' in entry:
                for pooled in entry['statistics'].get('pooled_counts',{}).values():
                    for key in ('deadlock_episode','wall_episode','deadlock_agents','wall_agents','canonical_joint_episode','actual_joint_episode'):pooled[key]=None
        result[stage]=entry
    if all_dynamics and len([s for s in result if s.startswith('optional_dynamics_') and result[s]['status']=='complete'])==4:
        result['dynamics_four_scene_type_equal']=summarize(all_dynamics,[('paper_dynamics',m) for m in FIXED],list(NAMES))
        if any(not r['diagnostics_available'] for r in all_dynamics):
            result['dynamics_four_scene_type_equal']['diagnosis_status']='N/A: one or more saved-trace diagnoses unavailable'
            for pooled in result['dynamics_four_scene_type_equal'].get('pooled_counts',{}).values():
                for key in ('deadlock_episode','wall_episode','deadlock_agents','wall_agents','canonical_joint_episode','actual_joint_episode'):pooled[key]=None
    from optional_revision_diagnostics_20261021 import report
    revision=report(result,all_dynamics,sources)
    write(HERE/'OPTIONAL_RESULTS.json',dict(utc=utc(),results=result,main_numbers_unchanged=True,revision=revision))
    write(HERE/'OPTIONAL_RESULT_SOURCES.json',sources)
    write(HERE/'COMPACT_DYNAMICS_RESULTS.json',all_dynamics)
    seal('OPTIONAL_NUMBERS.receipt.json',[HERE/n for n in ('OPTIONAL_RESULTS.json','OPTIONAL_RESULT_SOURCES.json','COMPACT_DYNAMICS_RESULTS.json','INSTANCE_CLASSES_DYNAMICS_REVISED_20261021.json','optional_revision_diagnostics_20261021.py','optional_results_correction_revised_20261021.py')],main_numbers_unchanged=True)

if __name__=='__main__':main()
