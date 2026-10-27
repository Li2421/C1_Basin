"""Required regret/drop/transition disclosures from already sealed evidence."""
from experiment import *
from statistics import FIXED,summarize
from result_inventory import qualification,find_raw
import numpy as np
import collections

def weighted_rates(rows,extra_metrics=()):
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['layer'],r['method'])].append(r)
    output={}
    metrics=('success','upper','collision','timeout','unknown','deadlock_episode','wall_episode','canonical_joint_episode','actual_joint_episode',*extra_metrics)
    for (layer,method),records in groups.items():
        clusters=collections.defaultdict(list)
        for r in records:clusters[(r['type'],r['controller'],r['scene'],r['N'],r['state'])].append(r)
        ctls=collections.defaultdict(list)
        for (type_code,ctl,scene,n,index),rr in clusters.items():
            if {r['seed'] for r in rr}!={0,1}:continue
            vector=np.asarray([[*[r[k] for k in metrics],r['deadlock_agents']/r['N'],r['wall_agents']/r['N']] for r in rr],float).mean(0)
            ctls[(type_code,ctl)].append(vector)
        types=collections.defaultdict(list)
        for (type_code,ctl),values in ctls.items():types[type_code].append(np.mean(values,axis=0))
        if not types:continue
        point=np.mean([np.mean(values,axis=0) for values in types.values()],axis=0)
        output[layer+'|'+method]=dict(rates=dict(zip((*metrics,'deadlock_agent_fraction','wall_agent_fraction'),point.tolist())),controllers=len(ctls),types=sorted(types),scenes=sorted({r['scene'] for r in records}),state_clusters=len(clusters),weighting='Equal type -> controller -> available (scene,N,index) -> paired seeds')
    return output

def controller_drops(rows):
    groups=collections.defaultdict(dict)
    for r in rows:groups[(r['layer'],r['scene'],r['controller'])].setdefault(r['method'],[]).append(r)
    out=[]
    for (layer,scene,ctl),methods in sorted(groups.items()):
        for method,rr in methods.items():
            for reference,label in [('fixed_eta10','catastrophe_vs_F10'),('safety','harm_vs_safety')]:
                if reference not in methods:continue
                keys={(r['N'],r['state'],r['seed']) for r in rr}
                base=[r for r in methods[reference] if (r['N'],r['state'],r['seed']) in keys]
                common={(r['N'],r['state'],r['seed']) for r in base};target=[r for r in rr if (r['N'],r['state'],r['seed']) in common]
                if not common:continue
                lo=np.mean([r['success'] for r in target]);hi=np.mean([r['upper'] for r in target]);blo=np.mean([r['success'] for r in base]);bhi=np.mean([r['upper'] for r in base])
                out.append(dict(layer=layer,scene=scene,controller=ctl,method=method,metric=label,paired_episodes=len(common),paired_scenes=sorted({r.get('original_scene',r['scene']) for r in target}),definite_drop_10pp=bool(hi-blo<=-.1+1e-12),possible_drop_10pp=bool(lo-bhi<=-.1+1e-12),delta_lower=float(lo-bhi),delta_upper=float(hi-blo)))
    return out

def training_menu(scene):
    bundle=HERE/f'LEARNING_BUNDLE_{scene}.npz'
    if not bundle.exists():return dict(status='N/A',reason='Scene was not informative under the preregistered rule')
    from learn_critic import source_key
    frozen=read(HERE/f'FROZEN_SCENE_CRITIC_{scene}.json');rows=[];prediction_sources=[]
    with np.load(bundle) as z:labels=z['labels'];types=z['types'];controllers=z['controllers']
    for ci,ctl in enumerate(controllers):
        fold=frozen['folds'][str(types[ci])]
        path=HERE/'critic_fits'/scene/source_key(fold['source_types'])/f"step{fold['step']}_predictions.npz"
        with np.load(path) as z:choices=np.argmax(z['probabilities'].mean(0),axis=-1)
        lower=(labels[ci]==1).astype(float);upper=(labels[ci]!=0).astype(float)
        best=int(np.argmax(lower.mean(0)));policy_lower=float(lower[np.arange(32),choices].mean());policy_upper=float(upper[np.arange(32),choices].mean())
        crosslower=[];crossupper=[]
        for held in (0,1):
            train=np.arange(32)%2!=held;test=~train;chosen=int(np.argmax(lower[train].mean(0)))
            crosslower+=lower[test,chosen].tolist();crossupper+=upper[test,chosen].tolist()
        rows.append(dict(controller=str(ctl),type=str(types[ci]),oracle_fixed_menu_action=best,oracle_in_sample_lower=float(lower.mean(0).max()),oracle_in_sample_upper=float(upper.mean(0).max()),
            oracle_state_crossfit_lower=float(np.mean(crosslower)),oracle_state_crossfit_upper=float(np.mean(crossupper)),held_out_type_policy_lower=policy_lower,held_out_type_policy_upper=policy_upper,
            signed_regret_vs_best_constant_menu_action=float(lower.mean(0).max()-policy_lower)))
        prediction_sources.append(dict(path=str(path),sha256=sha(path)))
    regret=[r['signed_regret_vs_best_constant_menu_action'] for r in rows]
    type_means=[np.mean([r['signed_regret_vs_best_constant_menu_action'] for r in rows if r['type']==t]) for t in sorted(set(r['type'] for r in rows))]
    return dict(status='descriptive_training_states_only',rows=rows,mean_signed_regret_type_equal=float(np.mean(type_means)),maximum_signed_regret=float(max(regret)),prediction_sources=prediction_sources,
        independent_evaluation_full_menu_regret='N/A: only chosen learned action and four reference arms are requested on independent evaluation states; do not generate a new twelve-action evaluation oracle',
        caveat='A state-dependent learned policy can outperform the best constant action; this signed comparator difference may be negative. Training-state oracle is not independent-state performance.')

def diagnosis_details(rows):
    from legacy_failure_analysis import cone_escape
    sensitivity=[];census=[];escape_cache={}
    for r in rows:
        path=HERE/'DIAGNOSTICS'/f"{r['rollout_uid']}.json";diag=read(path)
        if r['method']=='safety':
            grids=diag.get('grid',[])
            if not grids and r['unknown']:
                grids=[dict(window_seconds=W,cone_tolerance=tau,local=['unknown']*r['N'],canonical_joint=['unknown']*r['N'],actual_product_joint=['unknown']*r['N']) for W in (2.5,5.,10.) for tau in (.02,.05,.1)]
            for grid in grids:
                cats=grid['local'];sensitivity.append(dict(r,method=f"safety_W{grid['window_seconds']}_tau{grid['cone_tolerance']}",
                    deadlock_episode=int('robot_filter_equilibrium_deadlock' in cats),wall_episode=int('wall_filter_equilibrium' in cats),
                    deadlock_agents=cats.count('robot_filter_equilibrium_deadlock'),wall_agents=cats.count('wall_filter_equilibrium'),
                    canonical_joint_episode=int('filter_equilibrium_deadlock' in grid.get('canonical_joint',[])),actual_joint_episode=int('filter_equilibrium_deadlock' in grid.get('actual_product_joint',[]))))
        entries=diag.get('census',[])
        if not entries:continue
        speeds=[];bias=[];angles=[];distances=[];reasons=collections.Counter();neighbours=[]
        for item in entries:
            A=np.asarray(item['A'],float).reshape(-1,2);u=np.asarray(item['nominal'],float)
            speeds.append(item['safe_speed']);bias+=item['conditional_lower']
            if len(A):angles.extend((np.arccos(np.clip(A@u/np.maximum(np.linalg.norm(A,axis=1)*np.linalg.norm(u),1e-30),-1,1))*180/np.pi).tolist())
            key=(A.tobytes(),u.tobytes())
            if key not in escape_cache:
                try:escape_cache[key]=cone_escape(u,A)
                except Exception as exc:escape_cache[key]=dict(infimum=None,reason='Posthoc_escape_numeric_gap',error=str(exc))
            escape=escape_cache[key];reasons[escape['reason']]+=1
            if escape.get('infimum') is not None:distances.append(escape['infimum'])
            neighbours.append(len({k[2] if k[1]==item['agent'] else k[1] for k in item['labels'] if k[0]=='pair'}))
        census.append(dict(scene=r['scene'],layer=r['layer'],method=r['method'],controller=r['controller'],rollout_uid=r['rollout_uid'],candidate_agent_steps=len(entries),
            maximum_safe_speed=max(speeds),minimum_conditional_bound=min(bias) if bias else None,maximum_conditional_bound=max(bias) if bias else None,
            minimum_normal_angle_degrees=min(angles) if angles else None,maximum_normal_angle_degrees=max(angles) if angles else None,
            minimum_local_cone_escape_infimum=min(distances) if distances else None,maximum_local_cone_escape_infimum=max(distances) if distances else None,
            escape_reasons=dict(reasons),maximum_incident_active_pair_neighbours=max(neighbours),scope='Actual two-dimensional agent constraint blocks; descriptive finite-window candidates'))
    pooled=weighted_rates(sensitivity)
    by_scene={s:weighted_rates([r for r in sensitivity if r['scene']==s]) for s in NAMES}
    return dict(scenes=by_scene,qualified_scene_pool=pooled),census

def main():
    rows=read(HERE/'COMPACT_MAIN_RESULTS.json');eligible={s:{r['controller'] for r in qualification(s) if r['qualified']} for s in NAMES}
    qualified=[r for r in rows if r['controller'] in eligible[r['scene']]]
    pooled=[dict(r,original_scene=r['scene'],scene='qualified_scene_pool',state=f"{r['scene']}:{r['state']}") for r in qualified]
    drops=controller_drops(qualified)+controller_drops(pooled);menus={s:training_menu(s) for s in NAMES};restricted={};disclosures={}
    census=read(HERE/'CONTROLLER_LEVEL_FAILURES.json')
    failures={(r['scene'],r['layer'],r['controller']) for r in census if r['controller_failure_definite']}
    rates={s:weighted_rates([r for r in qualified if r['scene']==s]) for s in NAMES}
    rates['qualified_scene_pool']=weighted_rates(qualified)
    removed_rates={s:weighted_rates([r for r in qualified if r['scene']==s and (r['scene'],r['layer'],r['controller']) not in failures]) for s in NAMES}
    sensitivity,census_geometry=diagnosis_details(qualified)
    write(HERE/'GEOMETRY_CENSUS.json',census_geometry)
    write(HERE/'FAILURE_SENSITIVITY_SUMMARY.json',dict(baseline=sensitivity,all_methods_full_grid='DIAGNOSTICS/<rollout_uid>.json',source='Saved simulation arrays only; no new controller calls'))
    for scene in NAMES:
        choices=HERE/f'LEARNED_EVAL_CHOICES_{scene}.json'
        if choices.exists():
            keys={(r['N'],r['state']) for r in rows if r['scene']==scene and r['layer']=='distributed' and r['method']=='learned_scene'}
            subset=[r for r in rows if r['scene']==scene and (r['N'],r['state']) in keys]
            restricted[scene]=summarize(subset,[(l,m) for l in ('central','distributed') for m in FIXED],[scene])
            disclosures[scene]=dict(choices_sha256=sha(choices),choice_receipt_sha256=sha(HERE/f'LEARNED_EVAL_CHOICES_{scene}.receipt.json'),
                baseline_outcomes_already_known=True,reason='Required baseline stage precedes learning. No manual parameter tuning; models use only registered training labels and source-type nested validation.',
                copied_outcomes_opened_field_scope='The chooser does not open selected evaluation outcome files; it is not a claim that earlier baseline outcomes were globally blinded.')
    pool_keys={(r['scene'],r['N'],r['state']) for r in rows if r['layer']=='distributed' and r['method']=='learned_scene'}
    if pool_keys:
        subset=[r for r in rows if (r['scene'],r['N'],r['state']) in pool_keys]
        pool_scenes=sorted({k[0] for k in pool_keys})
        restricted['matched_evaluation_scene_pool']=summarize(subset,[(l,m) for l in ('central','distributed') for m in FIXED]+[('distributed','learned_scene')],pool_scenes)
    write(HERE/'MAIN_ADDITIONAL_METRICS.json',dict(utc=utc(),controller_drops=drops,weighted_outcome_and_diagnosis_rates=rates,per_layer_excluding_definite_controller_failure_rates=removed_rates,training_full_menu_oracles=menus,evaluation_fixed_reference_tables=restricted,choice_disclosures=disclosures,
        independent_full_menu_regret='N/A; requested evaluation does not exhaust twelve actions',
        standard_method_NA=dict(A3='Frozen old controller representation does not apply',A0_frozen='Frozen old physical input interface does not apply; L3 is separately retrained',G_E='No compatible frozen probe/selection interface',expert='No privileged expert benchmark requested',nominal_only_other_scenes='Only the new crowded scene background is requested')))
    seal('MAIN_ADDITIONAL_METRICS.receipt.json',[HERE/n for n in ('MAIN_ADDITIONAL_METRICS.json','GEOMETRY_CENSUS.json','FAILURE_SENSITIVITY_SUMMARY.json','derived_metrics.py')],new_rollouts=0,uses_saved_predictions_not_new_training=True)

if __name__=='__main__':main()
