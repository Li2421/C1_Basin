"""Physical content, context noise and local label conflicts, no new simulations."""
import numpy as np
from scipy.spatial.distance import cdist
from .audit import OUT,SRC,read,write,load,canonical_h,fit_predict,choose_alpha,evaluate

NAMES=['nominal_progress_integral','nominal_progress_late','nominal_goal_delta',
       'nominal_min_pair_center_distance','nominal_pair_closing_max','nominal_safety_mean',
       'nominal_safety_max','nominal_safety_active','nominal_final_speed','nominal_flow_change',
       'eta_progress_integral','eta_progress_late','eta_goal_delta','eta_min_pair_center_distance',
       'eta_pair_closing_max','eta_safety_mean','eta_safety_max','eta_safety_active',
       'eta_correction_mean','eta_correction_max','eta_correction_active',
       'eta_nominal_progress_difference','eta_nominal_final_position_distance',
       'eta_nominal_min_clearance_difference']


def main():
    d,x,rows,states,indices,tr,va,seeds,s,n,s4,n4=load()
    profiles=read(SRC/'protocol.json')['profiles']
    controller={p['sha256'] if p['path'] else None:i for i,p in enumerate(profiles)}
    lookup={(r['state_uid'],tuple(r['eta'])):i for i,r in enumerate(rows)}
    sd=np.full_like(d['context'],np.nan)
    for p in (SRC/'rich_response_cache').glob('*.json'):
        r=read(p)
        key=(r['state_uid'],tuple(r['eta']))
        if key not in lookup or r['alternate_sha'] not in controller:continue
        c,i=controller[r['alternate_sha']],lookup[key]
        np.testing.assert_allclose(r['features']['mean'],d['context'][c,i],rtol=1e-6,atol=1e-6)
        sd[c,i]=r['features']['probe_std']
    assert np.isfinite(sd).all()
    cx=d['context'][:,indices]
    ss=sd[:,indices]
    features=[]
    for k,name in enumerate(NAMES):
        a=cx[:,tr,:,k]
        variance=float(np.var(a,axis=1,ddof=1).mean())
        # For two roots, stored population std^2 is unbiased for Var(their mean).
        noise=float((ss[:,tr,:,k]**2).mean())
        features.append({'feature':name,'min':float(a.min()),'max':float(a.max()),
            'within_controller_eta_across_state_variance':variance,
            'two_root_mean_noise_variance_estimate':noise,
            'noise_to_observed_state_variance':noise/variance if variance>1e-14 else None,
            'fraction_above_0_01':float((np.abs(a)>.01).mean())})
    cache_result={'features':features,'all_1440_contexts_matched_to_frozen_matrix':True,
        'mean_noise_warning':'Only two fixed probe streams: rough variance diagnostic, not repeated independent experiments.',
        'semantics':{'H20_seconds':1.,'full_task_seconds':35.,
          'safety_channels':'safe minus Flow, measured along nominal or eta-conditioned trajectory; excludes final projection of safe+correction',
          'clearance_channels':'pair center distance, not pair surface gap, and no obstacle clearance channel',
          'flow_in_h':'One representative independent conditioning RNG draw; Ring flow_committed=False, not a committed future action.',
          'interpretation':'These are documented information limits, not evidence of mismatched execution or proof that omitted variables cause the prediction failures.'}}
    write('context_content_and_noise.json',cache_result)
    trajectories=[]
    for c in ('base','alt','second'):
        for r in read(OUT/f'micro_{c}.json')['rows']:
            tt=r['trajectory']
            nominal=[t['step'] for t in tt if t['nominal_safety_L2_over_speed']>=.01]
            first=tt[0]['goal_distance_mean'];at20=tt[19]['goal_distance_mean'];at60=tt[59]['goal_distance_mean'];last=tt[-1]['goal_distance_mean']
            trajectories.append({'state_uid':r['state_uid'],'eta_uid':r['eta_uid'],'controller':c,
                'nominal_material_safety_onset':nominal[0] if nominal else None,
                'initial_0_to_0_95s_progress_per_second':(first-at20)/.95,
                'last_2_95_to_3_95s_progress_per_second':at60-last,
                'H20_min_pair_surface_gap':min(t['min_pair_surface_gap'] for t in tt[:20]),
                'H80_min_obstacle_surface_gap':min(t['min_obstacle_surface_gap'] for t in tt),
                'initial_final_projection_deviation':tt[0]['corrected_projection_L2_over_speed']})
    initial=np.array([r['initial_0_to_0_95s_progress_per_second'] for r in trajectories])
    late=np.array([r['last_2_95_to_3_95s_progress_per_second'] for r in trajectories])
    write('micro_summary.json',{'truncated_probes':len(trajectories),'derived_simulation_steps':sum(read(OUT/f'micro_{c}.json')['physical_steps'] for c in ('base','alt','second')),
        'new_task_continuations':0,'new_success_labels':0,
        'nominal_material_safety_ever_active':sum(r['nominal_material_safety_onset'] is not None for r in trajectories),
        'median_initial_progress_mps':float(np.median(initial)),'median_late_progress_mps':float(np.median(late)),
        'late_progress_less_than_half_initial':int((late<initial/2).sum()),
        'corrected_projection_warning':'Immediate correction projection includes speed clipping; it is NOT evidence of obstacle/agent safety activation.',
        'rows':trajectories})
    # Closest source pair at same exact eta and same controller. No cross-semantics distance claim.
    h=canonical_h(x)
    hz=(h-h[tr].mean(0))/np.maximum(h[tr].std(0),.05)
    ch=(cx-cx[:,tr].mean((0,1,2)))/np.maximum(cx[:,tr].std((0,1,2)),.05)
    neighbors=[]
    for c in range(3):
        for j in va:
            for e in range(16):
                hd=((hz[tr]-hz[j])**2).mean(1)
                cd=((ch[c,tr,e]-ch[c,j,e])**2).mean(1)
                order=np.argsort(hd+cd)[:5]
                for rank,o in enumerate(order):
                    k=tr[o];qv=s[c,j,e]/max(n[c,j,e],1);qt=s4[c,k,e]/max(n4[c,k,e],1)
                    neighbors.append({'controller':c,'VAL_state':int(j),'TRAIN_state':int(k),'eta_index':e,'rank':rank+1,
                        'h_RMS':float(np.sqrt(hd[o])),'C_RMS':float(np.sqrt(cd[o])),
                        'Q_VAL':float(qv),'Q_TRAIN_first4':float(qt),'absolute_Q_gap':float(abs(qv-qt))})
    first=[r for r in neighbors if r['rank']==1]
    write('nearest_input_audit.json',{'exact_or_near_exact_aliases_at_1e_6':sum(r['h_RMS']<=1e-6 and r['C_RMS']<=1e-6 for r in first),
        'nearest_h_RMS_quantiles':np.quantile([r['h_RMS'] for r in first],[0,.25,.5,.75,1]).tolist(),
        'nearest_C_RMS_quantiles':np.quantile([r['C_RMS'] for r in first],[0,.25,.5,.75,1]).tolist(),
        'nearest_Q_gap_at_least_half':sum(r['absolute_Q_gap']>=.5 for r in first),'pairs':len(first),
        'caveat':'TRAIN mostly Q4; separated inputs with discordant outcomes do not establish representation aliasing or irreducible ambiguity.','neighbors':neighbors})


if __name__=='__main__':main()
