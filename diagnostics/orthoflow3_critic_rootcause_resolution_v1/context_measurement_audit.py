"""How much of the short-response state fingerprint varies across probe RNG?

The stored two-root standard deviations permit a measurement-noise diagnostic,
not a proof of feature insufficiency: fixed roots are common across states.
No task labels or target scenes are read here.
"""
import json
from pathlib import Path
import numpy as np
from .replication_analysis import SRC,OUT,read,write,csvwrite

def main():
    states=read(SRC/'source_states.json');state_index={s['uid']:i for i,s in enumerate(states)}
    profiles=read(SRC.parent/'orthoflow3_source_contrast_interaction_v1/protocol.json')['profiles']
    controllers={c['sha256']:i for i,c in enumerate(profiles)}
    d=np.load(SRC/'source_data_db.npz');eta_index={tuple(x):i for i,x in enumerate(d['eta'][:16].astype(float))}
    std=np.full((2,46,16,24),np.nan);matched=[]
    for directory in ('orthoflow3_controller_state_residual_factorial_v1','orthoflow3_controller_training_repair_v1','orthoflow3_source_contrast_interaction_v1'):
      for path in (SRC.parent/directory/'rich_response_cache').glob('*.json'):
        r=read(path)
        if not r.get('valid') or r['state_uid'] not in state_index or r['alternate_sha'] not in controllers:continue
        f=r['features']
        if f['steps']!=20:continue
        ei=eta_index.get(tuple(np.asarray(r['eta'],np.float32).astype(float)))
        if ei is None:continue
        ci=controllers[r['alternate_sha']];si=state_index[r['state_uid']]
        if not np.allclose(f['mean'],d['context'][ci,si*16+ei],atol=1e-6,rtol=0):continue
        value=np.array(f['probe_std'])
        if np.isfinite(std[ci,si,ei]).all():np.testing.assert_allclose(value,std[ci,si,ei],atol=1e-10,rtol=0)
        std[ci,si,ei]=value;matched.append(str(path))
    mean=d['context'].reshape(2,46,16,24)
    names=['nominal_progress_integral','nominal_progress_late','nominal_goal_delta','nominal_min_pair_distance','nominal_pair_closing_max','nominal_safety_mean','nominal_safety_max','nominal_safety_active','nominal_final_speed','nominal_flow_change','eta_progress_integral','eta_progress_late','eta_goal_delta','eta_min_pair_distance','eta_pair_closing_max','eta_safety_mean','eta_safety_max','eta_safety_active','eta_correction_mean','eta_correction_max','eta_correction_active','eta_nominal_progress_difference','eta_nominal_final_position_distance','eta_nominal_min_pair_distance_difference']
    rows=[]
    for ci in range(2):
     for ei in (10,15):
      available=np.isfinite(std[ci,:,ei]).all(-1)
      for j,name in enumerate(names):
        a=mean[ci,available,ei,j];s=std[ci,available,ei,j]
        var=float(np.var(a,ddof=1));noise=float(np.mean(s*s))
        rows.append(dict(controller=ci,eta_index=ei,feature=name,states=int(available.sum()),
            between_state_context_variance=var,estimated_variance_of_two_root_mean=noise,
            noise_to_observed_state_variance=noise/var if var>1e-12 else None,
            reliability_heuristic=1-noise/var if var>1e-12 else None))
    csvwrite(OUT/'context_probe_variability.csv',rows)
    valid=[r for r in rows if r['noise_to_observed_state_variance'] is not None]
    summary=dict(matched_cache_entries=int(np.isfinite(std).all(-1).sum()),expected=2*46*16,
        varying_controller_eta_channels=len(valid),channels_noise_ratio_above_one=sum(r['noise_to_observed_state_variance']>1 for r in valid),
        median_noise_ratio=float(np.median([r['noise_to_observed_state_variance'] for r in valid])),
        note='Heuristic only. Two roots are shared, not independently randomized across states; high ratio indicates unstable Monte-Carlo state descriptor, not proven label or implementation bug.',
        task_rollouts=0,target_labels_used=False)
    write(OUT/'context_probe_variability_summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
