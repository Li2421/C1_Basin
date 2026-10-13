"""Cache-only function-space and eta-signal coverage, source data only."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def run():
    d = np.load(ROOT/'data/source_augmented128/dataset.npz')
    assert set(d['split']) <= {'train','validation'}
    f = d['field'].reshape(-1,4,7,3,5,2)
    jac = np.stack((f[...,1,:]-f[...,2,:], f[...,3,:]-f[...,4,:]),-1)
    isotropic = np.trace(jac,axis1=-2,axis2=-1)[...,None,None]/2*np.eye(2)
    residual = np.linalg.norm(jac-isotropic,axis=(-2,-1))
    norm = np.linalg.norm(jac,axis=(-2,-1))
    field_rows = []
    for scene in sorted(set(d['scene'])):
        for family in ('analytic','flow'):
            ix = (d['scene']==scene) & (np.char.startswith(d['controller'],'analytic') if family=='analytic' else d['controller']=='flow')
            mask = d['entity_agent_mask'][ix].astype(bool)
            rr, nn = residual[ix][mask], norm[ix][mask]
            vals = rr/np.maximum(nn,1e-12)
            field_rows.append(dict(scene=str(scene), family=family, state_controller_rows=int(ix.sum()),
                mean_relative_anisotropy=float(vals.mean()), max_relative_anisotropy=float(vals.max()),
                median_relative_anisotropy=float(np.median(vals)),
                maximum_absolute_isotropic_residual=float(rr.max()),
                interpretation='Finite differences along common per-agent goal/lateral action directions; not the full joint-action Jacobian for native Flow'))
    eta_rows = []
    for scene in sorted(set(d['scene'])):
        for split in ('train','validation'):
            for ctl in sorted(set(d['controller'])):
                ix = (d['scene']==scene)&(d['split']==split)&(d['controller']==ctl)
                if not ix.any():continue
                s,f,u = d['success'][ix],d['failure'][ix],d['unknown'][ix]
                denominator=s+f+u
                lo=s/denominator;hi=(s+u)/denominator
                complete=(u==0).all(-1)
                q=s/np.maximum(s+f,1)
                fixed=int(lo.mean(0).argmax())
                eta_rows.append(dict(scene=str(scene),split=split,controller=str(ctl),
                    physical_states=int(ix.sum()),complete_state_panels=int(complete.sum()),
                    complete_panels_with_no_observed_eta_variation=int(((np.ptp(q,-1)<1e-12)&complete).sum()),
                    certified_eta_range_ge_3_over_8_states=int(((lo.max(-1)-hi.min(-1))>=.375).sum()),
                    observed_verified_oracle_minus_best_fixed=float(lo.max(-1).mean()-lo.mean(0).max()),
                    aggregate_verified_best_fixed_eta_index=fixed,
                    caution='Source descriptive empirical labels; Q1 TRAIN oracle/ranges may include seed noise, not a fresh generalization estimate'))
    output=dict(field=field_rows,eta=eta_rows,new_rollouts=0,final_states_read=False,
        literature_motivation='PD-VF/PAnDR cross policies with environments; examine which controller-function variations our crossed data actually cover.',
        exact_family_property='AnalyticField velocity in physical action coordinates is lambda*(target(s)-h); d(field)/d(h)=-lambda*I for frozen physical state. The family varies target behavior in five parameters but action-space linear response in only one isotropic parameter.',
        boundary='Safety projection, state-dependent eta lifting, and closed-loop future states can still generate nonlinear controller-by-eta interactions. This structural observation does not imply zero rollout interaction or establish the root cause of cross-scene error.')
    (ROOT/'source_family_diagnostics.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(field_rows))


if __name__=='__main__':run()
