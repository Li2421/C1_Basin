"""Combine independent Ring normal and curvature evidence with exclusions."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=ROOT/'diagnostics/placement_boundary_mechanism_audit_v1'


def angle(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return float(np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))


def main():
    old=json.loads((HERE/'curvature_analysis.json').read_text())
    new=json.loads((HERE/'curvature_followup_analysis.json').read_text())
    gradients=json.loads((HERE/'h1_gradient_analysis.json').read_text())['rows']
    orientation=[]
    for x in old['orientation']:
        orientation.append(dict(uid=x['uid'],own_boundary_angle_deg=x['orientation_angle_deg'],
            TT_scale_stability=x['TT_gradient_stability'],FF_scale_stability=x['FF_gradient_stability'],
            origin='prior_preselected'))
    for x in new['orientation']:
        uid=x['uid'];entry=dict(uid=uid,own_boundary_angle_deg=x['own_boundary_normal_angle_deg'],
            TT_scale_stability=x['TT_gradient_stability'],FF_scale_stability=x['FF_gradient_stability'],
            origin='regularity_filtered_followup',same_eta_angles={})
        for root in ('TT','FF'):
            a=next(y for y in gradients if y['uid']==uid and y['root_chain']==root and y['eval_chain']=='TT')
            b=next(y for y in gradients if y['uid']==uid and y['root_chain']==root and y['eval_chain']=='FF')
            entry['same_eta_angles'][root]=angle(a['estimates']['0.0005']['rho_gradient'],b['estimates']['0.0005']['rho_gradient'])
        orientation.append(entry)
    curvature=[]
    dominant_direction=[]
    for source,rows in (('prior_preselected',old['centers']),('regularity_filtered_followup',new['rows'])):
        for uid in sorted({r['uid'] for r in rows}):
            direction=[]
            for chain in ('TT','FF'):
                rr=[r for r in rows if r['uid']==uid and r['chain']==chain]
                r01=next(r for r in rr if r['radius']==.01)
                k01=np.asarray(r01['fit']['principal_curvatures'])
                index=int(np.argmax(np.abs(k01)))
                direction.append(np.asarray(r01['fit']['principal_directions'][index]))
                moderate=[r for r in rr if r['radius'] in (.005,.01) and r['branch_stable']]
                k=np.array([r['fit']['principal_curvatures'] for r in moderate])
                curvature.append(dict(uid=uid,chain=chain,origin=source,
                    radii=[r['radius'] for r in moderate],k_at_moderate_radii=k.tolist(),
                    k_median=np.median(k,axis=0).tolist() if len(k) else None,
                    max_relative_fit_rmse=max((r['fit']['fit_relative_rmse'] for r in moderate),default=None),
                    radius_variation=np.max(np.ptp(k,axis=0)) if len(k)>1 else None,
                    all_radii_branch_stable=all(r['branch_stable'] for r in rr)))
            dominant_direction.append(dict(uid=uid,origin=source,radius=.01,
                sign_invariant_angle_deg=float(np.degrees(np.arccos(np.clip(abs(direction[0]@direction[1]),0,1)))),
                caution='Dominant direction is unreliable if curvature eigenvalues are near zero or radius-unstable'))
    out=dict(orientation=orientation,curvature=curvature,dominant_curvature_direction=dominant_direction,
        caution='Curvature is claimed only where 0.005 and 0.01 radii agree with low fit residual and stable active factor; post-hoc regularity filtering is explicit.')
    (HERE/'geometry_summary.json').write_text(json.dumps(out,indent=2,default=float)+'\n')
    print(json.dumps(orientation,indent=2))


if __name__=='__main__':main()
