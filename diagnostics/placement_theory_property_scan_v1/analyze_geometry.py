"""Classify boundary smoothness scale from prior radius-swept curvature probes."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
STRONG=HERE.parent/'placement_strong_hypotheses_v1'

def main():
    p=json.loads((STRONG/'geometry_summary.json').read_text())
    rows=[]
    for r in p['curvature']:
        k=r['k_median']
        stable=bool(r['all_radii_branch_stable'] and r['max_relative_fit_rmse']<.05
                    and r['radius_variation']<.05)
        rows.append(dict(uid=r['uid'],chain=r['chain'],origin=r['origin'],
             stable_moderate_radius_fit=stable,k_median=k,
             dimensionless_normal_rotation_at_radius001=max(abs(z) for z in k)*.01,
             relative_fit_rmse=r['max_relative_fit_rmse'],radius_variation=r['radius_variation'],
             active_branch_stable=r['all_radii_branch_stable']))
    own=[]
    for r in p['orientation']:
        rr=[x for x in rows if x['uid']==r['uid']]
        own.append(dict(uid=r['uid'],normal_angle_deg=r['own_boundary_angle_deg'],
            both_curvature_regular=all(x['stable_moderate_radius_fit'] for x in rr),
            TT=next(x for x in rr if x['chain']=='TT'),FF=next(x for x in rr if x['chain']=='FF')))
    out=dict(points=len(rows),stable_curvature_points=sum(r['stable_moderate_radius_fit'] for r in rows),
        stable_pairs=sum(r['both_curvature_regular'] for r in own),pairs=own,
        interpretation='At eta radius .01 even stable |kappa|r is small (locally nearly planar), while TT/FF fitted curvatures differ. No global C2 claim.')
    (HERE/'local_geometry_properties.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(points=out['points'],stable_curvature_points=out['stable_curvature_points'],
        stable_pairs=out['stable_pairs']),indent=2))

if __name__=='__main__':main()
