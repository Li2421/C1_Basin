"""Reconstruct one physical collision from immutable saved arrays; no simulation."""
import bootstrap
from experiment import *
import numpy as np

def main():
    sources=read(HERE/'MOVING_CAPABILITY_COLLISION_AUDIT.json')['sources']
    for source in sources:
        raw=read(source['path']);assert sha(source['path'])==source['sha256']
        assert sha(raw['trace_path'])==raw['trace_sha256']
        with np.load(raw['trace_path']) as z:
            T=len(z['actions']);t=T-1;pieces=json.loads(str(z['moving_pieces_json']))[t]
            if len(pieces)!=1:continue
            p=z['positions'][t,0];q=z['moving_positions'][t];u=z['actions'][t,0];v=(z['moving_positions'][t+1]-q)/.05;d=p-q;n=d/np.linalg.norm(d);R=.3+.05+.0125+.0001;h=d@d-R*R
            geom=json.loads(str(z['first_geometry_json']))[t][0];k=next(i for i,label in enumerate(geom['labels']) if label[0]=='moving');A=np.asarray(geom['A'][k]);lower=geom['lower'][k];residual=float(A@u-lower)
            a=d;b=d+.05*(u-v);delta=b-a;fraction=float(np.clip(-(a@delta)/max(delta@delta,1e-30),0,1));minimum=float(np.linalg.norm(a+fraction*delta));enddist=float(np.linalg.norm(b))
            assert minimum<=.35 and abs(minimum-.35-raw['minimum_swept_clearance'])<1e-9
            out=dict(utc=utc(),source=source,trace_path=raw['trace_path'],trace_sha256=raw['trace_sha256'],controller=raw['case']['controller_payload']['nominal']['controller'],state=raw['case']['state']['index'],solo_agent=raw['case']['state']['agent'],seed=raw['case']['seed'],eta_a=raw['case']['eta_a'],eta_b=raw['case']['eta_b'],step_index_zero_based=t,time_seconds=t*.05,
                position=p,moving_position=q,executed_filtered_velocity=u,obstacle_velocity_for_posthoc_explanation_only=v,control_period=.05,physical_separation=.35,inflated_separation=R,distance_at_step_start=float(np.linalg.norm(d)),distance_at_step_end=enddist,minimum_distance_during_step=minimum,minimum_time_offset=fraction*.05,
                inflated_h_at_start=float(h),moving_constraint_residual=residual,moving_constraint_satisfied=bool(residual>=-1e-9),filtered_robot_radial_away_speed=float(n@u),obstacle_radial_toward_robot_speed=float(n@v),assumed_stationary_obstacle_barrier_expression=float(2*d@u+h),actual_moving_obstacle_barrier_expression=float(2*d@(u-v)+h),projection_rewrite=bool(z['projection_rewrite'][t,0]),second_projection_change_from_first=float(np.linalg.norm(z['safe'][t,0]-u)),constraint_violation_steps=raw['constraint_violation_steps'],restoration_steps=raw['restoration_steps'],new_rollouts=0)
            write(HERE/'MOVING_COLLISION_EXPLANATION.json',out)
            print({k:v for k,v in clean(out).items() if k not in ('source','trace_path','trace_sha256','position','moving_position','eta_a','eta_b')},flush=True)
            return
    raise RuntimeError('No single-waypoint segment collision example found')
if __name__=='__main__':main()
