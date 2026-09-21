"""Independent continuous-step geometry and CBF audit of all expanded trajectories."""
import json
import numpy as np
from audit_c1_joint_witness_risk import ROOT,save
from single_integrator.environment import Config,GiveWayEnv,segment_distance
from single_integrator.cbf import CBFConfig,barrier_constraints

BASE=ROOT/'results/c1_candidate_coverage_expansion'
OUT=ROOT/'results/c1_geometry_solver_safety_audit'

def main():
    OUT.mkdir(exist_ok=True);plant=Config(corridor_half_length=1.3);cfg=CBFConfig();env=GiveWayEnv(plant)
    walls=env.walls;seg=walls[:,1]-walls[:,0];r_safe=2*plant.agent_radius+plant.agent_collision_margin+cfg.separation_buffer
    wall_safe=plant.agent_radius+plant.wall_radius+plant.wall_collision_margin+cfg.separation_buffer
    reports=[]
    for meta in json.loads((BASE/'rows.json').read_text()):
        rid,cid=meta['rid'],meta['cid'];z=np.load(BASE/'traces'/f'{rid:04d}_{cid}.npz')
        x=z['positions_before'];y=z['positions_after'];u=z['applied'].reshape(-1,2,2)
        rel=x[:,0]-x[:,1];delta=(y[:,0]-y[:,1])-rel
        t=np.clip(-np.sum(rel*delta,axis=1)/np.maximum(np.sum(delta*delta,axis=1),1e-300),0,1)
        pair_dist=np.linalg.norm(rel+t[:,None]*delta,axis=1)
        wall_dist=segment_distance(x[:,:,None,:],y[:,:,None,:],walls[None,None,:,0,:],walls[None,None,:,1,:])
        foot=np.clip(np.sum((x[:,:,None,:]-walls[None,None,:,0,:])*seg,axis=-1)/np.sum(seg*seg,axis=1),0,1)
        normal=x[:,:,None,:]-(walls[None,None,:,0,:]+foot[:,:,:,None]*seg)
        distance=np.linalg.norm(normal,axis=-1);normal/=distance[:,:,:,None]
        pair_h=np.sum(rel*rel,axis=1)-r_safe**2;wall_h=distance-wall_safe
        pair_res=2*np.sum(rel*(u[:,0]-u[:,1]),axis=1)+cfg.gamma*pair_h
        wall_res=np.sum(normal*u[:,:,None,:],axis=-1)+cfg.gamma_wall*wall_h
        residual=np.concatenate([pair_res[:,None],wall_res.reshape(len(x),-1)],axis=1)
        # Cross-check independent vectorized expressions against authoritative rows.
        for k in [0,len(x)//2,len(x)-1]:
            A,b,_=barrier_constraints(dict(positions=x[k],walls=walls,config=plant.to_dict()),cfg)
            np.testing.assert_allclose(residual[k],A@u[k].reshape(4)-b,atol=1e-14,rtol=1e-12)
        endpoint=np.concatenate([x,y]);w=plant.corridor_width/2
        corridor=(np.abs(endpoint[:,:,0])<=plant.corridor_half_length)&(np.abs(endpoint[:,:,1])<=w)
        bay=(np.abs(endpoint[:,:,0])<=w)&(endpoint[:,:,1]>=w)&(endpoint[:,:,1]<=plant.bay_top)
        pair_clear=pair_dist-2*plant.agent_radius;wall_clear=wall_dist-plant.agent_radius-plant.wall_radius
        row=dict(rid=rid,cid=cid,steps=len(x),min_center_separation=float(pair_dist.min()),min_agent_surface_clearance=float(pair_clear.min()),
            min_wall_centerline_distance=float(wall_dist.min()),min_wall_surface_clearance=float(wall_clear.min()),
            min_pair_cbf_residual=float(pair_res.min()),min_wall_cbf_residual=float(wall_res.min()),
            min_pair_h=float(pair_h.min()),min_wall_h=float(wall_h.min()),
            max_speed=float(np.linalg.norm(u,axis=2).max()),max_integration_error=float(np.max(np.abs(y-x-plant.dt*u))),
            agent_collision_steps=int(np.sum(pair_clear<=plant.agent_collision_margin)),
            wall_collision_steps=int(np.sum(np.any(wall_clear<=plant.wall_collision_margin,axis=(1,2)))),
            outside_endpoint_agents=int(np.sum(~(corridor|bay))),
            cbf_violations_over_tolerance=int(np.sum(residual < -cfg.feasibility_tol)),
            negative_cbf_residual_entries=int(np.sum(residual<0)),
            speed_violations=int(np.sum(np.linalg.norm(u,axis=2)>plant.max_speed+cfg.speed_tol)),
            min_separation_step=int(np.argmin(pair_dist)),min_wall_step=int(np.unravel_index(np.argmin(wall_clear),wall_clear.shape)[0]))
        assert row['agent_collision_steps']+row['wall_collision_steps']==int(np.sum(z['collision']))
        reports.append(row)
    summary=dict(branches=len(reports),steps=sum(r['steps'] for r in reports),
        **{k:min(r[k] for r in reports) for k in ['min_center_separation','min_agent_surface_clearance','min_wall_centerline_distance','min_wall_surface_clearance','min_pair_cbf_residual','min_wall_cbf_residual','min_pair_h','min_wall_h']},
        **{k:max(r[k] for r in reports) for k in ['max_speed','max_integration_error']},
        **{k:sum(r[k] for r in reports) for k in ['agent_collision_steps','wall_collision_steps','outside_endpoint_agents','cbf_violations_over_tolerance','negative_cbf_residual_entries','speed_violations']},
        tolerances=cfg.to_dict(),plant=plant.to_dict(),method='Exact swept pair relative segment and swept center-to-wall-segment distances; applied controls and independent vectorized CBF rows, not candidate controls.')
    save(OUT/'safety_rows.json',reports);save(OUT/'safety_summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
