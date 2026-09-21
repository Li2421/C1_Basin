"""Instrument frozen LocalRisk subproblems without changing the scorer."""
import json
import numpy as np
from audit_c1_joint_witness_risk import ROOT,save
import audit_c1_joint_witness_risk as risk
from single_integrator.environment import Config,GiveWayEnv
from single_integrator.cbf import CBFConfig,barrier_constraints

BASE=ROOT/'results/c1_candidate_coverage_expansion'
OUT=ROOT/'results/c1_geometry_solver_safety_audit'
Original=risk.Projection
LOG=[]

class Instrumented(Original):
    def __call__(self,y):
        error=None
        try:return super().__call__(y)
        except Exception as e:error=str(e);raise
        finally:
            sol=self.solver.get_solution();info=self.solver.get_info();p=np.asarray(sol.x)
            norms=np.linalg.norm(self.A,axis=1)
            entry=dict(kind='physical' if self.speeds is not None else 'base',call=self.calls,
                target=np.asarray(y).tolist(),target_min_linear_slack=float(np.min(self.A@y-self.b)),
                status=str(sol.status),iterations=sol.iterations,solution=p.tolist(),solution_norm=float(np.linalg.norm(p)),
                min_slack=float(np.min(self.A@p-self.b)),error=error,
                primal_residual=sol.r_prim,dual_residual=sol.r_dual,gap_abs=info.gap_abs,
                step_length=info.step_length,zero_center_inscribed_ball=float(np.min(-self.b/norms)),
                max_speed=float(np.linalg.norm(p.reshape(-1,2),axis=1).max()),
                stationarity=float(np.max(np.abs(p-y+self.conic_A.T@np.asarray(sol.z)))))
            LOG.append(entry)

def main():
    OUT.mkdir(exist_ok=True);env=GiveWayEnv(Config(corridor_half_length=1.3));cfg=CBFConfig();rows=[]
    for r in json.loads((BASE/'rows.json').read_text()):
        if not r.get('geometry_errors'):continue
        z=np.load(BASE/'traces'/f"{r['rid']:04d}_{r['cid']}.npz")
        for e in r['geometry_errors']:
            t=e['step'];x=z['positions_before'][t];v=z['candidate'][t]
            A,b,h=barrier_constraints(dict(positions=x,walls=env.walls,config=env.config.to_dict()),cfg)
            LOG.clear();risk.Projection=Instrumented
            try:result=risk.LocalRisk(A,b,np.full(2,.5)).score(v);error=None
            except Exception as ex:result=None;error=str(ex)
            finally:risk.Projection=Original
            row=dict(rid=r['rid'],cid=r['cid'],step=t,x=x.tolist(),v=v.tolist(),A=A.tolist(),b=b.tolist(),
                pair_h=h['pairwise_h'],min_wall_h=h['min_wall_h'],error=error,result=result,calls=list(LOG))
            rows.append(row)
            print(r['cid'],t,[(d['kind'],d['call'],d['target'],d['status'],d['target_min_linear_slack'],d['zero_center_inscribed_ball'],d['gap_abs']) for d in LOG if d['error']],flush=True)
    save(OUT/'original_failure_details.json',rows)

if __name__=='__main__':main()
