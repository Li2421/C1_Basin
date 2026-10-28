"""Independently verify saved diagnostic witnesses with actual production rows."""
from core import *
from bottleneck_family.scenario import segment_distance
from single_integrator.environment import point_segment_distance
import inspect

OUT=HERE/'zero_coverage_audit'


def main():
    scenes={(e['type_id'],e['level']):e['scene']for e in read(HERE/'SCENES.json')['entries']};rows=[]
    for source in sorted((OUT/'clearance_variant').glob('t08_l*_s[0-9]_cbf_scheduled.json')):
        r=read(source)
        if not r['within_original_horizon']:continue
        s=scenes[(r['type_id'],r['level'])];state=s['states'][r['state_index']];cfg=s['config']
        z=np.load(r['trace']);p=z['positions'];u=z['actions'];env=RN.SwapEnv(state,RN.LiteratureConfig(**cfg));snap=env.snapshot()
        assert np.array_equal(env.walls,np.asarray(s['walls']))
        residual=np.inf
        for t in range(len(u)):
            snap['positions']=p[t].copy()
            snap['velocities']=np.zeros_like(u[t])if t==0 else u[t-1].copy()
            for i in range(len(p[t])):
                f=RN.SwapSafety(snap,i);residual=min(residual,float(np.min(f.A@u[t,i]-f.lower,initial=np.inf)))
        ii,jj=np.triu_indices(len(p[0]),1)
        before=p[:-1,ii]-p[:-1,jj];after=p[1:,ii]-p[1:,jj]
        pair=float(point_segment_distance(np.zeros(2),before,after).min()-2*cfg['agent_radius']-cfg['agent_collision_margin'])
        wall=float(segment_distance(p[:-1,:,None],p[1:,:,None],env.walls[None,None,:,0],env.walls[None,None,:,1]).min()-env.instance.geometry.margin)
        dynamics=float(np.max(abs(np.diff(p,axis=0)-cfg['dt']*u),initial=0.))
        speed=float(np.linalg.norm(u,axis=-1).max());dist=np.linalg.norm(p-np.asarray(state['goals'])[None],axis=-1)
        successes=np.flatnonzero(np.all(dist<=cfg['goal_tolerance'],axis=1));first=int(successes[0])
        assert np.array_equal(p[0],np.asarray(state['positions']))
        assert residual>=-1e-10 and pair>0 and wall>0 and dynamics<1e-12
        assert speed<=cfg['max_speed']+1e-12 and first<=cfg['max_steps']
        row=dict(type_id=r['type_id'],level=r['level'],state_index=r['state_index'],scene=s['name'],
            original_row_factory=inspect.getfile(RN.SwapSafety),row_factory_sha256=sha(inspect.getfile(RN.SwapSafety)),
            trace=r['trace'],trace_sha256=sha(r['trace']),source=str(source),
            steps_checked=len(u),original_CBF_minimum_residual=residual,minimum_swept_pair_clearance=pair,
            minimum_swept_wall_clearance=wall,dynamics_maximum_residual=dynamics,maximum_speed=speed,
            first_original_success_predicate_step=first,first_original_success_predicate_seconds=first*cfg['dt'],
            original_horizon_seconds=cfg['max_steps']*cfg['dt'],passed=True,environment_steps=0,solver_calls=0,
            interpretation='Numerical constructive joint-feasibility witness: actual production CBF rows, identical swept-geometry formulas/dynamics/goal predicate. No nominal-controller or safety-solver execution, no Oracle success label.')
        rows.append(row);print('INDEPENDENT_ORIGINAL_ROW_CHECK',r['level'],r['state_index'],first*cfg['dt'],residual,pair,wall,flush=True)
    dump(OUT/'ORIGINAL_ROW_VERIFICATION.json',dict(cases=rows,complete=True,passed=bool(rows)and all(r['passed']for r in rows),
        environment_steps=0,solver_calls=0,frozen_scenes_sha256=sha(HERE/'SCENES.json')))


if __name__=='__main__':main()
