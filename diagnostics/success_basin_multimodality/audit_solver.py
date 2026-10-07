"""Reconstruct every indexed SBGA projection failure and solve the identical SOCP."""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

SYSROOT=Path('/home/zhihan/research/02_C1_Toy_GiveWay')
sys.path.insert(0,str(SYSROOT))
import jax
import jax.numpy as jnp
from single_integrator.cbf import CBFConfig,CBFSolverError,barrier_constraints,project_velocity
from single_integrator.environment import Config,bounded_nominal
from single_integrator.evaluate import load_policy
from diagnostics.astra_true_q_audit.audit import restore
from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector,DiagnosticPhi
from diagnostics.success_basin_multimodality.legacy_solver import project_velocity_legacy
from diagnostics.success_basin_multimodality.setup import ROOT,HERE,OLD,sha,write

def main():
    jax.config.update('jax_enable_x64',True);jax.config.update('jax_platform_name','cpu')
    p=json.loads((HERE/'protocol.json').read_text());op=json.loads((OLD/'protocol.json').read_text())
    cat={s['state_id']:s for s in op['state_catalog']};cfg=Config(**p['environment']);cbf=CBFConfig()
    policy,_=load_policy(Path(p['checkpoint']))
    failures=[]
    for manifest in sorted((OLD/'raw').glob('*/manifest.json')):
        for r in json.loads(manifest.read_text())['records']:
            if r.get('outcome') is None: failures.append(r)
    assert len(failures)==754,len(failures)
    rows=[]
    for index,r in enumerate(failures):
        entry=cat[r['state_id']]
        with np.load(OLD/entry['state_file']) as state: env=restore(dict(state),cfg)
        with np.load(OLD/r['file']) as d:
            max_position_error=0.
            for k,u in enumerate(d['u_exec']):
                max_position_error=max(max_position_error,float(np.max(abs(env.positions-d['positions_before'][k]))))
                _,_,done,info=env.step(u)
                assert not done and info['termination']=='running'
                max_position_error=max(max_position_error,float(np.max(abs(env.positions-d['positions_after'][k]))))
        assert env.step_count==r['execution_error']['absolute_step']
        obs=env.observation();key0=jax.random.fold_in(jax.random.PRNGKey(r['seed']),entry['pair_id'])
        key=jax.random.fold_in(key0,env.step_count)
        flow=bounded_nominal(np.asarray(policy.sample_actions(obs[None],seed=key)[0]),cfg.max_speed)
        A,b,_=barrier_constraints(env.snapshot(),cbf);which=None;target=None;old_error=None
        try:
            safe,_=project_velocity_legacy(flow,A,b,cfg.max_speed,cbf)
        except CBFSolverError as exc:
            which='first';target=flow;old_error=exc
        if old_error is None:
            corrector=DiagnosticCorrector(DiagnosticPhi(*r['phi']))
            g=corrector(obs,safe,cfg.max_speed);target=safe+g
            try: project_velocity_legacy(target,A,b,cfg.max_speed,cbf)
            except CBFSolverError as exc: which='second';old_error=exc
        logged=r['execution_error']['message'].split(':',1)[0]
        reproduced=old_error is not None and old_error.status==logged
        diagnostics=[];robust_error=None
        try:
            u,status=project_velocity(target,A,b,cfg.max_speed,cbf,
                                      diagnostic_callback=diagnostics.append,
                                      diagnostic_context={'prior_failure_index':index})
        except Exception as exc:
            robust_error=f'{type(exc).__name__}: {exc}';u=None;status=None
        if u is not None:
            flat=np.asarray(u).reshape(4);minres=float(np.min(A@flat-b));maxspeed=float(np.linalg.norm(flat.reshape(2,2),axis=-1).max())
            classification='NUMERICAL_SOLVER_FAILURE'
        else:
            minres=maxspeed=None;classification='OTHER_IMPLEMENTATION_FAILURE'
        rows.append({'prior_file':r['file'],'state_id':r['state_id'],'eta':r['phi'],'seed':r['seed'],
                     'absolute_step':env.step_count,'legacy_logged_status':logged,
                     'legacy_reproduced_status':None if old_error is None else old_error.status,
                     'legacy_failure_reproduced':reproduced,'failed_projection':which,
                     'robust_status':status,'robust_error':robust_error,
                     'robust_min_linear_residual':minres,'robust_max_agent_speed':maxspeed,
                     'robust_diagnostic':diagnostics[-1] if diagnostics else None,
                     'prefix_max_position_replay_error':max_position_error,'classification':classification})
        if (index+1)%100==0: print(json.dumps({'audited':index+1,'total':len(failures)}),flush=True)
    counts=Counter(x['classification'] for x in rows);calls=Counter(x['failed_projection'] for x in rows)
    summary={'indexed_prior_unknown':len(rows),'classification_counts':dict(counts),
             'failed_projection_counts':dict(calls),
             'legacy_status_counts':dict(Counter(x['legacy_logged_status'] for x in rows)),
             'all_legacy_failures_reproduced':all(x['legacy_failure_reproduced'] for x in rows),
             'all_identical_socps_solved_and_certified':all(x['robust_error'] is None for x in rows),
             'minimum_robust_linear_residual':min(x['robust_min_linear_residual'] for x in rows if x['robust_min_linear_residual'] is not None),
             'maximum_robust_agent_speed':max(x['robust_max_agent_speed'] for x in rows if x['robust_max_agent_speed'] is not None),
             'maximum_prefix_replay_error':max(x['prefix_max_position_replay_error'] for x in rows),
             'mathematical_problem_unchanged':p['solver']['problem'],
             'rows':rows}
    write('projection_solver_audit.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
