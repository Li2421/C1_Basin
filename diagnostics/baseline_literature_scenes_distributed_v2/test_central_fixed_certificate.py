"""Check the already-proven search certificate on saved slow fixed trajectories."""
from experiment import *
import numpy as np
import time

def main():
    from registry import verify_receipt
    verify_receipt('RESERVATION_V2_EQUIVALENCE.receipt.json')
    profile=read(HERE/'CENTRAL_CIRCLE_COST_PROFILE.json');path=Path(profile['source']);assert sha(path)==profile['source_sha256']
    raw=read(path);assert sha(raw['trace_path'])==raw['trace_sha256'];case=raw['case']
    if sys.argv[1:]==['snapshot']:
        import jax
        from states import LiteratureConfig
        from flow_scene import LiteratureEnv
        from reservation_certificate_v2 import ReservationNominal
        from controllers import bound
        t=1300
        with np.load(raw['trace_path']) as z:
            state=dict(case['state'],positions=z['positions'][t].tolist(),velocities=z['velocities'][t].tolist());expected=z['nominal'][t].copy()
        env=LiteratureEnv(state,LiteratureConfig(**case['scenario_payload']['config']));actor=ReservationNominal(case['controller_payload']['nominal'])
        started=time.monotonic();actual=bound(actor.action(env,jax.random.PRNGKey(0)),env.config.max_speed).astype(np.float64)
        actual[np.linalg.norm(env.goals-env.positions,axis=-1)<=env.config.goal_tolerance]=0
        result=dict(utc=utc(),source=str(path),source_sha256=sha(path),saved_step=t,saved_original_step_seconds=profile['slowest_steps'][0]['seconds'],seconds=time.monotonic()-started,passed=actual.dtype==expected.dtype and actual.shape==expected.shape and actual.tobytes()==expected.tobytes(),certificates=actor.planner.certificates,new_physical_steps=0,new_research_rollouts=0)
        write(HERE/'CENTRAL_FIXED_CERTIFICATE_SNAPSHOT_TEST.json',result)
        print({k:v for k,v in result.items() if k!='certificates'},'certificate_count',len(result['certificates']),flush=True);assert result['passed']
    else:
        from central_reservation_reference_v2 import run
        started=time.monotonic();result,arrays=run(case);checks={}
        with np.load(raw['trace_path']) as z:
            for key in ('positions','velocities','actions','nominal','safe','goals','swept_clearance','cone_residual','projection_rewrite','active_json'):
                a=np.asarray(arrays[key]);b=z[key];checks[key]=a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
        terminal={k:result[k]==raw[k] for k in ('success','collision','timeout','numerical_failure','outcome','episode_length')}
        proof=dict(utc=utc(),source=str(path),source_sha256=sha(path),trace_sha256=raw['trace_sha256'],passed=all(checks.values()) and all(terminal.values()),arrays=checks,terminal=terminal,seconds=time.monotonic()-started,original_seconds=raw['elapsed_seconds'],certificate_count=len(result.get('reservation_no_goal_certificates',[])),new_research_rollouts=0,canonical_case_unchanged=True)
        write(HERE/'CENTRAL_FIXED_CERTIFICATE_FULL_REPLAY.json',proof);print(proof,flush=True);assert proof['passed']

if __name__=='__main__':main()
