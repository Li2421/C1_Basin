"""Freeze SBMA protocol and Phase-A jobs before observing new outcomes."""
import hashlib
import inspect
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'diagnostics/success_basin_geometry'
SYSROOT = Path('/home/zhihan/research/02_C1_Toy_GiveWay')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(name, value):
    (HERE/name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def main():
    old = json.loads((OLD/'protocol.json').read_text())
    primary = [x for x in old['state_catalog'] if x['state_id'] in
               {'D1_pair231','D2_pair228','D4_pair227'}]
    assert len(primary) == 3
    goal = [.5, .75, 1., 1.25]
    safe = [-.5, -.25, 0., .25, .5]
    relative = [0., .25, .5, .75]
    seeds = list(range(95101001, 95101017))
    protocol = {
      'study':'SBMA', 'locked_at_utc':datetime.now(timezone.utc).isoformat(),
      'notation':{'diagnostic_parameter':'eta','future_network_parameter':'theta'},
      'policy_family':{
        'dimension':3, 'coordinate_order':['goal_feedback','safe_feedback','relative_feedback'],
        'formula':'G_eta(z)=eta_goal*bounded(goal-position)+eta_safe*u_safe+eta_relative*bounded(self-other)',
        'deterministic_corrector':True, 'flow_randomness':'fresh frozen Flow Gaussian sample at every physical step',
        'same_eta_full_continuation':True, 'recomputed_every_step':True,
      },
      'primary_states':primary,
      'state_files_are_read_only_prior_artifacts':True,
      'environment':old['environment'], 'checkpoint':old['checkpoint'],
      'solver':{
        'outcome_backend':'clarabel_exact_socp',
        'problem':'minimize 0.5||u-target||^2 subject to A u >= b and ||u_i||_2 <= 0.5 for i=1,2',
        'no_slack_or_constraint_change':True,
        'current_source':str(SYSROOT/'single_integrator/cbf.py'),
        'current_source_sha256':sha(SYSROOT/'single_integrator/cbf.py'),
        'legacy_git_object':'cc8579e:single_integrator/cbf.py',
        'legacy_source_sha256':'f127d09cddb490c9367630fee0d2a8a8434fcb2579ba363ac312ee8ca1159eda',
        'existing_same_input_regression':str(SYSROOT/'results/giveway_projection_backend_fix_v1/stage_d/projection_regression.json'),
        'existing_regression_sha256':sha(SYSROOT/'results/giveway_projection_backend_fix_v1/stage_d/projection_regression.json'),
      },
      'classification':{
        'success_cell':'no numerical UNKNOWN, n>=16, and one-sided exact 95% LCB(Q_S)>=0.80',
        'q_success':.80, 'confidence':.95,
        'failure_cell':'no numerical UNKNOWN, n>=16, and one-sided exact 95% UCB(Q_S)<=0.20',
        'unknown_cell':'any projection/implementation failure, fewer than 16 trials, or neither confidence rule',
        'fixed_before_phase_a_outcomes':True,
      },
      'phase_a_design':{
        'type':'structured 3D lattice around eta_success=(1,0,.25)',
        'axes':{'goal':goal,'safe':safe,'relative':relative},
        'points_per_state':len(goal)*len(safe)*len(relative),
        'seeds':seeds, 'seeds_per_point':16,
        'two_dimensional_slices':['safe=0 goal-relative','relative=.25 goal-safe','goal=1 safe-relative'],
        'adjacency':'axis neighbors on the structured lattice',
        'new_continuations':len(primary)*len(goal)*len(safe)*len(relative)*len(seeds),
      },
      'fresh_validation_seeds':list(range(95102001,95102065)),
      'path_lambda':[i/10 for i in range(11)],
      'margin_directions':['+/-goal','+/-safe','+/-relative'],
      'source_hashes':{
        'old_protocol':sha(OLD/'protocol.json'),
        'old_manifest':sha(OLD/'manifest.json'),
        'environment':sha(SYSROOT/'single_integrator/environment.py'),
        'corrector':sha(ROOT/'diagnostics/cl_fhcb/closed_loop.py'),
        'flow_checkpoint':sha(old['checkpoint']),
      },
      'limits':['Finite sampled box cannot prove mathematical connectedness or disconnectedness.',
                'D1/D2/D4 are similar outcome-selected failing states, not a population sample.'],
    }
    write('protocol.json', protocol)
    for state in primary:
        jobs=[]
        for eta in product(goal,safe,relative):
            for seed in seeds:
                jobs.append({'state_id':state['state_id'],'eta':list(eta),'seed':seed,
                             'cell_id':'_'.join(f'{x:+.3f}' for x in eta)})
        write(f"phase_a_{state['state_id']}_jobs.json",jobs)
    print(json.dumps({'protocol_sha256':sha(HERE/'protocol.json'),
                      'phase_a_new_continuations':protocol['phase_a_design']['new_continuations'],
                      'max_physical_steps':sum((old['environment']['max_steps']-s['start_step'])*
                          len(goal)*len(safe)*len(relative)*len(seeds) for s in primary)}))

if __name__=='__main__': main()
