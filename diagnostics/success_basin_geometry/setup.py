"""Freeze SBGA states, parameter domain, thresholds, seeds and gates."""
import json
import hashlib
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):(HERE/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def main():
    if (HERE/'protocol.json').exists():raise RuntimeError('Protocol already locked')
    old=ROOT/'diagnostics/true_q_geometry'; audit=ROOT/'diagnostics/astra_true_q_audit'
    catalog=json.loads((old/'raw/q_map/state_catalog.json').read_text())
    prior=json.loads((audit/'replication_plan.json').read_text())
    states=HERE/'states';states.mkdir()
    selected=[]
    for sid in ['D1_pair231','D2_pair228','D4_pair227']:
        c=next(s for s in catalog if s['state_id']==sid)
        src=old/'raw/q_map'/c['state_file']; assert sha(src)==c['state_file_sha256']
        with np.load(src) as data:np.savez_compressed(states/f'{sid}.npz',**dict(data))
        selected.append({**c,'state_file':f'states/{sid}.npz','state_file_sha256':sha(states/f'{sid}.npz')})
    # A genuine same-source temporal sequence, fixed before observing SBGA outcomes.
    source=ROOT/'diagnostics/cl_fhcb_qualification/raw/stage1'
    source_manifest=json.loads((source/'manifest.json').read_text())
    rec=next(r for r in source_manifest['records'] if r['pair_id']==227 and r['phi_name']=='goal025')
    with np.load(source/rec['relative_path']) as d:trace=dict(d)
    goals=np.array([[1.09,0],[-1.09,0]])
    for offset in [8,2]:
        step=len(trace['event'])-int(offset/.05);sid=f'P227_offset{offset}s'
        seq=np.concatenate([trace['positions_before'][:1],trace['positions_after']],axis=0)
        state={'positions':seq[step],'last_velocity':trace['last_velocity_before'][step],
            'step':step,'history_start_step':step-40,'error_history':np.linalg.norm(goals[None]-seq[step-40:step+1],axis=-1),
            'candidate_since':trace['candidate_since_before'][step]}
        np.savez_compressed(states/f'{sid}.npz',**state)
        selected.append({'state_id':sid,'pair_id':227,'start_step':step,'nominal_offset_seconds':offset,
            'source_outcome':'deadlock','source_phi':'goal025','source_trace':rec['id'],
            'state_file':f'states/{sid}.npz','state_file_sha256':sha(states/f'{sid}.npz')})
    protocol={'locked_at_utc':datetime.now(timezone.utc).isoformat(),'study':'SBGA',
        'state_catalog':selected,'primary_states':['D1_pair231','D2_pair228','D4_pair227'],
        'temporal_states':['P227_offset8s','D4_pair227','P227_offset2s'],
        'phi_order':['goal','safe','relative'],'reference':[0.,0.,0.],
        'main_grid':{'goal':np.linspace(-1,1,9).tolist(),'relative':np.linspace(-1,1,9).tolist(),'safe':0.,
          'rationale':'Goal and relative directions influential in old audit. 9x9 with 16 seeds allows an exact lower bound above .75 and fits ~4000 rollouts.',
          'seeds':list(range(94001001,94001017)),'new_rollouts':3888},
        'success_basin':{'definition':'One-sided 95% Clopper-Pearson lower bound on Q_S >= .75',
          'threshold':.75,'sensitivity_threshold':.70,'coarse_note':'With n=16,16 successes qualify,15 do not. Descriptive cell bounds not simultaneous across grid.',
          'adjacency':'4-neighbor; compare 8-neighbor as sensitivity','minimum_component_size_for_region':2},
        'refinement':{'gate':'Any observed success on a primary state.',
          'rule':'For each component choose min-norm and max-success cells; unique representatives, at most 4/state. Map 3x3 neighborhoods with half-step .125 and 8 seeds; cap 108 new cells.',
          'seeds':list(range(94002001,94002009))},
        'validation':{'rule':'Independently validate representative and minimum-distance candidates, max 6/state; 32 fresh seeds/arm. If lower Q_S bound straddles threshold, append 32 seeds only for those candidates.',
          'seeds':list(range(94003001,94003033)),'extension_seeds':list(range(94003033,94003065))},
        'no_success_gate':{'rule':'If primary map has no success, test all new points of {-1,0,1}^3 (safe nonzero) at 8 seeds/state; plus a same-source temporal 5x5 goal-relative slice at offsets 8s and 2s.',
          'seeds':list(range(94004001,94004009)),'temporal_axis':[-1,-.5,0,.5,1]},
        'interpolation':{'lambda':[0,.25,.5,.75,1], 'seed_count':32,
          'gate':'Two independently validated components within same state; preserve endpoints. No averaging inference without this test.'},
        'temporal_success_gate':{'rule':'If primary success found, evaluate all primary success cells and immediate grid neighbors plus origin on other two same-source offsets, at 8 fresh seeds; cap 32 cells/offset.',
          'seeds':list(range(94005001,94005009))},
        'cross_state':'Apply validated representatives unchanged to other primary states with 32 fresh seeds; reuse exact compatible state/phi/seed only.',
        'budget':{'stage_A':3888,'conditional_stage_B_total_ceiling':10000,'unconditional_extras_no_success':832},
        'family_limit':'Finite box and slices do not prove nonexistence over all R^3 or all possible controls.',
        'benchmark':{'state_id':'D2_pair228','phis':[[0,0,0],[-.5,0,.5],[.5,0,-.5],[.5,0,.5]],
          'seeds':[94000001,94000002,94000003,94000004],'backends':['cpu','gpu'],'exclude_from_topology':True},
        'environment':prior['environment'],'checkpoint':prior['source_checkpoint'],
        'source_manifests':{str(p.relative_to(ROOT)):sha(p) for p in [old/'manifest.json',audit/'manifest.json']}}
    write('protocol.json',protocol)
    jobs=[]
    for sid in protocol['primary_states']:
        phis=sorted([(g,0.,r) for g in protocol['main_grid']['goal'] for r in protocol['main_grid']['relative']],key=lambda x:(sum(v*v for v in x),x))
        for cell,phi in enumerate(phis):
            for seed in protocol['main_grid']['seeds']:jobs.append({'state_id':sid,'phi':phi,'seed':seed,'cell_id':f'{sid}_g{phi[0]:+.3f}_r{phi[2]:+.3f}'})
    write('main_jobs.json',jobs)
    bench=[]
    for i,phi in enumerate(protocol['benchmark']['phis']):
        for seed in protocol['benchmark']['seeds']:bench.append({'state_id':'D2_pair228','phi':phi,'seed':seed,'cell_id':f'bench{i}'})
    write('benchmark_jobs.json',bench)
    print({'primary_states':protocol['primary_states'],'map_rollouts':len(jobs),'protocol_sha256':sha(HERE/'protocol.json')})

if __name__=='__main__':main()
