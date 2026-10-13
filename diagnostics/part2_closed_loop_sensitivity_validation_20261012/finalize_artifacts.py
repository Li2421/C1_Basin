"""Join derivative evidence, write provenance index and machine-readable verdicts."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'diagnostics/placement_boundary_mechanism_audit_v1'
HARD = Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
FIELD = Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')

def readcsv(name):
    with (HERE/name).open(newline='') as f:
        return list(csv.DictReader(f))

def main():
    path=HERE/'jacobian_validation.csv'
    rows=readcsv('jacobian_validation.csv')
    rows=[r for r in rows if r['layer'] not in ('S_heldout','S_scale')]
    fields=list(rows[0])
    for r in readcsv('trajectory_validation.csv'):
        if r['layer'] not in ('S_heldout','S_scale'):continue
        rows.append({**dict.fromkeys(fields,''), 'scene':r['scene'],'uid':r['uid'],
            'placement':r['placement'],'layer':r['layer'],'epsilon':r['epsilon'],
            'epsilon_index':r['step'],'abs_error':r['absolute_error'],
            'rel_error':r['relative_error'],'cosine':r['cosine'],
            'formula':'closed-loop H80 finite-difference sensitivity; see trajectory_validation.csv'})
    old=json.loads((OLD/'heldout_predictions.json').read_text())
    for record in old:
        if record['center']!='zero' or not record['uid'].endswith(('820013','920013')):continue
        for r in record['trajectory']:
            rows.append({**dict.fromkeys(fields,''),'scene':record['scene'],'uid':record['uid'],
                'placement':record['chain'],'layer':'S_heldout','epsilon':record['epsilon'],
                'epsilon_index':r['step'],'abs_error':r['absolute_error'],
                'rel_error':r['relative_error'],'cosine':r['cosine'],
                'formula':'reused H80 held-out trajectory prediction; prior rho definition excluded'})
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    verdict={
      'schema':'part2_closed_loop_sensitivity_validation_v1',
      'scope':'Toy Give-Way and Ring Exchange, frozen controllers, local fixed-noise derivatives; proxy rho distinct from B15',
      'overall':'PARTIAL',
      'ring':{'M_G':'PARTIAL','P_Q':'PARTIAL','K':'PARTIAL','S':'PARTIAL','rho_local':'PARTIAL',
              'regular_boundary':'PASS','gain_loss_mechanism':'PARTIAL','actual_b15_prediction':'INSUFFICIENT_EVIDENCE',
              'verdict':'PARTIAL'},
      'toy':{'M_G':'PARTIAL','P_Q':'NONSMOOTH','K':'PARTIAL','S':'NONSMOOTH','rho_local':'NONSMOOTH',
             'regular_boundary':'INSUFFICIENT_EVIDENCE','gain_loss_mechanism':'PARTIAL',
             'actual_b15_prediction':'INSUFFICIENT_EVIDENCE','verdict':'NONSMOOTH'},
      'combined':{'analytic_chain_all_four':'PARTIAL','closed_loop_recurrence':'PARTIAL',
                  'proxy_label_seed0':'PASS','sampled_gain_loss':'PASS',
                  'global_boundary_or_b15_prediction':'INSUFFICIENT_EVIDENCE','verdict':'PARTIAL'},
      'limitations':['K uses finite-differenced frozen-policy and basis terms; no full independent symbolic K',
                     'Q uses implicit KKT with finite-differenced CBF geometry',
                     'weakly active zero-centered projection is nonsmooth',
                     'H80 local continuations do not determine full-horizon binary success',
                     'B15 candidate sampling and Sobol bounded-box volume have different denominators']}
    (HERE/'verification.json').write_text(json.dumps(verdict,indent=2,ensure_ascii=False)+'\n')
    evidence={
      'source_repositories':{'workspace':str(ROOT),'field_poc':str(FIELD),'hard_field':str(HARD)},
      'historical':{
        'Q16_B15_results':str(HARD/'results'),
        'Q16_B15_database':str(HARD/'rollout.sqlite'),
        'TT_FF_H80_arrays':str(OLD/'trajectory_sensitivity_arrays.npz'),
        'TT_FF_H80_results':str(OLD/'results_h80'),
        'H1_H4_root_and_curvature':str(ROOT/'diagnostics/placement_strong_hypotheses_v1'),
        'Sobol64_and_128':str(ROOT/'diagnostics/placement_strong_hypotheses_v1')},
      'new_short_continuations':{'count':56,'horizon':80,'directory':str(HERE/'trajectory_results')},
      'new_full_continuations':0,
      'generated':{}}
    trajectory_files=sorted((HERE/'trajectory_results').glob('*'))
    evidence['new_short_continuations']['file_count']=len(trajectory_files)
    evidence['new_short_continuations']['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in trajectory_files if p.is_file()}
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.name not in ('cache_index.json','final_report.md'):
            evidence['generated'][p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (HERE/'cache_index.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')
    print('jacobian rows',len(rows),'generated indexed',len(evidence['generated']))

if __name__=='__main__':main()
