#!/usr/bin/env python3
import csv,json,hashlib,subprocess
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent;D=HERE.parent
def read(p):return list(csv.DictReader(open(p)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 required=['exact_q64_inventory.csv','scenario_state_inventory.csv','cv_folds.json','oracle_budget_subsamples.json',
  'affine_superbody/fitted_parameters.csv','affine_superbody/cv_results.csv','affine_superbody/retained_metrics.csv','affine_superbody/gate.json',
  'superbody_with_cuts/fitted_parameters.csv','superbody_with_cuts/cv_results.csv','superbody_with_cuts/retained_metrics.csv','superbody_with_cuts/gate.json',
  'native_conditional_band/fitted_parameters.csv','native_conditional_band/cv_results.csv','native_conditional_band/retained_metrics.csv','native_conditional_band/gate.json',
  'rotated_asymmetric_slab/fitted_parameters.csv','rotated_asymmetric_slab/cv_results.csv','rotated_asymmetric_slab/retained_metrics.csv','rotated_asymmetric_slab/gate.json',
  'property_diagnosis.md','search_history.json','fresh_retained_manifest.csv','fresh_retained_q64.csv','common_core_audit.csv','parameter_consistency.csv',
  'selected_family.json','selected_family_math.md','fitting_protocol.md','retained_set_spec.md','margin_loss_spec.md','final_decision.json','runtime_statistics.json','final_report.md',
  'working_state.json','evidence_index.json','hypothesis_status.json','experiment_ledger.csv','geometry_unit_tests.json']
 missing=[p for p in required if not (HERE/p).exists() or (HERE/p).stat().st_size==0]
 inv=read(HERE/'exact_q64_inventory.csv');panel=read(HERE/'scenario_state_inventory.csv');folds=json.load(open(HERE/'cv_folds.json'))['folds'];subs=json.load(open(HERE/'oracle_budget_subsamples.json'))['states'];dec=json.load(open(HERE/'final_decision.json'));core=json.load(open(HERE/'common_core_decision.json'));unit=json.load(open(HERE/'geometry_unit_tests.json'));rt=json.load(open(HERE/'runtime_statistics.json'))
 held=[s for f in folds for s in f['heldout']];panelids={r['state_id'] for r in panel};nested=all(set(v['16'])<=set(v['24'])<=set(v['32']) and list(v)==['16','24','32'] for v in subs.values())
 raw=sum(1 for p in (HERE/'raw/common_core').glob('shard*.jsonl') for _ in open(p));fresh=read(HERE/'fresh_common_core_q64.csv')
 gates=[json.load(open(HERE/p)) for p in ('affine_superbody/gate.json','superbody_with_cuts/gate.json','native_conditional_band/gate.json','rotated_asymmetric_slab/gate.json','synthesized_families/affine_capsule_with_cuts/gate.json','synthesized_families/two_superbody_union/gate.json','synthesized_families/domain_minus_boundary_caps/gate.json')]
 checks={
  'required_artifacts_present':not missing,
  'basis_sha256':sha(D/'double_bottleneck_eta_basis_redesign/tools/bases.py')=='51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38',
  'inventory_exact_unique':len(inv)==len({(r['state_id'],r['eta_key']) for r in inv})==2374 and all(int(r['trials'])==64 for r in inv),
  'panel_scope':len(panelids)==12 and set(r['scenario'] for r in panel)=={'ToyGiveWay_2A','DoubleBottleneck_4A'},
  'outer_folds':Counter(held)==Counter({s:1 for s in panelids}) and all({next(r['scenario'] for r in panel if r['state_id']==s) for s in f['heldout']}=={'ToyGiveWay_2A','DoubleBottleneck_4A'} for f in folds),
  'oracle_nested_and_exact':nested and set(subs)==panelids and all(len(v[str(b)])==b for v in subs.values() for b in (16,24,32)),
  'all_family_cached_gates_fail':not any(g['cached_screen_pass'] for g in gates),
  'fresh_retained_correctly_not_run':read(HERE/'fresh_retained_manifest.csv')[0]['status']=='NOT_RUN_NO_FAMILY_PASSED_CACHED_GATE',
  'common_core_batch_complete':raw==1536 and len(fresh)==24 and all(int(r['successes'])<=64 for r in fresh),
  'common_core_special_case_false':not core['any_shared_common_core'] and all(not m['shared_core_criterion'] for m in core['modes']),
  'geometry_unit_tests':unit['all_pass'] and unit['models']==84,
  'no_training':dec['training_executed'] is False and rt['training_runs']==0,
  'classification':dec['classification']=='NO_SHARED_CONSERVATIVE_FAMILY_EVIDENCED' and dec['READY_FOR_MARGIN_LOSS_TRAINING'] is False,
  'runtime_consistency':rt['new_exact_Q64']==24 and rt['new_continuations']==1536 and rt['new_physical_steps']==sum(int(r['physical_steps']) for r in fresh)
 }
 out=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,missing=missing,required_count=len(required),classification=dec['classification'])
 (HERE/'completion_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
 assert all(checks.values()),out
if __name__=='__main__':main()
