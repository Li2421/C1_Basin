#!/usr/bin/env python3
import csv,json,hashlib,time,shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
HERE=Path(__file__).resolve().parent
FAMS={
 'A':'affine_superbody','B':'superbody_with_cuts','C':'native_conditional_band','D':'rotated_asymmetric_slab',
 'E':'synthesized_families/affine_capsule_with_cuts','F':'synthesized_families/two_superbody_union','G':'synthesized_families/domain_minus_boundary_caps'}
def read(p):return list(csv.DictReader(open(p)))
def write(p,rows,fields=None):
 p=HERE/p
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(p,x):(HERE/p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def b(v):return str(v).lower()=='true'
def summarize_budget(path):
 r=read(path);out={}
 for budget in (16,24,32):
  q=[x for x in r if int(x['budget'])==budget];out[str(budget)]=dict(usable=sum(b(x['training_region_usable']) for x in q),states=len(q),median_independent_recall=float(np.median([float(x['independent_recall']) for x in q])),cached_false_inclusions=sum(int(x['cached_false_inclusion']) for x in q if x['cached_false_inclusion']!=''),median_diameter=float(np.median([float(x['retained_diameter']) for x in q])))
 return out
def parameter_rows():
 rows=[]
 for tag,path in FAMS.items():
  p=HERE/path/'fitted_parameters.csv'
  if not p.exists():continue
  for r in read(p):
   if not r.get('model') or r['model']=='null':continue
   m=json.loads(r['model']);center=np.array(m.get('center',m.get('anchor',[0,0,0])),float);sc=[]
   if 'axes' in m:sc=list(map(float,m['axes']))
   elif 'axes_tan' in m:
    width=float(np.array(m['upper'])[0]-np.array(m['lower'])[0]);sc=list(map(float,m['axes_tan']))+[width]
   elif m['family']=='affine_capsule_with_cuts':sc=[2*m['half_length']+2*m['radii'][0],m['radii'][1],m['radii'][2]]
   elif m['family']=='two_superbody_union':sc=[x for c in m['components'] for x in c['axes']]
   elif m['family']=='domain_minus_boundary_caps':sc=[c['radius'] for c in m['caps']]
   rows.append(dict(family_id=tag,family=m['family'],state_id=r['state_id'],scenario=r['scenario'],parameter_count=m.get('parameter_count',''),gamma=m.get('gamma',''),p=m.get('p',''),pieces=len(m.get('cuts',m.get('caps',m.get('components',[])))),center_norm=float(np.linalg.norm(center)),min_scale=min(sc) if sc else '',max_scale=max(sc) if sc else '',scale_condition=max(sc)/max(min(sc),1e-12) if sc else '',finite=bool(np.isfinite(np.array(sc+[float(np.linalg.norm(center))])).all())))
 return rows
def update_common_audit():
 p=HERE/'common_core_audit.csv';bak=HERE/'common_core_audit_pre_fresh.csv'
 if not bak.exists():shutil.copy2(p,bak)
 old=read(bak);fresh=read(HERE/'fresh_common_core_q64.csv');by=defaultdict(list)
 for r in fresh:by[r['eta_key']].append(r)
 out=[]
 for r in old:
  if r['eta_key'] in by:
   q=by[r['eta_key']];toy=sum(b(x['B63']) for x in q);r=dict(r,tested=12,B63=toy+4,toy_tested=8,toy_B63=toy,db_tested=4,db_B63=4)
  out.append(r)
 write('common_core_audit.csv',out)
 return out
def main():
 assert (HERE/'common_core_decision.json').exists()
 common=json.load(open(HERE/'common_core_decision.json'));core=common['any_shared_common_core'];audit=update_common_audit();gates={k:json.load(open(HERE/v/'gate.json')) for k,v in FAMS.items()};budgets={k:summarize_budget(HERE/v/'oracle_budget_performance.csv') for k,v in FAMS.items()}
 # No analytic family reached the cached screen, so no retained-set rollout was scientifically authorized.
 assert not any(g['cached_screen_pass'] for g in gates.values())
 classification='SHARED_COMMON_CORE_STRONGLY_SUPPORTED' if core else 'NO_SHARED_CONSERVATIVE_FAMILY_EVIDENCED'
 ready=False
 write('fresh_retained_manifest.csv',[dict(status='NOT_RUN_NO_FAMILY_PASSED_CACHED_GATE',candidate_family='',state_id='',eta='')])
 write('fresh_retained_q64.csv',[dict(status='NOT_RUN_NO_FAMILY_PASSED_CACHED_GATE',candidate_family='',state_id='',Q64='',B63='')])
 params=parameter_rows();write('parameter_consistency.csv',params)
 consistency={}
 for f in sorted(set(r['family'] for r in params)):
  q=[r for r in params if r['family']==f];consistency[f]=dict(states=len(q),finite=all(r['finite'] for r in q),median_scale_condition=float(np.median([float(r['scale_condition']) for r in q if r['scale_condition']!=''])) if any(r['scale_condition']!='' for r in q) else None,parameterization_well_conditioned=all(float(r['scale_condition'])<100 for r in q if r['scale_condition']!=''))
 # Compact hypothesis ledger.
 hyp=[]
 for tag,path in FAMS.items():
  g=gates[tag];hyp.append(dict(hypothesis_id=tag,mathematical_form=g['family'],current_status='REJECTED_CACHED_GATE',key_support={'usable_states':g['usable_states'],'median_diameter':g['median_retained_diameter']},key_failure={k:v for k,v in g['criteria'].items() if not v},unresolved_property='No fresh retained-set claim because cached gate failed',next_discriminating_test=None))
 hyp.append(dict(hypothesis_id='CORE',mathematical_form='single persistent eta shared across states/scenarios',current_status='SUPPORTED' if core else 'REJECTED_ON_FROZEN_PANEL',key_support=common['modes'],key_failure=None if core else 'No tested eta reached11/12 overall and7/8 Toy while4/4 DB',unresolved_property='Positive-volume common region not tested',next_discriminating_test=None))
 dump('hypothesis_status.json',hyp)
 search=dict(rounds=[
  {'round':'predeclared_A_D','outcome':'NO_PASS','best_usable':max(gates[x]['usable_states'] for x in 'ABCD'),'best_recall':max(gates[x]['median_independent_B63_recall'] for x in 'ABCD')},
  {'round':'synthesis_E_F','outcome':'NO_PASS','best_usable':max(gates[x]['usable_states'] for x in 'EF'),'best_recall':max(gates[x]['median_independent_B63_recall'] for x in 'EF'),'material_improvement_ge_005':False},
  {'round':'synthesis_G','outcome':'NO_PASS','best_usable':gates['G']['usable_states'],'best_recall':gates['G']['median_independent_B63_recall'],'material_improvement_ge_005':False},
  {'round':'oracle_48','outcome':'NO_PASS','any_pass':json.load(open(HERE/'oracle_heavy_diagnostic.json'))['any_cached_screen_pass']},
  {'round':'common_core_exact_transfer','outcome':'SUPPORTED' if core else 'NO_CROSS_SCENARIO_CORE','modes':common['modes']}
 ],scientific_saturation_met=not core,criterion='Two consecutive synthesis rounds yielded no new structural property and <5pp improvement;48-label diagnostic also produced no pass.' if not core else 'Special common-core criterion supersedes analytic-family saturation classification.')
 dump('search_history.json',search)
 selected=dict(selected=False,family=None,reason='No A-G analytic family passed cached gates; fresh retained validation was therefore not run.',common_core_special_case=core,ready_for_margin_loss_training=ready)
 dump('selected_family.json',selected)
 decision=dict(classification=classification,READY_FOR_MARGIN_LOSS_TRAINING=ready,shared_analytic_conservative_family_works=False,large_enough_for_margin_supervision=False,parameters_obtainable_without_full_mapping=False,new_scenario_requires='No supported shared form; current evidence cannot justify only theta changes.',scenarios=2,adequately_sampled_states=12,exact_Q64_reused=2374,new_exact_Q64=common['new_exact_Q64'],families={k:{'name':gates[k]['family'],'usable_states':gates[k]['usable_states'],'median_independent_recall':gates[k]['median_independent_B63_recall'],'median_transfer_coverage':gates[k]['median_transfer_coverage'],'median_retained_diameter':gates[k]['median_retained_diameter'],'cached_screen_pass':gates[k]['cached_screen_pass'],'failed_criteria':[x for x,v in gates[k]['criteria'].items() if not v]} for k in gates},oracle_budget_performance=budgets,common_core=common,parameter_consistency=consistency,scientific_saturation=search['scientific_saturation_met'],training_executed=False,next_scientific_step='Collect a small, source-diverse cross-scenario panel of verified local inner sets around scenario-specific robust cores, then test whether their normalized support functions share one parameterization; do not train margin loss yet.')
 dump('final_decision.json',decision)
 runtime=dict(prior_exact_Q64_reused=2374,panel_exact_Q64_reused=1968,new_exact_Q64=common['new_exact_Q64'],new_continuations=common['new_continuations'],reused_continuations=common['reused_continuations'],new_physical_steps=common['physical_steps'],rollout_critical_wall_seconds=common['critical_wall_seconds'],worker_wall_seconds_sum=common['worker_wall_seconds'],max_GPU_shards=2,max_CPU_threads=4,training_runs=0,offline_family_fit_seconds=sum(json.load(open(p)).get('runtime_seconds',0) for p in (HERE/'family_comparison.json',HERE/'synthesis_comparison.json',HERE/'synthesis_round2_comparison.json',HERE/'oracle_heavy_diagnostic.json')))
 dump('runtime_statistics.json',runtime)
 # Required mathematical handoff files explicitly state why no loss is authorized.
 (HERE/'selected_family_math.md').write_text('# Selected shared family\n\nNo analytic family was selected. Families A-D and synthesized E-G failed unchanged cached gates; therefore no mathematical set is asserted as a conservative cross-scenario label.\n')
 (HERE/'fitting_protocol.md').write_text('# Frozen fitting protocol\n\nSee `protocol.md`, `cv_folds.json`, `oracle_budget_subsamples.json`, and `oracle_revision.json`. Three scenario-stratified outer folds were used. Per-state theta used the same V1 sequential robust-anchor procedure in both scenarios with nested16/24/32 labels. Global exponent, erosion, and piece count were selected only on development states. A48-label diagnostic tested oracle heaviness without altering the primary gate.\n')
 (HERE/'retained_set_spec.md').write_text('# Retained-set status\n\nCandidate families used global homothetic erosion about an exact-B63 anchor. Membership was unit-tested to ensure every retained set is a subset of its full set. No candidate passed the cached shared-family gate, so no retained set is frozen for learning.\n')
 (HERE/'margin_loss_spec.md').write_text('# Margin-loss readiness\n\n`READY_FOR_MARGIN_LOSS_TRAINING = NO`. No certified shared retained family exists, so this task does not issue a deployable violation function or authorize controller training. Candidate batched violations were unit-tested only as experimental geometry code.\n')
 lines=['# Shared conservative success-set family audit','',f'**Classification:** `{classification}`  ',f'**READY_FOR_MARGIN_LOSS_TRAINING:** `NO`','',f'Two compatible scenarios and12 adequately sampled states were analyzed from {2374} reused exact-Q64 tuples; the shared-core check added {common["new_exact_Q64"]} exact-Q64 state–eta evaluations. No controller was trained.','', '## Family results','', '|ID|Family|Usable /12|Median independent B63 recall|Median transfer coverage|Median retained diameter|Cached gate|','|---|---|---:|---:|---:|---:|---|']
 for k in FAMS:
  g=gates[k];lines.append(f'|{k}|{g["family"]}|{g["usable_states"]}/12|{g["median_independent_B63_recall"]:.3f}|{(g["median_transfer_coverage"] or 0):.3f}|{g["median_retained_diameter"]:.3f}|FAIL|')
 lines += ['', 'A-D failed the unchanged gate. E (affine capsule), F (two-superbody union), and G (domain minus boundary-open caps) were the two preregistered synthesis refinement rounds; neither improved the joint reliability/coverage result by five percentage points. All tested families also failed at48 labels/state, so the result is not explained solely by the <=32-label target.','', '## Common-core result','']
 if core:lines.append('At least one DB-common eta passed the exact cross-scenario criterion (>=11/12 overall, >=7/8 Toy,4/4 DB). This supports a shared robust point, not a positive-volume margin label.')
 else:lines.append('None of the three eta that were B63 on all4 Double-Bottleneck states reached the cross-scenario threshold after exact evaluation on all8 Toy states. Existing Toy-common coordinates were already0/4 B63 on Double-Bottleneck. Scenario-specific common cores exist on the sampled panels, but no cross-scenario common eta is supported.')
 lines += ['', '## Answers','', '- **Does one shared analytic conservative success-set family work across compatible scenarios?** No family passed the reliability, state-coverage, moderate-recall, multimodal-support, and size gates together.', '- **Is it large enough for margin-loss supervision?** No.', '- **Can its parameters be obtained without remapping the full Basin?** Not under the tested16/24/32-label protocol;48 labels also did not produce a passing family.', '- **Would a new scenario need only new parameters?** Current evidence is insufficient; assuming only theta changes is not justified.', '', 'The single next step is to collect a small cross-scenario panel of verified local inner sets around each scenario-specific robust core and compare their normalized support functions before any margin-loss training.','']
 (HERE/'final_report.md').write_text('\n'.join(lines))
 # Final working memory and ledger.
 dump('working_state.json',dict(status='COMPLETE',completed_stages=['evidence freeze','A-D CV','two synthesis rounds','48-label oracle diagnostic','common-core exact transfer','final integrity audit'],current_stage='FINAL_HANDOFF',classification=classification,READY_FOR_MARGIN_LOSS_TRAINING=False,training=False,next_action='None; do not train in this task.',authoritative_files=['final_decision.json','final_report.md','manifest.json']))
 # Mark ledger terminal.
 ledger=read(HERE/'experiment_ledger.csv')
 for r in ledger:
  if r['status']=='RUNNING':r['status']='COMPLETE';r['notes']=(r['notes']+'; exact results aggregated').strip('; ')
 write('experiment_ledger.csv',ledger)
 # Manifest is last; exclude itself.
 files=sorted(p for p in HERE.rglob('*') if p.is_file() and not any(x in p.parts for x in ('runs','raw','logs','plans','__pycache__')) and p.name!='manifest.json')
 dump('manifest.json',dict(experiment='ORTHOFLOW3_SHARED_CONSERVATIVE_SUCCESS_SET_FAMILY_V1',created_unix=time.time(),authoritative_basis_sha256='51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38',controller_training=False,classification=classification,READY_FOR_MARGIN_LOSS_TRAINING=False,files={str(p.relative_to(HERE)):sha(p) for p in files}))
 print(json.dumps({'classification':classification,'core':core,'ready':False,'new_exact':common['new_exact_Q64']}))
if __name__=='__main__':main()
